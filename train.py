"""Training script. Usage: accelerate launch train.py -opt options/scd_minecraft.yml"""
import argparse
import gc
import os
import shutil
import time

import copy
import torch
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from scd.utils.logger_util import MessageLogger, dict2str, reduce_loss_dict, set_path_logger, setup_wandb


def train(args):
    
    from scd.data import build_dataset
    from scd.trainers import build_trainer
    def _log_ema_alignment(accelerator, train_pipeline, tag):
        if train_pipeline.ema is None:
            return
        model = accelerator.unwrap_model(train_pipeline.model)
        max_diff = 0.0
        max_name = None
        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = train_pipeline.ema.shadow.get(name)
                if shadow is None:
                    continue
                diff = (shadow.to(param.device) - param.detach()).abs().max().item()
                if diff > max_diff:
                    max_diff = diff
                    max_name = name
        accelerator.print(f'[EMA sanity check][{tag}] max |ema-param| diff: {max_diff:.6e} ({max_name})')

    # load config
    opt = OmegaConf.to_container(OmegaConf.load(args.opt), resolve=True)

    def sanitize_ddp_env(default_port="19043"):
        ddp_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        is_multiproc = any(k in os.environ for k in ddp_keys)

        if not is_multiproc or os.environ.get("WORLD_SIZE", "1") == "1":
            for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "NODE_RANK",
                      "SLURM_PROCID", "SLURM_NTASKS", "PMI_SIZE", "PMI_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                os.environ.pop(k, None)
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", default_port)
        else:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", default_port)
            assert "RANK" in os.environ and "WORLD_SIZE" in os.environ, "Missing DDP ranks"
            r, w = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
            assert 0 <= r < w, f"Inconsistent rank/world_size: {r}/{w}"

    sanitize_ddp_env(default_port="19043")
    os.environ.setdefault("NCCL_TIMEOUT_MS", "2000000")

    accelerator = Accelerator(mixed_precision=opt['mixed_precision'])

    # set experiment dir
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, opt, is_train=True)

    # get logger
    logger = get_logger('scd', log_level='INFO')
    logger.info(accelerator.state)
    logger.info(dict2str(opt))

    # get wandb
    if accelerator.is_main_process and opt['logger'].get('use_wandb', False):
        wandb_logger = setup_wandb(name=opt['name'], save_dir=opt['path']['log'])

        if getattr(wandb, 'run', None) is not None:
            wandb.config.update(opt)
            code_root = os.path.dirname(os.path.abspath(__file__))
            def _include_fn(path: str) -> bool:
                exclude = ['__pycache__', '.git', 'wandb', 'experiments', 'results']
                if any(p in path for p in exclude):
                    return False
                return path.endswith(('.py', '.yml', '.yaml', '.sh', '.json', '.txt'))
            wandb.run.log_code(root=code_root, include_fn=_include_fn)
    else:
        wandb_logger = None

    # If passed along, set the training seed now.
    if opt.get('manual_seed') is not None:
        set_seed(opt['manual_seed'] + accelerator.process_index)

    # load trainer pipeline
    train_pipeline = build_trainer(opt['train'].get('train_pipeline', 'SCDTrainer'))(**opt['models'], accelerator=accelerator)

    # set optimizer
    train_opt = opt['train']
    optim_type = train_opt['optim_g'].pop('type')
    assert optim_type == 'AdamW', 'only support AdamW now'
    base_lr = train_opt['optim_g']['lr']
    optimizer = torch.optim.AdamW(
        train_pipeline.get_params_to_optimize(train_opt['param_names_to_optimize'], base_lr),
        **train_opt['optim_g'],
    )

    # Get the training dataset
    trainset_cfg = opt['datasets']['train']
    train_dataset = build_dataset(trainset_cfg)
    accelerator.print(
        f"Built train dataset ({train_dataset.__class__.__name__} | len={len(train_dataset)})"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=trainset_cfg['batch_size_per_gpu'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

    if opt['datasets'].get('sample'):
        sampleset_cfg = opt['datasets']['sample']
        sample_dataset = build_dataset(sampleset_cfg)
        accelerator.print(
            f"Built sample dataset ({sample_dataset.__class__.__name__} | len={len(sample_dataset)})"
        )
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sampleset_cfg['batch_size_per_gpu'], shuffle=False)
    else:
        sample_dataloader = None
    # Prepare learning rate scheduler in accelerate config
    total_batch_size = opt['datasets']['train']['batch_size_per_gpu'] * accelerator.num_processes

    num_training_steps = total_iter = opt['train']['total_iter']
    num_warmup_steps = opt['train']['warmup_iter']

    if opt['train']['lr_scheduler'] == 'constant_with_warmup':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
        )
    elif opt['train']['lr_scheduler'] == 'cosine_with_warmup':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps * accelerator.num_processes,
        )
    else:
        raise NotImplementedError

    # Prepare everything with our `accelerator`.
    train_pipeline.model, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        train_pipeline.model, optimizer, train_dataloader, sample_dataloader, lr_scheduler)

    accelerator.print(
        f"Trainer prepared with model={train_pipeline.model.__class__.__name__}, "
        f"vae={getattr(train_pipeline, 'vae', None).__class__.__name__ if getattr(train_pipeline, 'vae', None) else 'None'}"
    )

    # set ema after prepare everything: sync the model init weight in ema
    train_pipeline.set_ema_model(ema_decay=opt['train'].get('ema_decay'))

    # Train!
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f"  Instantaneous batch size per device = {opt['datasets']['train']['batch_size_per_gpu']}")
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Total optimization steps = {total_iter}')
    steps_per_epoch = len(train_dataset) // total_batch_size

    if opt['path'].get('pretrain_network', None):
        load_path = opt['path'].get('pretrain_network')
    else:
        load_path = opt['path']['models']

    global_step = resume_checkpoint(args, accelerator, load_path, train_pipeline)
    if accelerator.is_main_process and train_pipeline.ema is not None:
        _log_ema_alignment(accelerator, train_pipeline, 'after-resume')

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    msg_logger = MessageLogger(opt, global_step)
    last_log_time = time.perf_counter()

    initial_inference_done = False
    last_eval_step = -1

    def _run_validation(current_step: int, run_tag: str, current_epoch_val: float = None) -> None:
        nonlocal initial_inference_done, last_log_time, last_eval_step
        if sample_dataloader is None:
            return

        guidance_scale_list = opt['val']['sample_cfg']['guidance_scale']
        if not isinstance(guidance_scale_list, list):
            guidance_scale_list = [guidance_scale_list]

        num_steps_opt = opt['val']['sample_cfg'].get('num_inference_steps')
        if isinstance(num_steps_opt, list):
            num_steps_list = num_steps_opt
        else:
            num_steps_list = [num_steps_opt]

        for scale_idx, scale in enumerate(guidance_scale_list):
            try:
                for steps_idx, steps in enumerate(num_steps_list):
                    if train_pipeline.ema is not None:
                        _log_ema_alignment(accelerator, train_pipeline, f'{run_tag}-scale-{scale}-steps-{steps}')

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    sample_time = 0.0
                    sample_start_time = time.perf_counter()
                    debug_flag = (not initial_inference_done) and (current_step <= 1000) and run_tag == 'loop'
                    effective_steps = 2 if debug_flag else steps
                    train_pipeline.sample(
                        sample_dataloader,
                        opt,
                        wandb_logger=wandb_logger,
                        global_step=current_step,
                        debug=debug_flag,
                        guidance_scale=scale,
                        num_inference_steps=steps,
                    )
                    sample_time = time.perf_counter() - sample_start_time

                    accelerator.wait_for_everyone()
                    if train_pipeline.ema is not None:
                        _log_ema_alignment(accelerator, train_pipeline, f'after-{run_tag}-scale-{scale}-steps-{steps}')

                    if accelerator.is_main_process and 'eval_cfg' in opt['val']:
                        result_dict = train_pipeline.eval_performance(
                            opt,
                            guidance_scale=scale,
                            global_step=current_step,
                            num_inference_steps=effective_steps,
                        )

                        if wandb_logger and initial_inference_done:
                            wandb_log_dict = {
                                f'eval/{k}_{scale}_steps{effective_steps}': v for k, v in result_dict.items()
                            }
                            if current_epoch_val is not None:
                                wandb_log_dict['eval/epoch'] = round(current_epoch_val, 3)
                            wandb_log_dict['eval/global_step'] = current_step
                            wandb_log_dict[f'eval/eval_time_sec_{scale}_steps{effective_steps}'] = sample_time
                            wandb_logger.log(wandb_log_dict)

            except Exception:
                import traceback
                accelerator.print(f"Exception during eval scale={scale}:\n{traceback.format_exc()}")
                raise
            finally:
                accelerator.wait_for_everyone()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                pass

        initial_inference_done = True
        last_eval_step = current_step
        last_log_time = time.perf_counter()

    first_batch_logged = False
    while global_step < total_iter:
        batch = next(train_data_yielder)
        if accelerator.is_main_process and not first_batch_logged:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    accelerator.print(f"  {key}: {tuple(value.shape)} {value.dtype}")
            first_batch_logged = True
        decoder_multi_batch = (
            opt.get('decoder_multi_batch')
            or opt.get('models', {}).get('decoder_multi_batch')
            or opt.get('model', {}).get('decoder_multi_batch')
            or 1
        )
        decoder_multi_batch = int(decoder_multi_batch)

        accumulated_loss = None
        aggregated_logs = {}

        for repeat_idx in range(decoder_multi_batch):
            step_logs = train_pipeline.train_step(batch, iters=global_step)
            step_loss = step_logs['total_loss']

            accumulated_loss = step_loss if accumulated_loss is None else accumulated_loss + step_loss

            for key, value in step_logs.items():
                if key == 'total_loss':
                    continue
                if isinstance(value, torch.Tensor):
                    value_detached = value.detach()
                else:
                    value_detached = torch.as_tensor(value, device=step_loss.device)
                if key in aggregated_logs:
                    aggregated_logs[key] = aggregated_logs[key] + value_detached
                else:
                    aggregated_logs[key] = value_detached

        accumulated_loss = accumulated_loss / decoder_multi_batch
        accelerator.backward(accumulated_loss)

        loss_dict = {'total_loss': accumulated_loss.detach()}
        for key, value in aggregated_logs.items():
            loss_dict[key] = (value / decoder_multi_batch).detach()

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(train_pipeline.model.parameters(), opt['train']['max_grad_norm'])

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        # --- end of iteration ---

        # Checks if the accelerator has performed an optimization step behind the scenes
        
        if accelerator.sync_gradients:

            if train_pipeline.ema is not None:
                train_pipeline.ema.step(accelerator.unwrap_model(train_pipeline.model))

            global_step += 1

            if global_step % opt['logger']['print_freq'] == 0:
                log_dict = reduce_loss_dict(accelerator, loss_dict)
                log_vars = {'iter': global_step}
                log_vars.update({'lrs': lr_scheduler.get_last_lr()})
                log_vars.update(log_dict)
                
                # Calculate current epoch as floating point value
                steps_per_epoch = len(train_dataset) // total_batch_size
                current_epoch = global_step / steps_per_epoch if steps_per_epoch > 0 else 0.0
                log_vars['epoch'] = round(current_epoch, 3)
                
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                log_vars['peak_mem (MB)'] = round(peak_mem, 2)
                
                msg_logger(log_vars)

                elapsed_between_logs_sec = time.perf_counter() - last_log_time
                # convert tensors to scalars for robust logging
                wandb_log_dict = {}
                for k, v in log_vars.items():
                    val = v
                    if isinstance(val, torch.Tensor):
                        with torch.no_grad():
                            if val.numel() == 1:
                                val = val.item()
                            else:
                                val = val.detach().float().mean().item()
                    wandb_log_dict[f'train/{k}'] = val
                wandb_log_dict['train/lrs'] = lr_scheduler.get_last_lr()[0]
                wandb_log_dict['train/elapsed_between_logs_sec'] = elapsed_between_logs_sec
                wandb_log_dict['train/epoch'] = round(current_epoch, 3)
                wandb_log_dict['train/sec_per_epoch'] = elapsed_between_logs_sec / 100 * len(train_dataset) / total_batch_size
                wandb_log_dict['train/global_step'] = global_step
                if wandb_logger and accelerator.is_main_process:
                    wandb_logger.log(wandb_log_dict)
                last_log_time = time.perf_counter()

                assert opt['val']['val_freq'] % opt['logger']['print_freq'] == 0, "val_freq must be divisible by print_freq"
                assert opt['logger']['save_checkpoint_freq'] % opt['logger']['print_freq'] == 0, "save_checkpoint_freq must be divisible by print_freq"

                if accelerator.is_main_process and (global_step % opt['logger']['save_checkpoint_freq'] == 0 or global_step == total_iter):
                    save_checkpoint(args, logger, accelerator, train_pipeline, global_step, opt['path']['models'])
                
                if (global_step % opt['val']['val_freq'] == 0 or global_step == total_iter or initial_inference_done == False) and global_step != last_eval_step:
                    _run_validation(global_step, 'loop', current_epoch)

                
                last_log_time = time.perf_counter()
            
        

def resume_checkpoint(args, accelerator, output_dir, train_pipeline):
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != 'latest':
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith('checkpoint')]
            dirs = sorted(dirs, key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split('-')[1])

            if train_pipeline.ema is not None:
                accelerator.print(f'Resuming ema from checkpoint {path}')
                ema_state = torch.load(os.path.join(output_dir, path, 'ema.pth'), map_location='cpu', weights_only=True)
                train_pipeline.ema.load_state_dict(ema_state)

    return global_step


def save_checkpoint(args, logger, accelerator, train_pipeline, global_step, output_dir):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints')
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f'checkpoint-{global_step}')
    accelerator.save_state(save_path)
    logger.info(f'Saved state to {save_path}')

    if train_pipeline.ema is not None:
        torch.save(train_pipeline.ema.state_dict(), os.path.join(save_path, 'ema.pth'))
        logger.info(f'Saved ema model to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default='latest')
    parser.add_argument('--checkpoints_total_limit', type=int, default=3, help=('Max number of checkpoints to store.'))
    args = parser.parse_args()

    train(args)
