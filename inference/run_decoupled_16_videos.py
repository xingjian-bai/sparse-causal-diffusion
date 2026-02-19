"""
Video Generation Script for Decoupled Encoder-Decoder Model

Generates a fixed number of long videos using the decoupled SCD model.
This is the main video generation script for the paper's method.

Usage:
    python inference/run_decoupled_16_videos.py \
        --opt options/scd_minecraft.yml \
        --checkpoint path/to/ema.pth \
        --num-videos 16
"""
import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scd.data import build_dataset
from scd.pipelines import build_pipeline
from scd.trainers import build_trainer
from scd.utils.logger_util import dict2str, set_path_logger
from scd.utils.vis_util import log_paired_video

# Default config path
DEFAULT_OPT_PATH = PROJECT_ROOT / 'options' / 'scd_minecraft.yml'

# Default checkpoint path - update this to your trained model
# Example: experiments/scd_minecraft/models/checkpoint-100000/ema.pth
DEFAULT_EMA_PATH = PROJECT_ROOT / 'pretrained' / 'decoupled' / 'model.pth'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a fixed number of videos with the decoupled SCD model.')
    parser.add_argument(
        '--opt',
        type=str,
        default=str(DEFAULT_OPT_PATH),
        help='Path to the decoupled evaluation YAML config.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(DEFAULT_EMA_PATH),
        help='Path to the EMA checkpoint (pth or safetensors).'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Optional override for the experiment name used under results/.'
    )
    parser.add_argument(
        '--guidance-scales',
        type=float,
        nargs='+',
        default=[1.0, 1.5],
        help='List of classifier-free guidance scales to evaluate.'
    )
    parser.add_argument(
        '--context-lengths',
        type=int,
        nargs='+',
        default=[36, 144],
        help='List of context lengths (frames) to evaluate.'
    )
    parser.add_argument(
        '--sample-trajectory',
        type=int,
        default=2,
        help='Number of trajectories sampled per video during evaluation.'
    )
    parser.add_argument(
        '--num-videos',
        type=int,
        default=16,
        help='Number of unique base videos to generate per (CFG, context) combination.'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of dataloader workers for the validation split.'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Optional root directory to prepend to dataset paths if they do not exist.'
    )
    parser.add_argument(
        '--use-wandb',
        dest='use_wandb',
        action='store_const',
        const=True,
        default=None,
        help='Force-enable wandb logging for this run.'
    )
    parser.add_argument(
        '--no-wandb',
        dest='use_wandb',
        action='store_const',
        const=False,
        help='Force-disable wandb logging for this run.'
    )
    parser.add_argument(
        '--generation-length',
        type=int,
        default=None,
        help='Number of frames to generate beyond the context window; defaults to config unroll_length.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=8,
        help='Frames per second when saving generated videos.'
    )
    parser.add_argument(
        '--max-cache-context',
        type=int,
        default=None,
        help='Override the model short_term_ctx_winsize for inference.'
    )
    return parser.parse_args()


def _load_config(path: str) -> Dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config at {path} did not resolve to a dict.')
    return cfg


def _ordered_unique(values: Iterable) -> List:
    seen = set()
    ordered = []
    for v in values:
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered


def _resolve_dataset_path(path_str: str, roots: List[Path]) -> str:
    if not path_str:
        return path_str

    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if candidate.exists():
        return str(candidate.resolve())

    for root in roots:
        if root is None:
            continue
        root_path = Path(root)
        resolved = root_path / candidate
        if resolved.exists():
            return str(resolved.resolve())

    return path_str


def _patch_dataset_paths(opt: Dict, data_root: Optional[str]) -> None:
    dataset_cfg = opt.get('datasets', {})
    env_root = os.environ.get('SCD_DATA_ROOT')
    roots: List[Path] = []
    if data_root:
        roots.append(Path(data_root))
    if env_root:
        roots.append(Path(env_root))
    roots.append(PROJECT_ROOT)

    for split_cfg in dataset_cfg.values():
        if not isinstance(split_cfg, dict):
            continue
        data_list = split_cfg.get('data_list')
        if isinstance(data_list, str):
            split_cfg['data_list'] = _resolve_dataset_path(data_list, roots)


def _get_nested(cfg: Dict, keys: Iterable[str]):
    node = cfg
    for key in keys:
        if node is None:
            return None
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _derive_unroll_length(opt: Dict, context_length: int) -> int:
    sample_cfg = opt.get('val', {}).get('sample_cfg', {})
    total_frames = _get_nested(opt, ('datasets', 'sample', 'data_cfg', 'num_frames'))
    if total_frames is None:
        return sample_cfg.get('unroll_length', 0)
    return max(0, int(total_frames) - int(context_length))


def _set_ctx_winsize(model_cfg: Dict, ctx_winsize: Optional[int]) -> None:
    if ctx_winsize is None:
        return
    value = int(ctx_winsize)

    transformer_cfg = model_cfg.get('transformer', {})
    init_cfg = transformer_cfg.get('init_cfg')
    if isinstance(init_cfg, dict):
        init_cfg['short_term_ctx_winsize'] = value
        if isinstance(init_cfg.get('config'), dict):
            init_cfg['config']['short_term_ctx_winsize'] = value

    decoder_cfg = model_cfg.get('decoder_cfg')
    if isinstance(decoder_cfg, dict):
        decoder_cfg['short_term_ctx_winsize'] = value
        if isinstance(decoder_cfg.get('config'), dict):
            decoder_cfg['config']['short_term_ctx_winsize'] = value


def _align_video_pair_length(pred_video: torch.Tensor, gt_video: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if gt_video is None:
        return None

    pred_frames = pred_video.shape[2]
    gt_frames = gt_video.shape[2]

    if gt_frames == pred_frames:
        return gt_video
    if gt_frames > pred_frames:
        return gt_video[:, :, :pred_frames]

    pad_frames = pred_frames - gt_frames
    if pad_frames <= 0:
        return gt_video

    last_frame = gt_video[:, :, -1:].repeat(1, 1, pad_frames, 1, 1, 1)
    return torch.cat([gt_video, last_frame], dim=2)


def _load_model_weights(model: torch.nn.Module, checkpoint_path: Optional[str], accelerator: Accelerator) -> None:
    if not checkpoint_path:
        return
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found at {ckpt_path}')

    accelerator.print(f'Loading model weights from: {ckpt_path}')
    if ckpt_path.suffix == '.safetensors':
        state_dict = load_file(str(ckpt_path))
    else:
        try:
            state_dict = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(str(ckpt_path), map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        accelerator.print(f'Missing keys when loading weights: {missing_keys}')
    if unexpected_keys:
        accelerator.print(f'Unexpected keys when loading weights: {unexpected_keys}')
    accelerator.print('Successfully loaded model weights!')


def main() -> None:
    args = parse_args()
    base_opt = _load_config(args.opt)

    if args.run_name:
        base_opt['name'] = args.run_name
    else:
        base_name = base_opt.get('name', Path(args.opt).stem)
        base_opt['name'] = f'{base_name}_decoupled_video_dump'

    model_cfg = base_opt.get('models', {}).get('model_cfg')
    if isinstance(model_cfg, dict):
        _set_ctx_winsize(model_cfg, args.max_cache_context)

    base_opt.setdefault('logger', {})
    if args.use_wandb is None:
        base_opt['logger'].setdefault('use_wandb', False)
    else:
        base_opt['logger']['use_wandb'] = args.use_wandb

    base_opt.setdefault('val', {}).setdefault('sample_cfg', {})
    base_opt['val']['sample_cfg']['sample_trajectory_per_video'] = args.sample_trajectory

    _patch_dataset_paths(base_opt, args.data_root)

    guidance_scales = _ordered_unique(args.guidance_scales)
    context_lengths = _ordered_unique(args.context_lengths)

    sample_dataset_cfg = base_opt.get('datasets', {}).get('sample')
    if isinstance(sample_dataset_cfg, dict):
        sample_list = sample_dataset_cfg.get('data_list')
        if isinstance(sample_list, str) and not Path(sample_list).exists():
            raise FileNotFoundError(
                f'Sample dataset list not found at {sample_list}. Pass --data-root or set SCD_DATA_ROOT.'
            )

    accelerator = Accelerator(mixed_precision=base_opt.get('mixed_precision', 'no'))
    with accelerator.main_process_first():
        set_path_logger(accelerator, args.opt, base_opt, is_train=False)

    logger = get_logger('scd', log_level='INFO')
    logger.info(accelerator.state)
    logger.info(dict2str(base_opt))

    if base_opt.get('manual_seed') is not None:
        set_seed(base_opt['manual_seed'] + accelerator.process_index)

    trainer_builder = build_trainer(base_opt['train'].get('train_pipeline', 'SCDTrainer'))
    train_pipeline = trainer_builder(**base_opt['models'], accelerator=accelerator)

    _load_model_weights(train_pipeline.model, args.checkpoint, accelerator)

    sample_dataset = build_dataset(base_opt['datasets']['sample'])
    if len(sample_dataset) < args.num_videos:
        raise ValueError(
            f'Requested {args.num_videos} videos but validation dataset only has {len(sample_dataset)} samples.'
        )

    subset_indices = list(range(args.num_videos))
    subset_dataset = torch.utils.data.Subset(sample_dataset, subset_indices)
    sample_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=base_opt['datasets']['sample']['batch_size_per_gpu'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_pipeline.model, sample_loader = accelerator.prepare(
        train_pipeline.model,
        sample_loader,
    )
    train_pipeline.set_ema_model(ema_decay=base_opt['train'].get('ema_decay'))

    base_opt_with_paths = copy.deepcopy(base_opt)

    for context_length in context_lengths:
        global_step = int(context_length)
        for guidance_scale in guidance_scales:
            logger.info(
                f'Generating videos | CFG={guidance_scale} | context={context_length}'
            )

            combo_opt = copy.deepcopy(base_opt_with_paths)
            combo_sample_cfg = combo_opt['val']['sample_cfg']
            combo_sample_cfg['context_length'] = int(context_length)
            combo_sample_cfg['guidance_scale'] = float(guidance_scale)
            combo_sample_cfg['unroll_length'] = int(
                args.generation_length
                if args.generation_length is not None
                else _derive_unroll_length(combo_opt, context_length)
            )

            model = accelerator.unwrap_model(train_pipeline.model)
            if train_pipeline.ema is not None:
                train_pipeline.ema.store(model)
                train_pipeline.ema.copy_to(model)

            train_pipeline.vae.eval()
            train_pipeline.vae.enable_slicing()
            model.eval()

            val_pipeline = build_pipeline(combo_opt['val'].get('val_pipeline', 'SCDPipeline'))(
                vae=train_pipeline.vae,
                transformer=model,
                scheduler=copy.deepcopy(train_pipeline.scheduler),
            )
            val_pipeline.execution_device = accelerator.device
            is_rank_0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            val_pipeline.set_progress_bar_config(disable=not is_rank_0)

            videos_generated = 0
            for batch_idx, batch in enumerate(sample_loader):
                if videos_generated >= args.num_videos:
                    break

                num_trajectory = combo_opt['val']['sample_cfg']['sample_trajectory_per_video']
                gt_video = batch['video'].unsqueeze(1).repeat((1, num_trajectory, 1, 1, 1, 1))
                gt_video = rearrange(gt_video, 'b n t c h w -> (b n) t c h w')

                context_sequence = gt_video[:, :context_length].clone()

                if 'label' in batch:
                    label_repeated = batch['label'].unsqueeze(1).repeat((1, num_trajectory)).reshape(-1)
                    conditions = {'label': label_repeated}
                elif 'action' in batch:
                    action_repeated = batch['action'].unsqueeze(1).repeat((1, num_trajectory, 1))
                    action_repeated = rearrange(action_repeated, 'b n t -> (b n) t')
                    conditions = {'action': action_repeated}
                else:
                    conditions = None

                input_params = {
                    'conditions': conditions,
                    'context_sequence': context_sequence,
                    'context_timestep_idx': train_pipeline.context_timestep_idx,
                    'unroll_length': combo_opt['val']['sample_cfg']['unroll_length'],
                    'num_inference_steps': combo_opt['val']['sample_cfg']['num_inference_steps'],
                    'sample_size': combo_opt['val']['sample_cfg']['sample_size'],
                    'use_kv_cache': combo_opt['val']['sample_cfg'].get('use_kv_cache', True),
                    'show_progress': accelerator.is_main_process,
                    'is_main_process': accelerator.is_main_process,
                    'debug': False,
                    'guidance_scale': guidance_scale,
                }

                pred_video, timing_dict = val_pipeline.generate(**input_params)

                if batch_idx == 0 and accelerator.is_main_process and base_opt['logger'].get('use_wandb', False):
                    timing_dict['timing/global_step'] = global_step
                    try:
                        import wandb
                        if getattr(wandb, 'run', None) is not None:
                            wandb.log(timing_dict)
                    except ImportError:
                        pass

                pred_video = rearrange(pred_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)
                gt_video = rearrange(gt_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)
                gt_video = _align_video_pair_length(pred_video, gt_video)

                log_paired_video(
                    sample=pred_video,
                    gt=gt_video,
                    context_frames=context_length,
                    save_suffix=batch['index'],
                    save_dir=os.path.join(
                        combo_opt['path']['visualization'],
                        f'iter_{global_step}',
                        f'CFG_{guidance_scale}'
                    ),
                    wandb_logger=None,
                    annotate_context_frame=combo_opt['val']['sample_cfg'].get('anno_context', False),
                    guidance_scale=guidance_scale,
                    fps=args.fps,
                )

                videos_generated += len(batch['index'])

                if videos_generated >= args.num_videos:
                    break

            accelerator.wait_for_everyone()

            if train_pipeline.ema is not None:
                train_pipeline.ema.restore(model)

            train_pipeline.vae.disable_slicing()

    if accelerator.is_main_process:
        logger.info('Video generation complete.')


if __name__ == '__main__':
    main()
