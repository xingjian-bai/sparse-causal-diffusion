"""
Decoupled SCD Model Inference Script

This script evaluates the decoupled encoder-decoder SCD model under
multiple classifier-free guidance (CFG) scales and context length settings.

Usage:
    accelerate launch inference/run_decoupled_inference.py \
        --opt options/scd_minecraft.yml \
        --checkpoint path/to/ema.pth \
        --data-root /path/to/data
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, tqdm
from omegaconf import OmegaConf
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scd.data import build_dataset
from scd.trainers import build_trainer
from scd.utils.logger_util import dict2str, set_path_logger, setup_wandb


# Default config path
DEFAULT_OPT_PATH = PROJECT_ROOT / 'options' / 'scd_minecraft.yml'

# Default checkpoint path - update this to your trained model
# Example: experiments/scd_minecraft/models/checkpoint-100000/ema.pth
DEFAULT_EMA_PATH = PROJECT_ROOT / 'pretrained' / 'decoupled' / 'model.pth'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate decoupled SCD model under multiple CFG/context settings.')
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
        default=None,
        help='List of classifier-free guidance scales to evaluate. Defaults to config values if omitted.'
    )
    parser.add_argument(
        '--context-lengths',
        type=int,
        nargs='+',
        default=None,
        help='List of context lengths (frames) to evaluate. Defaults to config values if omitted.'
    )
    parser.add_argument(
        '--sample-trajectory',
        type=int,
        default=2,
        help='Number of trajectories sampled per video during evaluation.'
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
        '--save-json',
        type=str,
        default=None,
        help='Optional path to dump aggregated metrics as JSON.'
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
        help='Override the number of frames generated beyond the context window.'
    )
    parser.add_argument(
        '--max-cache-context',
        type=int,
        default=None,
        help='Override short_term_ctx_winsize (KV cache length) during inference.'
    )
    return parser.parse_args()


def _load_config(path: str) -> Dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config at {path} did not resolve to a dict.')
    return cfg


def _get_nested(cfg: Dict, keys: Iterable[str]):
    node = cfg
    for key in keys:
        if node is None:
            return None
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _ensure_sample_cfg(base_opt: Dict, sample_trajectory: int, force_wandb: Optional[bool]) -> None:
    sample_cfg = base_opt.setdefault('val', {}).setdefault('sample_cfg', {})
    sample_cfg['sample_trajectory_per_video'] = sample_trajectory

    logger_cfg = base_opt.setdefault('logger', {})
    if force_wandb is None:
        logger_cfg.setdefault('use_wandb', False)
    else:
        logger_cfg['use_wandb'] = force_wandb


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


def _derive_unroll_length(opt: Dict, context_length: int) -> int:
    sample_cfg = opt.get('val', {}).get('sample_cfg', {})
    total_frames = _get_nested(opt, ('datasets', 'sample', 'data_cfg', 'num_frames'))
    if total_frames is None:
        return sample_cfg.get('unroll_length', 0)
    return max(0, int(total_frames) - int(context_length))


def _set_ctx_winsize(model_cfg: Dict, ctx_winsize: Optional[int]) -> None:
    if ctx_winsize is None or not isinstance(model_cfg, dict):
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


class _ProgressDataLoader:
    def __init__(self, dataloader, progress_bar, total_items: Optional[int]):
        self._dataloader = dataloader
        self._progress_bar = progress_bar
        self._total_items = max(0, total_items or 0)
        self._remaining = self._total_items

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        pending_update = 0

        def _advance(count: int):
            if self._progress_bar is None or count <= 0 or self._remaining <= 0:
                return
            step = min(count, self._remaining)
            self._progress_bar.update(step)
            self._remaining -= step

        for batch in self._dataloader:
            _advance(pending_update)
            pending_update = self._batch_size(batch)
            yield batch

        _advance(pending_update)
        if self._progress_bar is not None and self._remaining > 0:
            self._progress_bar.update(self._remaining)
            self._remaining = 0

    @staticmethod
    def _batch_size(batch) -> int:
        if isinstance(batch, dict) and 'index' in batch:
            index = batch['index']
            if isinstance(index, torch.Tensor):
                return int(index.shape[0])
            try:
                return len(index)
            except TypeError:
                return 0
        return 0


def _init_wandb_logger(accelerator: Accelerator, opt: Dict):
    if not accelerator.is_main_process:
        return None
    if not opt.get('logger', {}).get('use_wandb', False):
        return None
    return setup_wandb(name=opt['name'], save_dir=opt['path']['log'])


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
        base_opt['name'] = f'{base_name}_decoupled_eval'

    model_cfg = base_opt.get('models', {}).get('model_cfg')
    _set_ctx_winsize(model_cfg, args.max_cache_context)

    _ensure_sample_cfg(base_opt, args.sample_trajectory, args.use_wandb)
    _patch_dataset_paths(base_opt, args.data_root)

    sample_cfg_defaults = base_opt.get('val', {}).get('sample_cfg', {})

    if args.guidance_scales is not None:
        guidance_scales = _ordered_unique(args.guidance_scales)
    else:
        default_guidance = sample_cfg_defaults.get('guidance_scale', [])
        if isinstance(default_guidance, (list, tuple)):
            guidance_scales = _ordered_unique(default_guidance)
        elif default_guidance is not None:
            guidance_scales = [float(default_guidance)]
        else:
            guidance_scales = [1.0]

    if args.context_lengths is not None:
        context_lengths = _ordered_unique(args.context_lengths)
    else:
        default_context = sample_cfg_defaults.get('context_length')
        if isinstance(default_context, (list, tuple)):
            context_lengths = _ordered_unique(default_context)
        elif default_context is not None:
            context_lengths = [int(default_context)]
        else:
            context_lengths = [36]

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

    sample_dataloader = None
    sample_dataset = None
    if base_opt.get('datasets', {}).get('sample'):
        sample_dataset = build_dataset(base_opt['datasets']['sample'])
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset,
            batch_size=base_opt['datasets']['sample']['batch_size_per_gpu'],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError('Sample dataset config is required for evaluation.')

    train_pipeline.model, sample_dataloader = accelerator.prepare(
        train_pipeline.model,
        sample_dataloader,
    )
    train_pipeline.set_ema_model(ema_decay=base_opt['train'].get('ema_decay'))

    total_videos = None
    if sample_dataset is not None:
        total_videos = len(sample_dataset)
    elif base_opt.get('datasets', {}).get('sample', {}).get('num_sample'):
        total_videos = int(base_opt['datasets']['sample']['num_sample'])
    world_size = max(1, accelerator.state.num_processes)
    if total_videos is not None:
        base_videos_per_proc = total_videos // world_size
        remainder = total_videos % world_size
        local_target = base_videos_per_proc + (1 if accelerator.process_index < remainder else 0)
    else:
        local_target = None

    combined_results: Dict[Tuple[float, int], Dict[str, float]] = {}
    base_opt_with_paths = copy.deepcopy(base_opt)

    wandb_logger = _init_wandb_logger(accelerator, base_opt)

    for context_length in context_lengths:
        global_step = int(context_length)
        for guidance_scale in guidance_scales:
            combo_opt = copy.deepcopy(base_opt_with_paths)
            combo_sample_cfg = combo_opt['val']['sample_cfg']
            combo_sample_cfg['context_length'] = int(context_length)
            combo_sample_cfg['guidance_scale'] = float(guidance_scale)
            combo_sample_cfg['unroll_length'] = int(
                args.generation_length
                if args.generation_length is not None
                else _derive_unroll_length(combo_opt, context_length)
            )

            logger.info(
                f'Running inference | CFG={guidance_scale} | context={context_length} | '
                f'unroll={combo_sample_cfg["unroll_length"]}'
            )
            if local_target and local_target > 0:
                progress_bar = tqdm(
                    total=local_target,
                    desc=f'GPU {accelerator.process_index} | CFG {guidance_scale} | ctx {context_length}',
                    position=accelerator.process_index,
                    leave=True,
                    disable=False,
                )
            else:
                progress_bar = None

            progress_loader = _ProgressDataLoader(sample_dataloader, progress_bar, local_target)

            train_pipeline.sample(
                progress_loader,
                combo_opt,
                guidance_scale=float(guidance_scale),
                wandb_logger=wandb_logger if accelerator.is_main_process else None,
                global_step=global_step,
            )
            accelerator.wait_for_everyone()

            if progress_bar is not None:
                progress_bar.close()

            if accelerator.is_main_process and combo_opt['val'].get('eval_cfg'):
                metrics = train_pipeline.eval_performance(
                    combo_opt,
                    guidance_scale=float(guidance_scale),
                    global_step=global_step,
                )
                combined_results[(float(guidance_scale), int(context_length))] = metrics

                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            f'metrics/CFG_{guidance_scale:.2f}_CTX_{context_length}/{metric}': value
                            for metric, value in metrics.items()
                        },
                        step=global_step,
                    )

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info('\n===== Aggregated Metrics =====')
        for context_length in context_lengths:
            for guidance_scale in guidance_scales:
                key = (float(guidance_scale), int(context_length))
                metrics = combined_results.get(key)
                if metrics is None:
                    logger.warning(
                        f'Metrics missing for CFG={guidance_scale}, context={context_length}.'
                    )
                    continue
                metric_str = ', '.join(
                    f'{metric}: {value:.4f}' for metric, value in metrics.items()
                )
                logger.info(
                    f'CFG={guidance_scale:.2f} | context={context_length:3d} -> {metric_str}'
                )

        if args.save_json:
            output_path = Path(args.save_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                f'CFG_{cfg:.2f}_CTX_{ctx}': metrics
                for (cfg, ctx), metrics in combined_results.items()
            }
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2)
            logger.info(f'Saved metrics to {output_path}')

    if wandb_logger is not None:
        wandb_logger.finish()


if __name__ == '__main__':
    main()
