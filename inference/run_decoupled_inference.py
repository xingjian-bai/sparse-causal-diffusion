"""Evaluate the decoupled SCD model across CFG scales and context lengths.

Usage:
    accelerate launch inference/run_decoupled_inference.py \
        --opt options/scd_minecraft.yml \
        --checkpoint path/to/ema.pth
"""
import argparse
import copy
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
from scd.utils.logger_util import dict2str, set_path_logger

DEFAULT_OPT_PATH = PROJECT_ROOT / 'options' / 'scd_minecraft.yml'
DEFAULT_EMA_PATH = PROJECT_ROOT / 'pretrained' / 'decoupled' / 'model.pth'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=str(DEFAULT_OPT_PATH))
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_EMA_PATH))
    parser.add_argument('--guidance-scales', type=float, nargs='+', default=None,
                        help='CFG scales to sweep (default: from config).')
    parser.add_argument('--context-lengths', type=int, nargs='+', default=None,
                        help='Context lengths to sweep (default: from config).')
    return parser.parse_args()


def _load_config(path: str) -> Dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f'Config at {path} did not resolve to a dict.')
    return cfg


def _get_nested(cfg: Dict, keys: Iterable[str]):
    node = cfg
    for key in keys:
        if node is None or not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


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
        resolved = Path(root) / candidate
        if resolved.exists():
            return str(resolved.resolve())
    return path_str


def _patch_dataset_paths(opt: Dict) -> None:
    dataset_cfg = opt.get('datasets', {})
    roots: List[Path] = []
    env_root = os.environ.get('SCD_DATA_ROOT')
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


def _load_model_weights(model, checkpoint_path, accelerator):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found at {ckpt_path}')

    accelerator.print(f'Loading weights from {ckpt_path}')
    if ckpt_path.suffix == '.safetensors':
        state_dict = load_file(str(ckpt_path))
    else:
        try:
            state_dict = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(str(ckpt_path), map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        accelerator.print(f'Missing keys: {missing}')
    if unexpected:
        accelerator.print(f'Unexpected keys: {unexpected}')


def main():
    args = parse_args()
    base_opt = _load_config(args.opt)

    base_name = base_opt.get('name', Path(args.opt).stem)
    base_opt['name'] = f'{base_name}_decoupled_eval'

    base_opt.setdefault('val', {}).setdefault('sample_cfg', {})
    base_opt['val']['sample_cfg']['sample_trajectory_per_video'] = 4
    base_opt.setdefault('logger', {})['use_wandb'] = False

    _patch_dataset_paths(base_opt)

    sample_cfg_defaults = base_opt['val']['sample_cfg']

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
                f'Sample dataset list not found at {sample_list}. Set SCD_DATA_ROOT env var.'
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
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=base_opt['datasets']['sample']['batch_size_per_gpu'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_pipeline.model, sample_dataloader = accelerator.prepare(
        train_pipeline.model,
        sample_dataloader,
    )
    train_pipeline.set_ema_model(ema_decay=base_opt['train'].get('ema_decay'))

    total_videos = len(sample_dataset)
    world_size = max(1, accelerator.state.num_processes)
    base_per_proc = total_videos // world_size
    remainder = total_videos % world_size
    local_target = base_per_proc + (1 if accelerator.process_index < remainder else 0)

    combined_results: Dict[Tuple[float, int], Dict[str, float]] = {}
    base_opt_snapshot = copy.deepcopy(base_opt)

    for context_length in context_lengths:
        global_step = int(context_length)
        for guidance_scale in guidance_scales:
            combo_opt = copy.deepcopy(base_opt_snapshot)
            combo_sample_cfg = combo_opt['val']['sample_cfg']
            combo_sample_cfg['context_length'] = int(context_length)
            combo_sample_cfg['guidance_scale'] = float(guidance_scale)
            combo_sample_cfg['unroll_length'] = _derive_unroll_length(combo_opt, context_length)

            logger.info(
                f'Running | CFG={guidance_scale} | context={context_length} | '
                f'unroll={combo_sample_cfg["unroll_length"]}'
            )

            progress_bar = tqdm(
                total=local_target,
                desc=f'GPU {accelerator.process_index} | CFG {guidance_scale} | ctx {context_length}',
                position=accelerator.process_index,
                leave=True,
            ) if local_target > 0 else None

            progress_loader = _ProgressDataLoader(sample_dataloader, progress_bar, local_target)

            train_pipeline.sample(
                progress_loader,
                combo_opt,
                guidance_scale=float(guidance_scale),
                wandb_logger=None,
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

            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info('\n===== Aggregated Metrics =====')
        for context_length in context_lengths:
            for guidance_scale in guidance_scales:
                key = (float(guidance_scale), int(context_length))
                metrics = combined_results.get(key)
                if metrics is None:
                    continue
                metric_str = ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
                logger.info(f'CFG={guidance_scale:.2f} | ctx={context_length:3d} -> {metric_str}')


if __name__ == '__main__':
    main()
