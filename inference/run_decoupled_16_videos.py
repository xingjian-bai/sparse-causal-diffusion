"""Generate videos with the decoupled SCD model for qualitative evaluation.

Usage:
    python inference/run_decoupled_16_videos.py \
        --opt options/scd_minecraft.yml \
        --checkpoint path/to/ema.pth
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

DEFAULT_OPT_PATH = PROJECT_ROOT / 'options' / 'scd_minecraft.yml'
DEFAULT_EMA_PATH = PROJECT_ROOT / 'pretrained' / 'decoupled' / 'model.pth'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=str(DEFAULT_OPT_PATH))
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_EMA_PATH))
    parser.add_argument('--num-videos', type=int, default=16)
    parser.add_argument('--guidance-scale', type=float, default=1.5)
    parser.add_argument('--context-length', type=int, default=36)
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


def _align_video_pair_length(pred_video, gt_video):
    if gt_video is None:
        return None
    pred_frames = pred_video.shape[2]
    gt_frames = gt_video.shape[2]
    if gt_frames == pred_frames:
        return gt_video
    if gt_frames > pred_frames:
        return gt_video[:, :, :pred_frames]
    pad_frames = pred_frames - gt_frames
    last_frame = gt_video[:, :, -1:].repeat(1, 1, pad_frames, 1, 1, 1)
    return torch.cat([gt_video, last_frame], dim=2)


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
    base_opt['name'] = f'{base_name}_video_dump'

    base_opt.setdefault('logger', {})['use_wandb'] = False
    base_opt.setdefault('val', {}).setdefault('sample_cfg', {})
    base_opt['val']['sample_cfg']['sample_trajectory_per_video'] = 4

    _patch_dataset_paths(base_opt)

    guidance_scale = args.guidance_scale
    context_length = args.context_length

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
            f'Requested {args.num_videos} videos but dataset only has {len(sample_dataset)}.'
        )

    subset = torch.utils.data.Subset(sample_dataset, list(range(args.num_videos)))
    sample_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=base_opt['datasets']['sample']['batch_size_per_gpu'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_pipeline.model, sample_loader = accelerator.prepare(
        train_pipeline.model,
        sample_loader,
    )
    train_pipeline.set_ema_model(ema_decay=base_opt['train'].get('ema_decay'))

    combo_opt = copy.deepcopy(base_opt)
    combo_sample_cfg = combo_opt['val']['sample_cfg']
    combo_sample_cfg['context_length'] = context_length
    combo_sample_cfg['guidance_scale'] = guidance_scale
    combo_sample_cfg['unroll_length'] = _derive_unroll_length(combo_opt, context_length)

    logger.info(f'Generating {args.num_videos} videos | CFG={guidance_scale} | ctx={context_length}')

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

    num_trajectory = combo_sample_cfg['sample_trajectory_per_video']
    videos_generated = 0

    for batch in sample_loader:
        if videos_generated >= args.num_videos:
            break

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

        pred_video, _ = val_pipeline.generate(
            conditions=conditions,
            context_sequence=context_sequence,
            context_timestep_idx=train_pipeline.context_timestep_idx,
            unroll_length=combo_sample_cfg['unroll_length'],
            num_inference_steps=combo_sample_cfg['num_inference_steps'],
            sample_size=combo_sample_cfg['sample_size'],
            use_kv_cache=combo_sample_cfg.get('use_kv_cache', True),
            show_progress=accelerator.is_main_process,
            is_main_process=accelerator.is_main_process,
            debug=False,
            guidance_scale=guidance_scale,
        )

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
                f'iter_{context_length}',
                f'CFG_{guidance_scale}'
            ),
            wandb_logger=None,
            annotate_context_frame=combo_sample_cfg.get('anno_context', False),
            guidance_scale=guidance_scale,
            fps=8,
        )

        videos_generated += len(batch['index'])

    accelerator.wait_for_everyone()

    if train_pipeline.ema is not None:
        train_pipeline.ema.restore(model)
    train_pipeline.vae.disable_slicing()

    if accelerator.is_main_process:
        logger.info('Done.')


if __name__ == '__main__':
    main()
