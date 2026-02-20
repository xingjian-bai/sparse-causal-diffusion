import os
from typing import List

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from PIL import Image


def log_paired_video(
    sample,
    guidance_scale,
    gt=None,
    context_frames=0,
    save_suffix=None,
    save_dir=None,
    wandb_logger=None,
    wandb_cfg=None,
    annotate_context_frame=True,
    fps=8,
):

    def _resize_video(tensor: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tensor is None:
            return None
        b, n, f, c, h, w = tensor.shape
        if (h, w) == target_hw:
            return tensor
        tensor_flat = tensor.reshape(-1, c, h, w)
        tensor_resized = F.interpolate(tensor_flat, size=target_hw, mode='bilinear', align_corners=False)
        return tensor_resized.reshape(b, n, f, c, target_hw[0], target_hw[1])

    if gt is not None:
        gt_h, gt_w = gt.shape[-2:]
        sample = _resize_video(sample, (gt_h, gt_w))
        gt = _resize_video(gt, (gt_h, gt_w))

    # Add red border of 1 pixel width to the context frames
    if annotate_context_frame:
        color = [255, 0, 0]
        for i, c in enumerate(color):
            c = c / 255.0
            sample[:, :, :context_frames, i, [0, -1], :] = c
            sample[:, :, :context_frames, i, :, [0, -1]] = c
            if gt is not None:
                gt[:, :, :context_frames, i, [0, -1], :] = c
                gt[:, :, :context_frames, i, :, [0, -1]] = c
    if gt is not None:
        video = torch.cat([sample, gt], dim=-1).float().detach().cpu().numpy()
    else:
        video = sample.float().detach().cpu().numpy()
    video = (video.clip(0, 1) * 255).astype(np.uint8)
    video = rearrange(video, 'b n f c h w -> b (n f) h w c')

    os.makedirs(save_dir, exist_ok=True)

    for vid, idx in zip(video, save_suffix):

        video_path = save_video_to_dir(vid, save_dir=save_dir, save_suffix=f'sample_gt_{idx}', save_type='video', fps=fps)

        if wandb_logger:
            # wandb_logger.log({f"{wandb_cfg['namespace']}/sample_{idx}": wandb.Video(video_path, fps=8, format='mp4')})
            wandb_logger.log({f"{wandb_cfg['namespace']}_{guidance_scale}/sample_{idx}": wandb.Video(video_path, fps=fps, format='mp4'), 'Step': wandb_cfg['step']})


def save_video_to_dir(video, save_dir, save_suffix, save_type='frame', fps=8):
    if isinstance(video, np.ndarray):
        video = [Image.fromarray(frame).convert('RGB') for frame in video]
    elif isinstance(video, list):
        video = video
    else:
        raise NotImplementedError

    os.makedirs(save_dir, exist_ok=True)

    save_type_list = save_type.split('_')

    # save frame
    if 'frame' in save_type_list:
        frame_save_dir = os.path.join(save_dir, 'frames')
        os.makedirs(frame_save_dir, exist_ok=True)
        for idx, img in enumerate(video):
            img.save(os.path.join(frame_save_dir, f'{idx:05d}_{save_suffix}.jpg'))

    # save to gif
    if 'gif' in save_type_list:
        gif_save_path = os.path.join(save_dir, f'{save_suffix}.gif')
        save_images_as_gif(video, gif_save_path, fps=fps)

    # save to video
    video_save_path = None
    if 'video' in save_type_list:
        video_save_path = os.path.join(save_dir, f'{save_suffix}.mp4')
        export_to_video(video, video_save_path, fps=fps)
    
    return video_save_path


def save_images_as_gif(images: List[Image.Image], save_path: str, fps=8) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=int(1000 / fps),
    )


def export_to_video(video_frames: List[Image.Image], output_video_path: str, fps=8) -> str:
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    video_writer = imageio.get_writer(output_video_path, fps=fps)
    for img in video_frames:
        video_writer.append_data(np.array(img))
    video_writer.close()
    return output_video_path
