import copy
import json
import os
from glob import glob

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from einops import rearrange
from pytorchvideo.data.encoded_video import EncodedVideo
from safetensors.torch import load_file
from tqdm import tqdm

from scd.metrics.metric import VideoMetric
from scd.models import build_model
from scd.models.autoencoder_dc_model import MyAutoencoderDC
from scd.pipelines import build_pipeline
from scd.utils.ema_util import EMAModel
from scd.utils.registry import TRAINER_REGISTRY
from scd.utils.vis_util import log_paired_video


@TRAINER_REGISTRY.register()
class SCDTrainer:

    def __init__(
        self,
        accelerator,
        model_cfg,
        clean_context_ratio,
        weighting_scheme='uniform',
        context_timestep_idx=-1,
        training_type='base',
        train_no_action_ratio=0.0,
        encoder_input_type = 'default',
        encoder_lr_ratio = 1.0,
        decoder_lr_ratio = 1.0,
        input_space: str = 'latent',
    ):
        super(SCDTrainer, self).__init__()
        self.train_no_action_ratio = train_no_action_ratio
        assert self.train_no_action_ratio == 0, "train_no_action_ratio must be 0 for now"
        self.encoder_input_type = encoder_input_type

        self.accelerator = accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # build model
        if model_cfg['transformer']['from_pretrained']:
            raise NotImplementedError
        else:
            init_cfg = model_cfg['transformer']['init_cfg']
            base_model = build_model(init_cfg['type'])(**init_cfg.get('config', {}))
            if init_cfg.get('pretrained_path'):
                pretrained_path = init_cfg['pretrained_path']
                accelerator.print(f'Loading pretrained model weights from: {pretrained_path}')
                state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    accelerator.print(f'Missing keys when loading pretrained model: {missing_keys}')
                if unexpected_keys:
                    accelerator.print(f'Unexpected keys when loading pretrained model: {unexpected_keys}')
                accelerator.print(f'Successfully loaded pretrained model weights!')

            decoder_cfg = model_cfg['transformer'].get('decoder_cfg')
            if decoder_cfg is not None:
                self.decoder = build_model(decoder_cfg['type'])(**decoder_cfg.get('config', {}))
                from scd.models.scd_model import SCDEncoderDecoder
                self.model = SCDEncoderDecoder(base_model, self.decoder)
            else:
                self.decoder = None
                self.model = base_model

        if model_cfg['transformer'].get('gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()

        # input space configuration (latent vs pixel)
        # default stays 'latent' for backward compatibility
        self.input_space = input_space if input_space is not None else 'latent'
        self.pixel_space = (self.input_space == 'pixel')

        if not self.pixel_space:
            if model_cfg['vae'].get('from_pretrained'):
                if model_cfg['vae']['type'] == 'AutoencoderKL':
                    from diffusers import AutoencoderKL
                    self.vae = AutoencoderKL.from_pretrained(model_cfg['vae']['from_pretrained']).to(accelerator.device, dtype=weight_dtype)
                    self.vae.requires_grad_(False)
                    sf = getattr(self.vae.config, 'scaling_factor', None)
                    if sf is None:
                        latents_std = getattr(self.vae.config, 'latents_std', None)
                        if latents_std:
                            denom = sum(latents_std) / len(latents_std)
                            if denom <= 0:
                                sf = 1.0
                                accelerator.print(
                                    "[SCDTrainer] latents_std non-positive; falling back to scaling_factor=1.0"
                                )
                            else:
                                sf = 1.0 / float(denom)
                                accelerator.print(
                                    f"[SCDTrainer] Derived scaling_factor={sf:.6f} from mean latents_std={denom:.6f}"
                                )
                        else:
                            sf = 1.0
                            accelerator.print(
                                "[SCDTrainer] latents_std unavailable; falling back to scaling_factor=1.0"
                            )
                        try:
                            setattr(self.vae.config, 'scaling_factor', sf)
                        except Exception:
                            pass
                    sf = float(sf)
                    lc = getattr(self.vae.config, 'latent_channels', 'NA')
                    accelerator.print(f"Loaded AutoencoderKL | scaling_factor={sf} latent_channels={lc}")
                    latents_mean = getattr(self.vae.config, 'latents_mean', None)
                    latents_std = getattr(self.vae.config, 'latents_std', None)
                    if latents_mean and latents_std:
                        mean_tensor = torch.tensor(latents_mean, dtype=weight_dtype, device=accelerator.device).view(1, -1, 1, 1)
                        std_tensor = torch.tensor(latents_std, dtype=weight_dtype, device=accelerator.device).view(1, -1, 1, 1)
                        std_tensor = torch.clamp(std_tensor, min=1e-6)
                        inv_std_tensor = torch.reciprocal(std_tensor)
                        setattr(self.vae, '_diffusion_latent_offset', mean_tensor)
                        setattr(self.vae, '_diffusion_latent_scale', inv_std_tensor)
                        accelerator.print(f"[SCDTrainer] Using VA-VAE normalization with per-channel mean/std | mean_norm={mean_tensor.abs().mean().item():.4f} | inv_std_mean={inv_std_tensor.mean().item():.4f}")
                    else:
                        setattr(self.vae, '_diffusion_latent_offset', None)
                        setattr(self.vae, '_diffusion_latent_scale', None)
                else:
                    self.vae = build_model(model_cfg['vae']['type']).from_pretrained(
                        model_cfg['vae']['from_pretrained']).to(accelerator.device, dtype=weight_dtype)
                    self.vae.requires_grad_(False)
            elif model_cfg['vae'].get('from_config'):
                with open(model_cfg['vae']['from_config'], 'r') as fr:
                    config = json.load(fr)
                self.vae = build_model(model_cfg['vae']['type']).from_config(config)
                if model_cfg['vae'].get('from_config_pretrained'):
                    state_dict_path = model_cfg['vae']['from_config_pretrained']
                    if state_dict_path.endswith('.safetensors'):
                        state_dict = load_file(model_cfg['vae']['from_config_pretrained'])
                    else:
                        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
                    self.vae.load_state_dict(state_dict)
                self.vae.to(accelerator.device, dtype=weight_dtype)
                self.vae.requires_grad_(False)
            else:
                raise NotImplementedError
        else:
            # pixel space: no VAE used
            self.vae = None

        if model_cfg['scheduler']['from_pretrained']:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_cfg['scheduler']['from_pretrained'], subfolder='scheduler')
        else:
            raise NotImplementedError

        self.weighting_scheme = weighting_scheme
        self.ema = None
        self.clean_context_ratio = clean_context_ratio
        self.context_timestep_idx = context_timestep_idx
        self.training_type = training_type
        self.encoder_lr_ratio = float(encoder_lr_ratio)
        self.decoder_lr_ratio = float(decoder_lr_ratio)

        self.last_iter = -1
        self.last_encoder_output = None
        self._cached_batch = None

    def set_ema_model(self, ema_decay):
        logger = get_logger('scd', log_level='INFO')

        if ema_decay is not None:
            self.ema = EMAModel(self.accelerator.unwrap_model(self.model), decay=ema_decay)
            logger.info(f'enable EMA training with decay {ema_decay}')

    def get_params_to_optimize(self, param_names_to_optimize, base_lr=None):
        logger = get_logger('scd', log_level='INFO')

        params_to_optimize = []
        params_to_fix = []
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_optimize.append(param)
                logger.info(f'optimize params: {name}')
                if self.decoder is not None and name.startswith('decoder.'):
                    decoder_params.append(param)
                else:
                    encoder_params.append(param)
            else:
                params_to_fix.append(param)
                logger.info(f'fix params: {name}')

        logger.info(
            f'#Trained Parameters: {sum([p.numel() for p in params_to_optimize]) / 1e6} M'
        )
        logger.info(
            f'#Fixed Parameters: {sum([p.numel() for p in params_to_fix]) / 1e6} M'
        )

        if base_lr is None:
            return params_to_optimize

        param_groups = []
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': base_lr * self.encoder_lr_ratio,
            })
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': base_lr * self.decoder_lr_ratio,
            })

        return param_groups

    def _clone_batch(self, batch):
        cloned = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _normalize_latents_for_diffusion(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        offset = getattr(self.vae, '_diffusion_latent_offset', None)
        scale = getattr(self.vae, '_diffusion_latent_scale', None)
        if offset is not None and scale is not None:
            offset = offset.to(device=latents.device, dtype=latents.dtype)
            scale = scale.to(device=latents.device, dtype=latents.dtype)
            latents = (latents - offset) * scale
        else:
            sf = float(getattr(self.vae.config, 'scaling_factor', 1.0))
            latents = latents * sf
        return latents

    def sample_frames(self, batch):
        video = batch['video'] if 'video' in batch else batch['latent']
        total_frames = video.shape[1]

        min_frames = self.accelerator.unwrap_model(self.model).config.short_term_ctx_winsize + 1
        if total_frames <= min_frames:
            num_sample_frames = torch.tensor([total_frames], device=self.accelerator.device)
        else:
            num_sample_frames = torch.randint(
                low=min_frames,
                high=total_frames + 1,
                size=(1, ),
                device=self.accelerator.device)
        num_sample_frames = broadcast(num_sample_frames)

        num_sample_frames_int = int(num_sample_frames.item())
        max_start = max(total_frames - num_sample_frames_int, 0)
        if max_start > 0:
            start_frame_idx = torch.randint(0, max_start + 1, (1,), device=self.accelerator.device)
            start_frame_idx = int(broadcast(start_frame_idx).item())
        else:
            start_frame_idx = 0

        video = video[:, start_frame_idx:start_frame_idx + num_sample_frames_int]
        num_sample_frames = num_sample_frames_int

        if 'label' in batch:
            batch['label'] = batch['label'].long()
        elif 'action' in batch:
            batch['action'] = batch['action'][:, start_frame_idx:start_frame_idx + num_sample_frames]
        else:
            raise NotImplementedError

        if 'video' in batch:
            batch['video'] = video
        else:
            batch['latent'] = video

        return batch

    def train_step(self, batch, iters=-1):
        if self.vae is not None:
            self.vae.eval()
        self.model.train()

        if iters != self.last_iter:
            self.last_encoder_output = None
            self._cached_batch = None
        if self.training_type == 'long_context':
            if iters == self.last_iter and self._cached_batch is not None:
                batch = self._clone_batch(self._cached_batch)
            else:
                batch = self.sample_frames(batch)
                self._cached_batch = self._clone_batch(batch)

        if 'video' in batch:
            video = batch['video'].to(dtype=self.weight_dtype)  # [0,1]
            batch_size, num_frames = video.shape[:2]
            if self.vae is not None:
                if isinstance(self.vae, MyAutoencoderDC):
                    video_btchw = rearrange(video, 'b t c h w -> (b t) c h w')
                    latents = self.vae.encode((video_btchw * 2 - 1)).latent
                else:
                    # diffusers AutoencoderKL branch
                    video_btchw = rearrange(video, 'b t c h w -> (b t) c h w')
                    latents_dist = self.vae.encode((video_btchw * 2 - 1).to(dtype=self.vae.dtype)).latent_dist
                    latents = latents_dist.sample()
            else:
                # pixel space: treat pixels as latents after normalization to [-1,1]
                latents = rearrange((video * 2 - 1), 'b t c h w -> (b t) c h w')
        else:
            latents = batch['latent'].to(dtype=self.weight_dtype)
            batch_size, num_frames = latents.shape[0], latents.shape[1]
            latents = rearrange(latents, 'b t c h w -> (b t) c h w')

        if 'label' in batch:
            labels = batch['label']
            if labels.ndim > 1:
                labels = labels.squeeze(-1)
            conditions = {'label': labels.to(self.accelerator.device, dtype=torch.long)}
        elif 'action' in batch:
            actions = batch['action']
            if self.train_no_action_ratio > 0:
                mask = torch.rand(actions.shape[0], device=actions.device) < self.train_no_action_ratio
                if mask.any():
                    null_action_idx = self.accelerator.unwrap_model(self.model).config.condition_cfg['num_action_classes']
                    actions = actions.clone()
                    actions[mask] = null_action_idx
            conditions = {'action': actions}
        else:
            conditions = None

        if self.vae is not None:
            # normalize encoded latents before diffusion
            latents = self._normalize_latents_for_diffusion(latents)

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample a random timestep for each image
        # flow matching requires float timesteps (retrieve from scheduler)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size * num_frames,
            logit_mean=0,
            logit_std=1,
        )

        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=latents.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.scheduler.scale_noise(latents, timesteps, noise)

        timesteps = rearrange(timesteps, '(b t) -> b t', b=batch_size)
        latents = rearrange(latents, '(b t) c h w -> b t c h w', b=batch_size)
        noisy_latents = rearrange(noisy_latents, '(b t) c h w -> b t c h w', b=batch_size)
        noise = rearrange(noise, '(b t) c h w -> b t c h w', b=batch_size)

        short_term_ctx_winsize = self.accelerator.unwrap_model(
            self.model).config.short_term_ctx_winsize
        original_noisy_latents = noisy_latents.clone()
        if self.clean_context_ratio is None:
            context_mask = torch.zeros((batch_size, num_frames), device=latents.device).bool()
        else:
            if self.training_type == 'long_context':
                context_mask = torch.ones((batch_size, num_frames - short_term_ctx_winsize), device=latents.device).bool()
                noise_mask = torch.rand((batch_size, short_term_ctx_winsize), device=latents.device) < self.clean_context_ratio
                context_mask = torch.cat([context_mask, noise_mask], dim=-1)
                assert self.context_timestep_idx == -1
                timesteps[context_mask] = self.context_timestep_idx
                noisy_latents[context_mask] = latents[context_mask]  # clean context
            else:
                context_mask = torch.rand((batch_size, num_frames), device=latents.device) < self.clean_context_ratio
                assert self.context_timestep_idx == -1
                timesteps[context_mask] = self.context_timestep_idx
                noisy_latents[context_mask] = latents[context_mask]  # clean context
        
        if self.encoder_input_type == 'clean':
            assert self.training_type == 'long_context'
            noisy_latents_encoder = latents.clone()
        elif self.encoder_input_type == 'mask_0.4':
            assert self.training_type == 'long_context'
            noisy_latents_encoder = original_noisy_latents.clone()
            timesteps_encoder = timesteps.clone()
            context_mask_encoder = torch.ones((batch_size, num_frames - short_term_ctx_winsize), device=latents.device).bool()
            noise_mask_encoder = torch.rand((batch_size, short_term_ctx_winsize), device=latents.device) < 0.4
            context_mask_encoder = torch.cat([context_mask_encoder, noise_mask_encoder], dim=-1)
            timesteps_encoder[context_mask_encoder] = self.context_timestep_idx
            noisy_latents_encoder[context_mask_encoder] = latents[context_mask_encoder]  # clean context
        else:  # default encoder input
            noisy_latents_encoder = None

        model_ret = self.model(noisy_latents, 
                               timestep=timesteps, 
                               conditions=conditions, 
                               context_timestep_idx=self.context_timestep_idx, 
                               hidden_states_encoder=noisy_latents_encoder, 
                               last_encoder_output=self.last_encoder_output if iters == self.last_iter else None)
        self.last_encoder_output = model_ret.encoder_output.clone()
        self.last_iter = iters
        model_pred = model_ret.sample
        target = noise - latents

        pred_ctx = model_pred.shape[1]
        model_pred = model_pred[:, -pred_ctx:]
        target = target[:, -pred_ctx:]
        context_mask = context_mask[:, -pred_ctx:]


        loss = torch.mean(((model_pred.float() - target.float())**2).reshape(target.shape[0], target.shape[1], -1), -1)

        loss_mask = ~context_mask
        loss = (loss * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1) + 1e-9)
        loss = loss.mean()

        log_dict = {'total_loss': loss}
        # merge optional metrics for logging (keeps training behavior unchanged)
        try:
            if hasattr(model_ret, 'metrics') and isinstance(model_ret.metrics, dict):
                for k, v in model_ret.metrics.items():
                    if isinstance(v, torch.Tensor):
                        log_dict[k] = v
        except Exception:
            pass

        return log_dict

    @torch.no_grad()
    def sample(self, val_dataloader, opt, guidance_scale, wandb_logger=None, global_step=None, debug=False, num_inference_steps=None):
        model = self.accelerator.unwrap_model(self.model)

        if self.ema is not None:
            self.ema.store(model)
            self.ema.copy_to(model)

        if self.vae is not None:
            self.vae.eval()
            self.vae.enable_slicing()
        model.eval()

        val_pipeline = build_pipeline(opt['val'].get('val_pipeline', 'SCDPipeline'))(
            vae=self.vae,
            transformer=model,
            scheduler=copy.deepcopy(self.scheduler),
            pixel_space=self.pixel_space,
        )
        val_pipeline.execution_device = self.accelerator.device
        # Only show progress bar on rank 0 to avoid multiple progress bars
        is_rank_0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        val_pipeline.set_progress_bar_config(disable=not is_rank_0)

        # decide inference steps (override if provided, unless debug)
        _num_infer_steps = num_inference_steps if not debug else 2
        if _num_infer_steps is None:
            _num_infer_steps = opt['val']['sample_cfg']['num_inference_steps']

        # Video-level progress bar (rank 0 only)
        pbar = None
        if self.accelerator.is_main_process:
            try:
                total_videos = len(val_dataloader)
                pbar = tqdm(total=total_videos, desc=f"eval videos scale={guidance_scale} steps={_num_infer_steps}", leave=False)
            except Exception:
                pbar = None

        # Only show tqdm progress per video on rank 0
        for batch_idx, batch in enumerate(val_dataloader):
            max_wandb_videos = opt['val']['sample_cfg'].get('wandb_max_videos', 8)
            num_trajectory = opt['val']['sample_cfg']['sample_trajectory_per_video']
            gt_video = batch['video'].unsqueeze(1).repeat((1, num_trajectory, 1, 1, 1, 1))

            if 'select_last_frame' in opt['val']['sample_cfg']:
                gt_video = gt_video[:, :, -opt['val']['sample_cfg']['select_last_frame']:]
                batch['action'] = batch['action'][:, -opt['val']['sample_cfg']['select_last_frame']:]

            gt_video = rearrange(gt_video, 'b n t c h w -> (b n) t c h w')

            cfg_context_len = opt['val']['sample_cfg']['context_length'] if not debug else gt_video.shape[1] - 2
            context_length = min(cfg_context_len, gt_video.shape[1])
            context_sequence = gt_video[:, :context_length].clone()


            if 'label' in batch:
                # Repeat label to match the video trajectory repetition
                label_repeated = batch['label'].unsqueeze(1).repeat((1, num_trajectory)).reshape(-1)
                conditions = {'label': label_repeated}
            elif 'action' in batch:
                # Repeat action to match the video trajectory repetition
                action_repeated = batch['action'].unsqueeze(1).repeat((1, num_trajectory, 1))
                action_repeated = rearrange(action_repeated, 'b n t -> (b n) t')
                conditions = {'action': action_repeated}
            else:
                conditions = None

            input_params = {
                'conditions': conditions,
                'context_sequence': context_sequence,
                'context_timestep_idx': self.context_timestep_idx,
                'unroll_length': opt['val']['sample_cfg']['unroll_length'] if not debug else 2,
                'num_inference_steps': _num_infer_steps,
                'sample_size': opt['val']['sample_cfg']['sample_size'],
                'use_kv_cache': opt['val']['sample_cfg'].get('use_kv_cache', True),
                'show_progress': False,  # disable per-frame pbar; we show per-video pbar here
                'is_main_process': self.accelerator.is_main_process,
                'debug': debug,
                'guidance_scale': guidance_scale,
                'cfg_type': 'action' if 'cfg_type' not in opt['val']['sample_cfg'] else opt['val']['sample_cfg']['cfg_type']
            }

            pred_video, timing_dict = val_pipeline.generate(**input_params)

            if batch_idx == 0 and self.accelerator.is_main_process and wandb_logger:
                # suffix timing keys with steps if overridden to avoid collisions
                if num_inference_steps is not None:
                    timing_dict = {f"{k}_steps{_num_infer_steps}": v for k, v in timing_dict.items()}
                timing_dict['timing/global_step'] = global_step
                wandb_logger.log(timing_dict)

            pred_video = rearrange(pred_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)
            gt_video = rearrange(gt_video, '(b n) f c h w -> b n f c h w', n=num_trajectory)

            # organize save directory; include steps subdir only when steps are explicitly overridden
            _save_dir = os.path.join(opt['path']['visualization'], f'iter_{global_step}', f'CFG_{guidance_scale}')
            if num_inference_steps is not None:
                _save_dir = os.path.join(_save_dir, f'STEPS_{_num_infer_steps}')

            log_paired_video(
                sample=pred_video,
                gt=gt_video,
                context_frames=context_length,
                save_suffix=batch['index'],
                save_dir=_save_dir,
                wandb_logger=wandb_logger if (batch_idx < max_wandb_videos) else None,  # log first N samples to wandb
                wandb_cfg={
                    'namespace': 'eval_vis',
                    'step': global_step,
                },
                annotate_context_frame=opt['val']['sample_cfg'].get('anno_context', False),
                guidance_scale=guidance_scale)

            if debug:
                break

            # update video-level pbar
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass

        self.accelerator.wait_for_everyone()

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

        if self.ema is not None:
            self.ema.restore(model)

        if self.vae is not None:
            self.vae.disable_slicing()

    def read_video_folder(self, video_dir, num_trajectory):
        logger = get_logger('scd', log_level='INFO')
        video_path_list = sorted(glob(os.path.join(video_dir, '*.mp4')))
        video_list = []
        for video_path in video_path_list:
            try:
                video = EncodedVideo.from_path(video_path, decode_audio=False)
                video = video.get_clip(start_sec=0.0, end_sec=video.duration)['video']
                video_list.append(video)
            except Exception:
                logger.warning(f'Error opening {video_path}')

        videos = torch.stack(video_list)
        videos = rearrange(videos, 'b c (n f) h w -> b n f c h w', n=num_trajectory)

        videos = videos / 255.0
        videos_sample, videos_gt = torch.chunk(videos, 2, dim=-1)

        return videos_sample, videos_gt

    def eval_performance(self, opt, guidance_scale, global_step=0, num_inference_steps=None):
        logger = get_logger('scd', log_level='INFO')
        sample_dir = os.path.join(opt['path']['visualization'], f'iter_{global_step}', f'CFG_{guidance_scale}')
        if num_inference_steps is not None:
            sample_dir = os.path.join(sample_dir, f'STEPS_{num_inference_steps}')
        logger.info(f'begin evaluate {sample_dir}')

        video_metric = VideoMetric(metric=opt['val']['eval_cfg']['metrics'], device=self.accelerator.device)

        videos_sample, videos_gt = self.read_video_folder(sample_dir, num_trajectory=opt['val']['sample_cfg']['sample_trajectory_per_video'])

        result_dict = video_metric.compute(videos_sample.contiguous(), videos_gt.contiguous(), context_length=opt['val']['sample_cfg']['context_length'])
        logger.info(f'result_dict: {result_dict}')

        return result_dict
