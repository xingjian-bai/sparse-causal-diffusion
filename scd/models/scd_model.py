from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import FluxPosEmbed, LabelEmbedding, TimestepEmbedding, Timesteps, apply_rotary_emb
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import LayerNorm, RMSNorm
from diffusers.utils import is_torch_version
from einops import rearrange

from scd.utils.registry import MODEL_REGISTRY


class AdaLayerNormContinuous(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type='layer_norm',
        unconditional=False,
        df_noise_strength=0.0,
    ):
        super().__init__()
        self.df_noise_strength = df_noise_strength
        self.unconditional = unconditional
        if not self.unconditional:
            self.silu = nn.SiLU()
            self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == 'layer_norm':
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == 'rms_norm':
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f'unknown norm_type {norm_type}')
        assert elementwise_affine is False, "elementwise_affine must be False for AdaLayerNormContinuous"

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor, inference_noise_strength = 0.0, guidance_scale = 1.0) -> torch.Tensor:
        x = self.norm(x)
        if self.training and self.df_noise_strength > 0.0:
            noise = torch.randn_like(x)
            x = x + noise * self.df_noise_strength
        elif not self.training and inference_noise_strength > 0.0:
            if guidance_scale > 1.00001:
                # Inference noise is only added to the unconditional (first) half of the batch
                half_batch_size = x.shape[0]//2
                noise = torch.randn_like(x[:half_batch_size, ...])
                x[:half_batch_size, ...] = x[:half_batch_size, ...] + noise * inference_noise_strength
            elif guidance_scale == 1.0:
                pass  # No CFG, no noise
            elif guidance_scale == -1.0:
                # Unconditional: noise added to the entire batch
                noise = torch.randn_like(x)
                x = x + noise * inference_noise_strength
            else:
                raise NotImplementedError(f"guidance_scale {guidance_scale} is not supported")
            
        if not self.unconditional:
            emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
            scale, shift = torch.chunk(emb, 2, dim=-1)
            x = x * (1 + scale) + shift
            return x
        else:
            return x


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type='layer_norm', bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == 'layer_norm':
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa


class SCDAttnProcessor:

    def __init__(self):
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError('SCDAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.')

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states=None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        layer_kv_cache=None
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        weight_dtype = query.dtype

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if not attn.training:  # inference time
            if layer_kv_cache['kv_cache'] is not None:
                if len(layer_kv_cache['kv_cache']) != 0:  # contain cache
                    key = torch.cat([layer_kv_cache['kv_cache']['key'], key], dim=2)
                    value = torch.cat([layer_kv_cache['kv_cache']['value'], value], dim=2)

                if layer_kv_cache['is_cache_step']:  # need to record the cache
                    layer_kv_cache['kv_cache']['key'] = key[:, :, :, :].to(weight_dtype)
                    layer_kv_cache['kv_cache']['value'] = value[:, :, :, :].to(weight_dtype)

            attention_mask = attention_mask[:, -query.shape[2]:, :] if attention_mask is not None else None
            query_rotary_emb = (image_rotary_emb[0][-query.shape[2]:, :], image_rotary_emb[1][-query.shape[2]:, :])
        else:  # training
            query_rotary_emb = image_rotary_emb

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, layer_kv_cache


class SCDTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLayerNormZeroSingle(dim)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=SCDAttnProcessor(),
            qk_norm='rms_norm',
            eps=1e-6,
        )
        self.norm2 = AdaLayerNormZeroSingle(dim)
        self.mlp = FeedForward(dim=dim, dim_out=dim, activation_fn='gelu-approximate')

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        attention_mask=None,
        layer_kv_cache=None
    ):
        norm_hidden_states, gate = self.norm1(hidden_states, emb=temb)

        attn_output, layer_kv_cache = self.attn(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            layer_kv_cache=layer_kv_cache)
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = self.norm2(hidden_states, emb=temb)
        hidden_states = hidden_states + gate * self.mlp(norm_hidden_states)
        return hidden_states, layer_kv_cache


class SCDTransformer(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = ['SCDTransformerBlock']

    @register_to_config
    def __init__(
        self,
        num_layers: int,
        patch_size: int = 1,
        in_channels: int = 32,
        attention_head_dim: int = 64,
        num_attention_heads: int = 12,
        axes_dims_rope: Tuple[int] = (16, 24, 24),
        out_channels=32,
        slope_scale=0,
        short_term_ctx_winsize=16,
        condition_cfg=None,
        decouple_type=None,
        decoder_input_combine="concat",
        norm_out_unconditional=False,
        noise_strength=0.0,
    ):
        super().__init__()
        self.decouple_type = decouple_type
        self.decoder_input_combine = decoder_input_combine
        assert decoder_input_combine in ["concat", "add", "token_concat", "token_concat_with_proj"], \
            f"decoder_input_combine must be one of ['concat', 'add', 'token_concat', 'token_concat_with_proj'], but got {decoder_input_combine}"
        assert decouple_type in ["encoder", "decoder", "causal"], f"decouple_type must be either 'encoder' or 'decoder' or 'causal', but got {decouple_type}"

        if self.decouple_type == "encoder":
            pass
        elif self.decouple_type == "decoder":
            assert noise_strength == 0.0, "noise_strength must be 0.0 for decoder"
        elif self.decouple_type == "causal":
            pass
        else:
            raise NotImplementedError
        
        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.x_embedder = torch.nn.Linear(self.config.in_channels * self.config.patch_size * self.config.patch_size, self.inner_dim)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        if condition_cfg is not None:
            if condition_cfg['type'] == 'label':
                self.label_embedder = LabelEmbedding(condition_cfg['num_classes'], self.inner_dim, dropout_prob=0.1)
            elif condition_cfg['type'] == 'action':
                self.action_embedder = LabelEmbedding(condition_cfg['num_action_classes'], self.inner_dim, dropout_prob=0.1)
            else:
                raise NotImplementedError

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.transformer_blocks = nn.ModuleList([
            SCDTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
            ) for i in range(self.config.num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            unconditional=self.config.norm_out_unconditional,
            df_noise_strength=noise_strength
        )

        if self.decouple_type == "encoder" or self.decouple_type == "causal":
            self.proj_out = None
        else:
            self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
            if self.decoder_input_combine == "concat":
                self.decoder_input_proj = nn.Linear(self.inner_dim * 2, self.inner_dim, bias=True)
                self.decoder_alignment = None
            elif self.decoder_input_combine == "token_concat_with_proj":
                self.decoder_input_proj = None
                self.decoder_alignment = nn.Sequential(
                    nn.LayerNorm(self.inner_dim),
                    nn.Linear(self.inner_dim, self.inner_dim)
                )
            else:
                self.decoder_input_proj = None
                self.decoder_alignment = None

        self.gradient_checkpointing = False
        self.initialize_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value

    def _pack_latent_sequence(self, latents, patch_size):
        batch_size, num_frames, channel, height, width = latents.shape
        height, width = height // patch_size, width // patch_size

        latents = rearrange(
            latents, 'b f c (h p1) (w p2) -> b (f h w) (c p1 p2)', b=batch_size, f=num_frames, c=channel, h=height, p1=patch_size, w=width, p2=patch_size)

        return latents

    def _prepare_latent_sequence_ids(self, batch_size, num_frames, height, width, patch_size, device, dtype):
        patch_size = self.config.patch_size
        height, width = height // patch_size, width // patch_size
        latent_image_ids = torch.zeros(num_frames, height, width, 3)

        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(num_frames)[:, None, None]
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[None, :, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, None, :]

        latent_image_id_num_frames, latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_num_frames * latent_image_id_height * latent_image_id_width, latent_image_id_channels)
        return latent_image_ids.to(device=device, dtype=dtype)

    def _unpack_latent_sequence(self, latents, num_frames, height, width):
        batch_size, num_patches, channels = latents.shape
        patch_size = self.config.patch_size
        height, width = height // patch_size, width // patch_size

        latents = latents.view(batch_size * num_frames, height, width, channels // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, num_frames, channels // (patch_size * patch_size), height * patch_size, width * patch_size)
        return latents

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        if hasattr(self, 'label_embedder'):
            nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        if hasattr(self, 'action_embedder'):
            nn.init.normal_(self.action_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm2.linear.weight, 0)
            nn.init.constant_(block.norm2.linear.bias, 0)

        # Zero-out output layers:
        if not self.config.norm_out_unconditional:
            nn.init.constant_(self.norm_out.linear.weight, 0)
            nn.init.constant_(self.norm_out.linear.bias, 0)
        if self.proj_out is not None:
            nn.init.constant_(self.proj_out.weight, 0)
            nn.init.constant_(self.proj_out.bias, 0)

        if hasattr(self, 'decoder_input_proj') and self.decoder_input_proj is not None:
            # Initialize as [I | I] to act as addition initially
            # Weight shape: (inner_dim, inner_dim * 2)
            with torch.no_grad():
                # Start with zeros
                self.decoder_input_proj.weight.zero_()
                # Set left half as identity
                self.decoder_input_proj.weight[:, :self.inner_dim] = torch.eye(self.inner_dim)
                # Set right half as identity
                self.decoder_input_proj.weight[:, self.inner_dim:] = torch.eye(self.inner_dim)
                # Bias stays at zero
                if self.decoder_input_proj.bias is not None:
                    self.decoder_input_proj.bias.zero_()

        if hasattr(self, 'decoder_alignment') and self.decoder_alignment is not None:
            for module in self.decoder_alignment:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _build_causal_mask(self, input_shape, device, dtype):

        batch_size, num_frames, seq_len = input_shape
        token_per_frame = seq_len // num_frames

        def get_relative_positions(seq_len) -> torch.tensor:
            frame_idx = torch.arange(seq_len, device=device) // token_per_frame
            return (frame_idx.unsqueeze(0) - frame_idx.unsqueeze(1)).unsqueeze(0)

        # step 1: build context-context causal mask
        idx = torch.arange(seq_len, device=device)
        row_idx = idx.unsqueeze(1)  # (seq_len, 1)
        col_idx = idx.unsqueeze(0)  # (1, seq_len)
        # floor(i / N) >= floor(j / N)
        attention_mask = (row_idx // token_per_frame >= col_idx // token_per_frame).unsqueeze(0)

        attn_mask = torch.zeros(attention_mask.shape, device=device)
        attn_mask.masked_fill_(attention_mask.logical_not(), float('-inf'))

        linear_bias = self.config.slope_scale * get_relative_positions(seq_len)
        linear_bias.masked_fill_(attention_mask.logical_not(), 0)

        attn_mask += linear_bias
        return attn_mask.to(dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor = None,
        context_cache={'kv_cache': None},
        conditions: torch.LongTensor = None,
        return_dict: bool = True,
        encoder_output: torch.Tensor = None,
        debug: bool = False,
        noise_strength: float = 0.0,
        guidance_scale: float = 1.0,
    ):
        batch_size, num_frames, _, height, width = hidden_states.shape
        token_per_frame = (height // self.config.patch_size) * (width // self.config.patch_size)

        # Pack latent sequence and compute RoPE
        hidden_states = self._pack_latent_sequence(hidden_states, patch_size=self.config.patch_size)
        latent_seq_ids = self._prepare_latent_sequence_ids(
            batch_size,
            num_frames,
            height,
            width,
            patch_size=self.config.patch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype)

        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1).repeat((1, num_frames))

        if self.decouple_type == "encoder" or self.decouple_type == "causal":
            if context_cache['kv_cache'] is not None:
                if context_cache['is_cache_step'] is True:
                    current_seq_len = hidden_states.shape[1] - context_cache['cached_seqlen']
                    context_cache['cached_seqlen'] = hidden_states.shape[1]

                    if current_seq_len == 0:
                        raise NotImplementedError("Encoder received zero new tokens")
                    hidden_states = hidden_states[:, -current_seq_len:, ...]
                    timestep = timestep[:, -(current_seq_len // token_per_frame):]

                    if self.config.condition_cfg is not None and self.config.condition_cfg['type'] == 'action':
                        conditions['action'] = conditions['action'][:, -(current_seq_len // token_per_frame):]
                else:
                    assert context_cache['cached_seqlen'] == hidden_states.shape[1]
                    raise NotImplementedError("Cache retrieval without new encoding is not supported")

        else:
            assert context_cache['kv_cache'] is None

        # Generate attention mask
        if self.decouple_type == "encoder":
            attention_mask = self._build_causal_mask(
                input_shape=(batch_size, num_frames, num_frames * token_per_frame),
                device=hidden_states.device,
                dtype=hidden_states.dtype)
        else:
            attention_mask = None

        # Input projection
        hidden_states = self.x_embedder(hidden_states)
        base_seq_len = hidden_states.shape[1]
        seq_ids_for_rotary = latent_seq_ids
        use_token_concat = self.decouple_type == "decoder" and self.decoder_input_combine in {"token_concat", "token_concat_with_proj"}

        if self.decouple_type == "decoder":
            assert encoder_output is not None
            assert hidden_states.shape == encoder_output.shape, \
                f"hidden_states and encoder_output must have the same shape, but got {hidden_states.shape} and {encoder_output.shape}"

            if self.decoder_input_combine == "concat":
                decoder_input_concat = torch.cat([hidden_states, encoder_output], dim=-1)
                hidden_states = self.decoder_input_proj(decoder_input_concat)
                assert hidden_states.shape[1] == base_seq_len, "Decoder concat should preserve sequence length"
            elif self.decoder_input_combine == "add":
                hidden_states = hidden_states + encoder_output
            elif self.decoder_input_combine in {"token_concat", "token_concat_with_proj"}:
                aligned_encoder = encoder_output
                if self.decoder_alignment is not None:
                    assert aligned_encoder.shape[-1] == self.inner_dim, "Encoder alignment expects inner_dim features"
                    aligned_encoder = self.decoder_alignment(aligned_encoder)
                hidden_states = torch.cat([aligned_encoder, hidden_states], dim=1)
                assert hidden_states.shape[1] == base_seq_len * 2, "Token concat must double the sequence length"
                seq_ids_for_rotary = torch.cat([latent_seq_ids, latent_seq_ids], dim=0)
            else:
                raise NotImplementedError

        seq_rotary_emb = self.pos_embed(seq_ids_for_rotary)

        # noise timestep embedding
        timestep = rearrange(timestep, 'b t -> (b t)')
        timestep_proj = self.time_proj(timestep.to(hidden_states.dtype))
        temb = self.timestep_embedder(timestep_proj.to(dtype=hidden_states.dtype))  # (N, D)
        temb = rearrange(temb, '(b t) c -> b t c', b=batch_size).repeat_interleave(token_per_frame, dim=1)
        if use_token_concat:
            temb = torch.cat([temb, temb], dim=1)

        if self.config.condition_cfg is not None:
            if self.config.condition_cfg['type'] == 'label':
                label_emb = self.label_embedder(conditions['label']).unsqueeze(1)
                temb = temb + label_emb
            elif self.config.condition_cfg['type'] == 'action':
                action = rearrange(conditions['action'], 'b t -> (b t)')
                action_emb = self.action_embedder(action)
                action_emb = rearrange(action_emb, '(b t) c -> b t c', b=batch_size)
                action_emb = action_emb.repeat_interleave(token_per_frame, dim=1)
                if use_token_concat:
                    action_emb = torch.cat([action_emb, action_emb], dim=1)
                temb = temb + action_emb
            else:
                raise NotImplementedError

        if use_token_concat:
            assert temb.shape[1] == hidden_states.shape[1], "Temporally expanded embeddings must align with hidden states"


        for index_block, block in enumerate(self.transformer_blocks):

            if context_cache['kv_cache'] is None:
                layer_kv_cache = {'kv_cache': None}
            elif index_block not in context_cache['kv_cache']:
                layer_kv_cache = {
                    'is_cache_step': context_cache['is_cache_step'],
                    'kv_cache': {},
                    'token_per_frame': token_per_frame
                }
            else:
                layer_kv_cache = {
                    'is_cache_step': context_cache['is_cache_step'],
                    'kv_cache': context_cache['kv_cache'][index_block],
                    'token_per_frame': token_per_frame
                }

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
                hidden_states, layer_kv_cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    seq_rotary_emb,
                    attention_mask,
                    layer_kv_cache,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, layer_kv_cache = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=seq_rotary_emb,
                    attention_mask=attention_mask,
                    layer_kv_cache=layer_kv_cache)

            if context_cache['kv_cache'] is not None:
                context_cache['kv_cache'][index_block] = layer_kv_cache['kv_cache']

        hidden_states = self.norm_out(hidden_states, temb, inference_noise_strength=noise_strength, guidance_scale=guidance_scale)
        if use_token_concat:
            hidden_states = hidden_states[:, base_seq_len:, :]
            assert hidden_states.shape[1] == base_seq_len, "Slicing after token concat must restore base sequence length"
        if self.decouple_type == "encoder":
            output = hidden_states
        else:
            output = self.proj_out(hidden_states)

            if context_cache['kv_cache'] is not None:
                output = output[:, -token_per_frame:, :]
                output = self._unpack_latent_sequence(output, num_frames=1, height=height, width=width)
            else:
                output = self._unpack_latent_sequence(output, num_frames=num_frames, height=height, width=width)
                if not self.training:
                    output = output[:, -1:, ...]

        if not return_dict:
            return (output, context_cache)

        return SimpleNamespace(sample=output, context_cache=context_cache)
    
@MODEL_REGISTRY.register()
def SCD_B(**kwargs):
    in_channels = kwargs.pop('in_channels', 32)
    out_channels = kwargs.pop('out_channels', 32)
    patch_size = kwargs.pop('patch_size', 1)
    return SCDTransformer(in_channels=in_channels, out_channels=out_channels, num_layers=12, attention_head_dim=64, patch_size=patch_size, num_attention_heads=12, **kwargs)

@MODEL_REGISTRY.register()
def SCD_B_decoder(**kwargs):
    in_channels = kwargs.pop('in_channels', 32)
    out_channels = kwargs.pop('out_channels', 32)
    patch_size = kwargs.pop('patch_size', 1)
    return SCDTransformer(in_channels=in_channels, out_channels=out_channels, attention_head_dim=64, patch_size=patch_size, num_attention_heads=12, **kwargs)

@MODEL_REGISTRY.register()
def SCD_B_encoder(**kwargs):
    in_channels = kwargs.pop('in_channels', 32)
    out_channels = kwargs.pop('out_channels', 32)
    patch_size = kwargs.pop('patch_size', 1)
    return SCDTransformer(in_channels=in_channels, out_channels=out_channels, attention_head_dim=64, patch_size=patch_size, num_attention_heads=12, **kwargs)

@MODEL_REGISTRY.register()
def SCD_M(**kwargs):
    return SCDTransformer(in_channels=32, out_channels=32, num_layers=12, attention_head_dim=64, patch_size=1, num_attention_heads=16, **kwargs)

@MODEL_REGISTRY.register()
def SCD_M_decoder(**kwargs):
    return SCDTransformer(in_channels=32, out_channels=32, attention_head_dim=64, patch_size=1, num_attention_heads=16, **kwargs)

@MODEL_REGISTRY.register()
def SCD_M_encoder(**kwargs):
    return SCDTransformer(in_channels=32, out_channels=32, attention_head_dim=64, patch_size=1, num_attention_heads=16, **kwargs)

@MODEL_REGISTRY.register()
def SCD_L(**kwargs):
    return SCDTransformer(in_channels=32, out_channels=32, num_layers=24, attention_head_dim=64, patch_size=1, num_attention_heads=16, **kwargs)

@MODEL_REGISTRY.register()
def SCD_XL(**kwargs):
    return SCDTransformer(in_channels=32, out_channels=32, num_layers=28, attention_head_dim=64, patch_size=1, num_attention_heads=18, **kwargs)


class SCDEncoderDecoder(nn.Module):
    def __init__(self,
        encoder,
        decoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Store original dimensions info for decoder
        self._cached_dims = None
        self.lazy_encoder_output = None

    def __getattr__(self, name):
        """
        Override __getattr__ to inherit attributes from encoder.
        This allows accessing encoder's config and other attributes directly.
        """
        # First check if it's a normal attribute
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If not found, try to get from encoder
            if hasattr(self.encoder, name):
                return getattr(self.encoder, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward_train(
        self, 
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor = None,
        context_cache={'kv_cache': None},
        conditions: torch.LongTensor = None,
        return_dict: bool = True,
        context_timestep_idx = -1,
        hidden_states_encoder = None, 
        last_encoder_output = None,
    ):
        """
        Unified forward method for encoder-decoder architecture.
        
        The encoder processes the input and produces encoded features,
        which are then passed to the decoder for final prediction.
        
        Args:
            hidden_states: Input tensor of shape (B, T, C, H, W)
            timestep: Timestep tensor
            context_cache: Cache for KV attention (used by decoder)
            conditions: Conditional inputs (labels or actions)
            return_dict: Whether to return a dict or tuple
        
        Returns:
            Model output with decoded predictions
        """
        if context_cache is None:
            context_cache = {'kv_cache': None}

        if hidden_states_encoder is None:
            hidden_states_encoder = hidden_states

        if last_encoder_output is None:
            # Shift actions left by 1 frame, filling the last with the final action
            if conditions is not None and 'action' in conditions:
                conditions['action'] = torch.cat([conditions['action'][:, 1:], conditions['action'][:, -1:]], dim=1)

            encoder_output, context_cache = self.encoder(
                hidden_states=hidden_states_encoder,
                timestep=torch.ones_like(timestep) * context_timestep_idx,
                context_cache=context_cache,
                conditions=conditions,
                return_dict=False,
            )
        else:
            encoder_output = last_encoder_output.clone()

        batch_size, num_tokens, hidden_dim = encoder_output.shape
        channels, height, width = hidden_states.shape[2:]
        patch_size = getattr(self.encoder.config, 'patch_size', 1)
        tokens_per_frame = (height // patch_size) * (width // patch_size)
        num_frames = num_tokens // tokens_per_frame

        short_term_ctx_winsize = self.encoder.config.short_term_ctx_winsize
        ctx_win = min(short_term_ctx_winsize, num_frames - 1)

        encoder_output_shifted = encoder_output.reshape(batch_size, num_frames, tokens_per_frame, hidden_dim)
        encoder_output_shifted = encoder_output_shifted[:, -ctx_win-1:-1, ...]
        encoder_output_shifted = encoder_output_shifted.reshape(batch_size * ctx_win, tokens_per_frame, hidden_dim)

        hidden_states_decoder = hidden_states[:, -ctx_win:, ...]
        hidden_states_decoder = hidden_states_decoder.reshape(batch_size * ctx_win, 1, channels, height, width)

        timestep_decoder = timestep[:, -ctx_win:]
        timestep_decoder = timestep_decoder.reshape(batch_size * ctx_win, 1)

        conditions_decoder = None
        if conditions is not None:
            conditions_decoder = conditions.copy()
            if 'action' in conditions_decoder:
                conditions_decoder['action'] = conditions_decoder['action'][:, -ctx_win:]
                conditions_decoder['action'] = conditions_decoder['action'].reshape(batch_size * ctx_win, 1)
            if 'label' in conditions_decoder:
                labels = conditions_decoder['label']
                if labels.ndim > 1:
                    labels = labels.reshape(-1)
                else:
                    labels = labels.flatten()
                conditions_decoder['label'] = labels.repeat_interleave(ctx_win)

        decoder_output, _ = self.decoder(
            hidden_states=hidden_states_decoder,
            timestep=timestep_decoder,
            context_cache={'kv_cache': None},
            conditions=conditions_decoder,
            return_dict=False,
            encoder_output=encoder_output_shifted
        )

        decoder_output = decoder_output.reshape(batch_size, ctx_win, channels, height, width)

        if not return_dict:
            return decoder_output, context_cache
        else:
            return SimpleNamespace(sample=decoder_output, context_cache=context_cache, encoder_output=encoder_output)
    
    def forward_eval(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor = None,
        context_cache={'kv_cache': None},
        conditions: torch.LongTensor = None,
        return_dict: bool = True,
        noisy_current_frame = None,
        context_timestep_idx = -1,
        debug = False,
        cfg_type = 'action',
        guidance_scale = 1.0,
    ):
        if context_cache is None:
            context_cache = {'kv_cache': None}

        conditions_decoder = None
        conditions_encoder = None
        if conditions is not None:
            conditions_decoder = conditions.copy()
            conditions_encoder = conditions.copy()

            if 'action' in conditions_decoder:
                conditions_decoder['action'] = conditions_decoder['action'][:, -1:]
                conditions_encoder['action'] = conditions_encoder['action'][:, 1:]
                assert conditions_encoder['action'].shape == (hidden_states.shape[0], hidden_states.shape[1]), \
                    f"conditions_encoder['action'].shape: {conditions_encoder['action'].shape}, hidden_states.shape: {hidden_states.shape}"

            if 'label' in conditions_decoder:
                conditions_decoder['label'] = conditions_decoder['label'].flatten()
                conditions_encoder['label'] = conditions_encoder['label'].flatten()

        if context_cache['is_cache_step'] is True:
            if cfg_type == 'action':
                encoder_output, context_cache = self.encoder(
                    hidden_states=hidden_states,
                    timestep=torch.ones_like(timestep) * context_timestep_idx,
                    context_cache=context_cache,
                    conditions=conditions_encoder,
                    return_dict=False,
                    debug=debug
                )
            elif cfg_type == 'decoder':
                noise_strength = max(self.encoder.config.noise_strength, 0.05)
                encoder_output, context_cache = self.encoder(
                    hidden_states=hidden_states,
                    timestep=torch.ones_like(timestep) * context_timestep_idx,
                    context_cache=context_cache,
                    conditions=conditions_encoder,
                    return_dict=False,
                    debug=debug,
                    guidance_scale=guidance_scale,
                    noise_strength=noise_strength,
                )

            self.lazy_encoder_output = encoder_output.clone()
        else:
            encoder_output = self.lazy_encoder_output.clone()

        batch_size, num_tokens, hidden_dim = encoder_output.shape
        patch_size = getattr(self.encoder.config, 'patch_size', 1)
        height, width = hidden_states.shape[-2:]
        tokens_per_frame = (height // patch_size) * (width // patch_size)
        num_frames = num_tokens // tokens_per_frame

        short_term_ctx_winsize = self.encoder.config.short_term_ctx_winsize
        encoder_output_shifted = encoder_output.reshape(batch_size, num_frames, tokens_per_frame, hidden_dim)
        encoder_output_shifted = encoder_output_shifted[:, -1, ...]

        hidden_states_decoder = noisy_current_frame[:, -1:, ...]
        timestep_decoder = timestep[:, -1:]

        decoder_output, _ = self.decoder(
            hidden_states=hidden_states_decoder,
            timestep=timestep_decoder,
            context_cache={'kv_cache': None},
            conditions=conditions_decoder,
            return_dict=False,
            encoder_output=encoder_output_shifted,
            debug=debug
        )

        if not return_dict:
            return decoder_output, context_cache
        else:
            return SimpleNamespace(sample=decoder_output, context_cache=context_cache)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for both encoder and decoder."""
        if hasattr(self.encoder, 'enable_gradient_checkpointing'):
            self.encoder.enable_gradient_checkpointing()
        if hasattr(self.decoder, 'enable_gradient_checkpointing'):
            self.decoder.enable_gradient_checkpointing()