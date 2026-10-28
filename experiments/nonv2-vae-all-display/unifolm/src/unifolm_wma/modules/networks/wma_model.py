import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from functools import partial
from abc import abstractmethod
from einops import rearrange
from omegaconf import OmegaConf
from typing import Optional, Sequence, Any, Tuple, Union, List, Dict
from collections.abc import Mapping, Iterable, Callable

from unifolm_wma.utils.diffusion import timestep_embedding
from unifolm_wma.utils.common import checkpoint
from unifolm_wma.utils.basics import (zero_module, conv_nd, linear,
                                      avg_pool_nd, normalization)
from unifolm_wma.modules.attention import (CrossAttention, SpatialTransformer,
                                           TemporalTransformer)
from unifolm_wma.utils.utils import instantiate_from_config


@contextmanager
def _temporary_cudnn_tf32(enabled):
    old = torch.backends.cudnn.allow_tf32
    try:
        if enabled:
            torch.backends.cudnn.allow_tf32 = True
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_temporal_conv: if True, use the temporal convolution.
    :param use_image_dataset: if True, the temporal parameters will not be optimized.
    """

    def __init__(self,
                 channels,
                 emb_channels,
                 dropout,
                 out_channels=None,
                 use_scale_shift_norm=False,
                 dims=2,
                 use_checkpoint=False,
                 use_conv=False,
                 up=False,
                 down=False,
                 use_temporal_conv=False,
                 tempspatial_aware=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv
        self._fused_groupnorm_silu = False

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           3,
                                           padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels,
                                           1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware)

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, emb)
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple,
                              self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x, emb, batch_size=None):
        if self._fused_groupnorm_silu:
            from unifolm_wma.utils.triton_group_norm import fused_group_norm
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            if self._fused_groupnorm_silu:
                h = fused_group_norm(x, self.in_layers[0])
            else:
                h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            if self._fused_groupnorm_silu:
                h = self.in_layers[2](fused_group_norm(x, self.in_layers[0]))
            else:
                h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            if self._fused_groupnorm_silu:
                h = fused_group_norm(
                    h, out_norm, scale=scale, shift=shift, apply_silu=True)
                h = self.out_layers[2:](h)
            else:
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
        else:
            h = h + emb_out
            if self._fused_groupnorm_silu:
                h = fused_group_norm(h, self.out_layers[0])
                h = self.out_layers[2:](h)
            else:
                h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TemporalConvBlock(nn.Module):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 dropout=0.0,
                 spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._fused_framewise_norm = False
        self._fused_exact_temporal_norm = False
        self._inference_conv_dtype = None
        th_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 1)
        th_padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 0)
        tw_kernel_shape = (3, 1, 1) if not spatial_aware else (3, 1, 3)
        tw_padding_shape = (1, 0, 0) if not spatial_aware else (1, 0, 1)

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels,
                      out_channels,
                      th_kernel_shape,
                      padding=th_padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      tw_kernel_shape,
                      padding=tw_padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      th_kernel_shape,
                      padding=th_padding_shape))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      in_channels,
                      tw_kernel_shape,
                      padding=tw_padding_shape))

        # Zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x

        def apply_conv(value, conv):
            if self._inference_conv_dtype is None:
                return conv(value)
            output_dtype = value.dtype
            return conv(value.to(self._inference_conv_dtype)).to(output_dtype)

        if self._fused_framewise_norm or self._fused_exact_temporal_norm:
            from unifolm_wma.utils.triton_group_norm import (
                fused_framewise_group_norm_silu,
                fused_temporal_group_norm_silu)

            def fused_block(value, block):
                if self._fused_exact_temporal_norm:
                    value = fused_temporal_group_norm_silu(value, block[0])
                else:
                    value = fused_framewise_group_norm_silu(value, block[0])
                value = block[2:-1](value)
                return apply_conv(value, block[-1])

            x = fused_block(x, self.conv1)
            x = fused_block(x, self.conv2)
            x = fused_block(x, self.conv3)
            x = fused_block(x, self.conv4)
        else:
            def mixed_block(value, block):
                if self._inference_conv_dtype is None:
                    return block(value)
                value = block[:-1](value)
                return apply_conv(value, block[-1])

            x = mixed_block(x, self.conv1)
            x = mixed_block(x, self.conv2)
            x = mixed_block(x, self.conv3)
            x = mixed_block(x, self.conv4)

        return identity + x


class WMAModel(nn.Module):
    """
    The full World-Model-Action model.
    """

    def __init__(self,
                 in_channels: int,
                 model_channels: int,
                 out_channels: int,
                 num_res_blocks: int,
                 attention_resolutions: Sequence[int],
                 dropout: float = 0.0,
                 channel_mult: Sequence[int] = (1, 2, 4, 8),
                 conv_resample: bool = True,
                 dims: int = 2,
                 context_dim: int | None = None,
                 use_scale_shift_norm: bool = False,
                 resblock_updown: bool = False,
                 num_heads: int = -1,
                 num_head_channels: int = -1,
                 transformer_depth: int = 1,
                 use_linear: bool = False,
                 use_checkpoint: bool = False,
                 temporal_conv: bool = False,
                 tempspatial_aware: bool = False,
                 temporal_attention: bool = True,
                 use_relative_position: bool = True,
                 use_causal_attention: bool = False,
                 temporal_length: int | None = None,
                 use_fp16: bool = False,
                 addition_attention: bool = False,
                 temporal_selfatt_only: bool = True,
                 image_cross_attention: bool = False,
                 cross_attention_scale_learnable: bool = False,
                 default_fs: int = 4,
                 fs_condition: bool = False,
                 n_obs_steps: int = 1,
                 num_stem_token: int = 1,
                 unet_head_config: OmegaConf | None = None,
                 stem_process_config: OmegaConf | None = None,
                 base_model_gen_only: bool = False):
        """
        Initialize the World-Model-Action network.

        Args:
            in_channels: Number of input channels to the backbone.
            model_channels: Base channel width for the UNet/backbone.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks per resolution stage.
            attention_resolutions: Resolutions at which to enable attention.
            dropout: Dropout probability used inside residual/attention blocks.
            channel_mult: Multipliers for channels at each resolution level.
            conv_resample: If True, use convolutional resampling for up/down sampling.
            dims: Spatial dimensionality of the backbone (1/2/3).
            context_dim: Optional context embedding dimension (for cross-attention).
            use_scale_shift_norm: Enable scale-shift (FiLM-style) normalization in blocks.
            resblock_updown: Use residual blocks for up/down sampling (instead of plain conv).
            num_heads: Number of attention heads (if >= 0). If -1, derive from num_head_channels.
            num_head_channels: Channels per attention head (if >= 0). If -1, derive from num_heads.
            transformer_depth: Number of transformer/attention blocks per stage.
            use_linear: Use linear attention variants where applicable.
            use_checkpoint: Enable gradient checkpointing in blocks to save memory.
            temporal_conv: Include temporal convolution along the time dimension.
            tempspatial_aware: If True, use time–space aware blocks.
            temporal_attention: Enable temporal self-attention.
            use_relative_position: Use relative position encodings in attention.
            use_causal_attention: Use causal (uni-directional) attention along time.
            temporal_length: Optional maximum temporal length expected by the model.
            use_fp16: Enable half-precision layers/normalization where supported.
            addition_attention: Add auxiliary attention modules.
            temporal_selfatt_only: Restrict attention to temporal-only (no spatial) if True.
            image_cross_attention: Enable cross-attention with image embeddings.
            cross_attention_scale_learnable: Make cross-attention scaling a learnable parameter.
            default_fs: Default frame-stride / fps.
            fs_condition: If True, condition on frame-stride/fps features.
            n_obs_steps: Number of observed steps used in conditioning heads.
            num_stem_token: Number of stem tokens for action tokenization.
            unet_head_config: OmegaConf for UNet heads (e.g., action/state heads).
            stem_process_config: OmegaConf for stem/preprocessor module.
            base_model_gen_only: Perform the generation using the base model with out action and state outputs.
        """

        super(WMAModel, self).__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.cross_attention_scale_learnable = cross_attention_scale_learnable
        self.default_fs = default_fs
        self.fs_condition = fs_condition
        self.n_obs_steps = n_obs_steps
        self.num_stem_token = num_stem_token
        self.base_model_gen_only = base_model_gen_only

        # Time embedding blocks
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if fs_condition:
            self.fps_embedding = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        # Input Block
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(model_channels,
                                    n_heads=8,
                                    d_head=num_head_channels,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    use_checkpoint=use_checkpoint,
                                    only_self_att=temporal_selfatt_only,
                                    causal_attention=False,
                                    relative_position=use_relative_position,
                                    temporal_length=temporal_length))

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch,
                             time_embed_dim,
                             dropout,
                             out_channels=mult * model_channels,
                             dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm,
                             tempspatial_aware=tempspatial_aware,
                             use_temporal_conv=temporal_conv)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            agent_state_context_len=self.n_obs_steps,
                            agent_action_context_len=self.temporal_length *
                            num_stem_token,
                            image_cross_attention=self.image_cross_attention,
                            cross_attention_scale_learnable=self.
                            cross_attention_scale_learnable,
                        ))
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,
                                 time_embed_dim,
                                 dropout,
                                 out_channels=out_ch,
                                 dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 down=True)
                        if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        layers = [
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     tempspatial_aware=tempspatial_aware,
                     use_temporal_conv=temporal_conv),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                use_linear=use_linear,
                use_checkpoint=use_checkpoint,
                disable_self_attn=False,
                video_length=temporal_length,
                agent_state_context_len=self.n_obs_steps,
                agent_action_context_len=self.temporal_length * num_stem_token,
                image_cross_attention=self.image_cross_attention,
                cross_attention_scale_learnable=self.
                cross_attention_scale_learnable)
        ]
        if self.temporal_attention:
            layers.append(
                TemporalTransformer(ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    use_linear=use_linear,
                                    use_checkpoint=use_checkpoint,
                                    only_self_att=temporal_self_att_only,
                                    causal_attention=use_causal_attention,
                                    relative_position=use_relative_position,
                                    temporal_length=temporal_length))
        layers.append(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     tempspatial_aware=tempspatial_aware,
                     use_temporal_conv=temporal_conv))

        # Middle Block
        self.middle_block = TimestepEmbedSequential(*layers)

        # Output Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich,
                             time_embed_dim,
                             dropout,
                             out_channels=mult * model_channels,
                             dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm,
                             tempspatial_aware=tempspatial_aware,
                             use_temporal_conv=temporal_conv)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_linear=use_linear,
                            use_checkpoint=use_checkpoint,
                            disable_self_attn=False,
                            video_length=temporal_length,
                            agent_state_context_len=self.n_obs_steps,
                            image_cross_attention=self.image_cross_attention,
                            cross_attention_scale_learnable=self.
                            cross_attention_scale_learnable))
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                use_linear=use_linear,
                                use_checkpoint=use_checkpoint,
                                only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention,
                                relative_position=use_relative_position,
                                temporal_length=temporal_length))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch,
                                 time_embed_dim,
                                 dropout,
                                 out_channels=out_ch,
                                 dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 up=True)
                        if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # Action and state prediction unet
        unet_head_config['params']['context_dims'] = [
            mult * model_channels for mult in channel_mult
        ]
        self.action_unet = instantiate_from_config(unet_head_config)
        self.state_unet = instantiate_from_config(unet_head_config)

        # Initialize action token_projector
        self.action_token_projector = instantiate_from_config(
            stem_process_config)
        self._condition_cache_enabled = False
        self._prepared_context_cache = None
        self._channels_last_2d_enabled = False
        self._fixed_cast_cache_enabled = False
        self._fixed_cast_cache = {}
        self._fp32_feedback_heads = False

    def _prepare_inference_condition_dtype(self, context, context_action):
        def cast(name, value):
            if not self._fixed_cast_cache_enabled or self.training or torch.is_grad_enabled():
                return value.to(torch.float16)
            from unifolm_wma.utils.fixed_condition_cast import cast_fixed_condition
            return cast_fixed_condition(self._fixed_cast_cache, name, value,
                                        torch.float16)

        if context is not None:
            context = cast('cross_attention_context', context)
        if context_action is not None and not getattr(self, '_fp32_feedback_heads', False):
            context_action = type(context_action)(
                cast('action_context_' + str(index), item)
                if isinstance(item, torch.Tensor) and item.is_floating_point()
                else item for index, item in enumerate(context_action))
        return context, context_action

    def _prepare_inference_context(self, context, t):
        """Prepare timestep-invariant context, optionally reusing its storage."""
        _, l_context, _ = context.shape
        key = (context.data_ptr(), context._version, tuple(context.shape),
               tuple(context.stride()), context.dtype, context.device, t)
        use_cache = (self._condition_cache_enabled and not self.training and
                     not torch.is_grad_enabled())
        if (use_cache and self._prepared_context_cache is not None and
                self._prepared_context_cache["key"] == key):
            return self._prepared_context_cache["output"]

        if self.base_model_gen_only:
            assert l_context == 77 + self.n_obs_steps * 16, ">>> ERROR Context dim 1 ..."
            prepared = context
        elif l_context == self.n_obs_steps + 77 + t * 16:
            context_agent_state = context[:, :self.n_obs_steps]
            context_text = context[:, self.n_obs_steps:self.n_obs_steps + 77, :]
            context_img = context[:, self.n_obs_steps + 77:, :]
            context_agent_state = context_agent_state.repeat_interleave(
                repeats=t, dim=0)
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img,
                                    'b (t l) c -> (b t) l c', t=t)
            prepared = torch.cat(
                [context_agent_state, context_text, context_img], dim=1)
        elif l_context == self.n_obs_steps + 16 + 77 + t * 16:
            context_agent_state = context[:, :self.n_obs_steps]
            context_agent_action = context[:, self.n_obs_steps:self.n_obs_steps + 16, :]
            context_agent_action = rearrange(
                context_agent_action.unsqueeze(2), 'b t l d -> (b t) l d')
            context_agent_action = self.action_token_projector(
                context_agent_action)
            context_agent_action = rearrange(
                context_agent_action, '(b o) l d -> b o l d', o=t)
            context_agent_action = rearrange(
                context_agent_action, 'b o (t l) d -> b o t l d', t=t)
            context_agent_action = context_agent_action.permute(0, 2, 1, 3, 4)
            context_agent_action = rearrange(
                context_agent_action, 'b t o l d -> (b t) (o l) d')
            context_text = context[:, self.n_obs_steps + 16:
                                   self.n_obs_steps + 16 + 77, :]
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = context[:, self.n_obs_steps + 16 + 77:, :]
            context_img = rearrange(context_img,
                                    'b (t l) c -> (b t) l c', t=t)
            context_agent_state = context_agent_state.repeat_interleave(
                repeats=t, dim=0)
            prepared = torch.cat([
                context_agent_state, context_agent_action, context_text,
                context_img
            ], dim=1)
        else:
            raise ValueError(f"Unexpected context length: {l_context}")

        if use_cache:
            cached = self._prepared_context_cache
            if (cached is not None and cached["output"].shape == prepared.shape and
                    cached["output"].dtype == prepared.dtype and
                    cached["output"].device == prepared.device):
                cached["output"].copy_(prepared)
                output = cached["output"]
            else:
                output = prepared
            self._prepared_context_cache = {
                "key": key, "source": context, "output": output,
            }
            return output
        return prepared

    def prepare_condition_cache(self, context, t, context_action=None,
                                action_shape=None):
        """Refresh only fixed context buffers used by captured attention."""
        prepared = self._prepare_inference_context(context, t)
        count = 0
        for module in self.modules():
            if type(module) is CrossAttention and module.image_cross_attention:
                module.prepare_condition_cache(prepared)
                count += 1
        if context_action is not None:
            if action_shape is None:
                raise ValueError("action_shape is required with context_action")
            batch_size, horizon = action_shape
            fixed = context_action[:2]
            skip_unused_heads = (
                os.environ.get("UNIFOLM_SKIP_UNUSED_HEADS", "0") == "1")
            sim_mode = bool(context_action[2])
            if not skip_unused_heads or not sim_mode:
                self.action_unet.prepare_condition_cache(
                    fixed, batch_size, horizon)
            if not skip_unused_heads or sim_mode:
                self.state_unet.prepare_condition_cache(
                    fixed, batch_size, horizon)
        return count

    def forward(self,
                x: Tensor,
                x_action: Tensor,
                x_state: Tensor,
                timesteps: Tensor,
                context: Tensor | None = None,
                context_action: Tensor | None = None,
                features_adapter: Any = None,
                fs: Tensor | None = None,
                **kwargs) -> Tensor | tuple[Tensor, ...]:

        """
        Forward pass of the World-Model-Action backbone.

        Args:
            x: Input tensor (latent video), shape (B, C,...).
            x_action: action stream input.
            x_state: state stream input.
            timesteps: Diffusion timesteps, shape (B,) or scalar Tensor.
            context: conditioning context for cross-attention.
            context_action: conditioning context specific to action/state (implementation-specific).
            features_adapter: module or dict to adapt intermediate features.
            fs: frame-stride / fps conditioning.

        Returns:
            Tuple of Tensors for predictions:

        """

        if self.dtype == torch.float16:
            x = x.to(torch.float16)
            if not self._fp32_feedback_heads:
                x_action = x_action.to(torch.float16)
                x_state = x_state.to(torch.float16)
            context, context_action = self._prepare_inference_condition_dtype(
                context, context_action)

        b, _, t, _, _ = x.shape
        sim_mode = bool(context_action[2]) if context_action is not None else False
        sim_base_tf32 = os.environ.get("UNIFOLM_SIM_BASE_CUDNN_TF32", "none")
        sim_base_tf32_scopes = (set() if sim_base_tf32 == "none" else
                                set(sim_base_tf32.split("+")))
        if not sim_base_tf32_scopes <= {
                "input", "middle", "output", "output_first", "output_last",
                "output_q1", "output_q2", "output_q3", "output_q4"}:
            raise ValueError(
                "UNIFOLM_SIM_BASE_CUDNN_TF32 must be none or a + separated "
                "combination of input, middle, output, output_first, and "
                "output_last, or output_q1 through output_q4")
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)

        context = self._prepare_inference_context(context, t)

        emb = emb.repeat_interleave(repeats=t, dim=0)

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if self._channels_last_2d_enabled:
            x = x.contiguous(memory_format=torch.channels_last)

        # Combine emb
        if self.fs_condition:
            if fs is None:
                fs = torch.tensor([self.default_fs] * b,
                                  dtype=torch.long,
                                  device=x.device)
            fs_emb = timestep_embedding(fs,
                                        self.model_channels,
                                        repeat_only=False).type(x.dtype)

            fs_embed = self.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        hs_a = []
        with _temporary_cudnn_tf32(
                sim_mode and "input" in sim_base_tf32_scopes):
            for id, module in enumerate(self.input_blocks):
                h = module(h, emb, context=context, batch_size=b)
                if id == 0 and self.addition_attention:
                    h = self.init_attn(h, emb, context=context, batch_size=b)
                # plug-in adapter features
                if ((id + 1) % 3 == 0) and features_adapter is not None:
                    h = h + features_adapter[adapter_idx]
                    adapter_idx += 1
                if id != 0:
                    if isinstance(module[0], Downsample):
                        hs_a.append(
                            rearrange(hs[-1], '(b t) c h w -> b t c h w', t=t))
                hs.append(h)
        hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))

        if features_adapter is not None:
            assert len(
                features_adapter) == adapter_idx, 'Wrong features_adapter'
        with _temporary_cudnn_tf32(
                sim_mode and "middle" in sim_base_tf32_scopes):
            h = self.middle_block(h, emb, context=context, batch_size=b)
        hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))

        hs_out = []
        output_split = len(self.output_blocks) // 2
        output_quarter = max(1, len(self.output_blocks) // 4)
        for output_id, module in enumerate(self.output_blocks):
            output_quarter_id = min(4, output_id // output_quarter + 1)
            output_tf32 = (
                "output" in sim_base_tf32_scopes or
                ("output_first" in sim_base_tf32_scopes and
                 output_id < output_split) or
                ("output_last" in sim_base_tf32_scopes and
                 output_id >= output_split) or
                f"output_q{output_quarter_id}" in sim_base_tf32_scopes)
            with _temporary_cudnn_tf32(sim_mode and output_tf32):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context=context, batch_size=b)
                if isinstance(module[-1], Upsample):
                    hs_a.append(
                        rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
                hs_out.append(h)
        h = h.type(x.dtype)
        hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))

        y = self.out(h)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)

        if not self.base_model_gen_only:
            if self._fp32_feedback_heads:
                hs_a = [feature.float() for feature in hs_a]
            ba, _, _ = x_action.shape
            state_timesteps = timesteps[:ba] if b > 1 else timesteps
            skip_unused_heads = os.environ.get(
                "UNIFOLM_SKIP_UNUSED_HEADS", "0")
            if skip_unused_heads not in {"0", "1"}:
                raise ValueError("UNIFOLM_SKIP_UNUSED_HEADS must be 0 or 1")
            if skip_unused_heads == "1":
                if sim_mode:
                    a_y = torch.zeros_like(x_action)
                    head_tf32 = os.environ.get("UNIFOLM_SIM_HEAD_TF32", "0")
                    if head_tf32 not in {"0", "1"}:
                        raise ValueError("UNIFOLM_SIM_HEAD_TF32 must be 0 or 1")
                    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                    try:
                        if head_tf32 == "1":
                            torch.backends.cudnn.allow_tf32 = True
                        s_y = self.state_unet(x_state, state_timesteps, hs_a,
                                              context_action[:2], **kwargs)
                    finally:
                        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
                else:
                    a_y = self.action_unet(x_action, timesteps[:ba], hs_a,
                                           context_action[:2], **kwargs)
                    s_y = torch.zeros_like(x_state)
            else:
                a_y = self.action_unet(x_action, timesteps[:ba], hs_a,
                                       context_action[:2], **kwargs)
                s_y = self.state_unet(x_state, state_timesteps, hs_a,
                                      context_action[:2], **kwargs)
        else:
            a_y = torch.zeros_like(x_action)
            s_y = torch.zeros_like(x_state)

        return y, a_y, s_y
