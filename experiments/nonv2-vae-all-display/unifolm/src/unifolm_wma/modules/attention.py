import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from functools import partial

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from unifolm_wma.utils.common import (
    checkpoint,
    exists,
    default,
)
from unifolm_wma.utils.basics import zero_module


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -self.max_relative_position,
                                           self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 relative_position=False,
                 temporal_length=None,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 image_cross_attention_scale=1.0,
                 agent_state_cross_attention_scale=1.0,
                 agent_action_cross_attention_scale=1.0,
                 cross_attention_scale_learnable=False,
                 text_context_len=77):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

        self.relative_position = relative_position
        self.temporal_length = temporal_length
        if self.relative_position:
            assert (temporal_length is not None)
            self.relative_position_k = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(
                num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.agent_state_cross_attention_scale = agent_state_cross_attention_scale
        self.agent_action_cross_attention_scale = agent_action_cross_attention_scale
        self.text_context_len = text_context_len
        self.agent_state_context_len = agent_state_context_len
        self.agent_action_context_len = agent_action_context_len
        self.cross_attention_scale_learnable = cross_attention_scale_learnable
        # Inference masks depend only on shape, device and dtype. Keep the
        # query dimension broadcast instead of materializing it per DDIM step.
        self._action_mask_cache = {}
        self._condition_cache_enabled = False
        self._condition_projection_cache = {}
        self._fused_self_qkv_enabled = False
        self._fused_self_qkv_cache = None
        self._torch_cross_attention_enabled = False
        self._torch_small_k_enabled = False
        self._inference_attention_layout = "packed"
        self._inference_fused_mix = False
        self._inference_fused_temporal_attention = False
        self._inference_attention_dtype = torch.float32
        self._inference_attention_self_only = False
        self._inference_projection_dtype = torch.float32
        self._lowp_projection_cache = {}
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_as = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_k_aa = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_aa = nn.Linear(context_dim, inner_dim, bias=False)
            if cross_attention_scale_learnable:
                self.register_parameter('alpha_ctx',
                                        nn.Parameter(torch.tensor(0.)))
                self.register_parameter('alpha_cas',
                                        nn.Parameter(torch.tensor(0.)))
                self.register_parameter('alpha_caa',
                                        nn.Parameter(torch.tensor(0.)))

    def forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        h = self.heads
        if spatial_self_attn and self._fused_self_qkv_enabled:
            q, k, v = self._project_self_qkv(x).chunk(3, dim=-1)
            context = x
        else:
            q = self.to_q(x)
            context = default(context, x)

        if self.image_cross_attention and not spatial_self_attn:
            assert 1 > 2, ">>> ERROR: should setup xformers and use efficient_forward ..."
            context_agent_state = context[:, :self.agent_state_context_len, :]
            context_agent_action = context[:,
                                           self.agent_state_context_len:self.
                                           agent_state_context_len +
                                           self.agent_action_context_len, :]
            context_ins = context[:, self.agent_state_context_len +
                                  self.agent_action_context_len:self.
                                  agent_state_context_len +
                                  self.agent_action_context_len +
                                  self.text_context_len, :]
            context_image = context[:, self.agent_state_context_len +
                                    self.agent_action_context_len +
                                    self.text_context_len:, :]

            k = self.to_k(context_ins)
            v = self.to_v(context_ins)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
            k_as = self.to_k_as(context_agent_state)
            v_as = self.to_v_as(context_agent_state)
            k_aa = self.to_k_aa(context_agent_action)
            v_aa = self.to_v_aa(context_agent_action)
        elif not (spatial_self_attn and self._fused_self_qkv_enabled):
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)

        if (self._inference_fused_temporal_attention and not self.training and
                not torch.is_grad_enabled() and spatial_self_attn and
                self.temporal_length == 16 and not self.relative_position and
                mask is None and self.dim_head == 64 and q.is_cuda and q.dtype == torch.float16 and
                q.shape[1] == 16 and all(tensor.is_contiguous() for tensor in (q,k,v))):
            from unifolm_wma.utils.triton_temporal_attention import fused_temporal_attention
            return self.to_out(fused_temporal_attention(q, k, v, h, self.scale))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q,
                          k2) * self.scale  # TODO check
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask > 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if k_ip is not None and k_as is not None and k_aa is not None:
            ## for image cross-attention
            k_ip, v_ip = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (k_ip, v_ip))
            sim_ip = torch.einsum('b i d, b j d -> b i j', q,
                                  k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

            ## for agent state cross-attention
            k_as, v_as = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (k_as, v_as))
            sim_as = torch.einsum('b i d, b j d -> b i j', q,
                                  k_as) * self.scale
            del k_as
            sim_as = sim_as.softmax(dim=-1)
            out_as = torch.einsum('b i j, b j d -> b i d', sim_as, v_as)
            out_as = rearrange(out_as, '(b h) n d -> b n (h d)', h=h)

            ## for agent action cross-attention
            k_aa, v_aa = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (k_aa, v_aa))
            sim_aa = torch.einsum('b i d, b j d -> b i j', q,
                                  k_aa) * self.scale
            del k_aa
            sim_aa = sim_aa.softmax(dim=-1)
            out_aa = torch.einsum('b i j, b j d -> b i d', sim_aa, v_aa)
            out_aa = rearrange(out_aa, '(b h) n d -> b n (h d)', h=h)

        if out_ip is not None and out_as is not None and out_aa is not None:
            if self.cross_attention_scale_learnable:
                out = out + \
                    self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha_ctx) + 1) + \
                    self.agent_state_cross_attention_scale * out_as * (torch.tanh(self.alpha_cas) + 1) + \
                    self.agent_action_cross_attention_scale * out_aa * (torch.tanh(self.alpha_caa) + 1)
            else:
                out = out + \
                    self.image_cross_attention_scale * out_ip + \
                    self.agent_state_cross_attention_scale * out_as + \
                    self.agent_action_cross_attention_scale * out_aa

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        input_dtype = x.dtype
        spatial_self_attn = (context is None)
        k, v, out = None, None, None
        k_ip, v_ip, out_ip = None, None, None
        k_as, v_as, out_as = None, None, None
        k_aa, v_aa, out_aa = None, None, None

        if spatial_self_attn and self._fused_self_qkv_enabled:
            q, k, v = self._project_self_qkv(x).chunk(3, dim=-1)
            context = x
        else:
            projection_x = x
            if self._inference_projection_dtype != x.dtype:
                projection_x = x.to(self._inference_projection_dtype)
            q = self._apply_projection(self.to_q, projection_x)
            context = projection_x if spatial_self_attn else context

        if self.image_cross_attention and not spatial_self_attn:
            if context.shape[1] == self.text_context_len + self.video_length:
                context_ins, context_image = context[:, :self.text_context_len, :], context[:,self.text_context_len:, :]
                k = self._project_condition("k", self.to_k, context)
                v = self._project_condition("v", self.to_v, context)
                k_ip = self._project_condition("k_ip", self.to_k_ip, context_image)
                v_ip = self._project_condition("v_ip", self.to_v_ip, context_image)
            elif context.shape[1] == self.agent_state_context_len + self.text_context_len + self.video_length:
                context_agent_state = context[:, :self.agent_state_context_len, :]
                context_ins = context[:, self.agent_state_context_len:self.agent_state_context_len+self.text_context_len, :]
                context_image = context[:, self.agent_state_context_len+self.text_context_len:, :]
                k = self._project_condition("k", self.to_k, context_ins)
                v = self._project_condition("v", self.to_v, context_ins)
                k_ip = self._project_condition("k_ip", self.to_k_ip, context_image)
                v_ip = self._project_condition("v_ip", self.to_v_ip, context_image)
                k_as = self._project_condition("k_as", self.to_k_as, context_agent_state)
                v_as = self._project_condition("v_as", self.to_v_as, context_agent_state)
            else:
                context_agent_state = context[:, :self.agent_state_context_len, :]
                context_agent_action = context[:, self.agent_state_context_len:self.agent_state_context_len+self.agent_action_context_len, :]
                context_ins = context[:, self.agent_state_context_len+self.agent_action_context_len:self.agent_state_context_len+self.agent_action_context_len+self.text_context_len, :]
                context_image = context[:, self.agent_state_context_len+self.agent_action_context_len+self.text_context_len:, :]

                k = self._project_condition("k", self.to_k, context_ins)
                v = self._project_condition("v", self.to_v, context_ins)
                k_ip = self._project_condition("k_ip", self.to_k_ip, context_image)
                v_ip = self._project_condition("v_ip", self.to_v_ip, context_image)
                k_as = self._project_condition("k_as", self.to_k_as, context_agent_state)
                v_as = self._project_condition("v_as", self.to_v_as, context_agent_state)
                k_aa = self._project_condition("k_aa", self.to_k_aa, context_agent_action)
                v_aa = self._project_condition("v_aa", self.to_v_aa, context_agent_action)

                attn_mask_aa = self._get_cached_attn_mask_aa(
                    x.shape[0], q.shape[1], k_aa.shape[1],
                    device=q.device,
                    dtype=(torch.float32
                           if self._inference_attention_self_only else
                           self._inference_attention_dtype))
        elif not (spatial_self_attn and self._fused_self_qkv_enabled):
            if not spatial_self_attn:
                assert 1 > 2, ">>> ERROR: you should never go into here ..."
                context = context[:, :self.text_context_len, :]
            k = self._apply_projection(self.to_k, context)
            v = self._apply_projection(self.to_v, context)

        b, _, _ = q.shape
        use_bmhk = (self._inference_attention_layout == "bmhk" and
                    not self.training and not torch.is_grad_enabled())

        def split_heads(value):
            if use_bmhk:
                return value.reshape(b, value.shape[1], self.heads, self.dim_head)
            return value.reshape(b, value.shape[1], self.heads,
                                 self.dim_head).permute(0, 2, 1, 3).reshape(
                                     b * self.heads, value.shape[1],
                                     self.dim_head).contiguous()

        def merge_heads(value):
            if use_bmhk:
                return value.reshape(b, value.shape[1], self.heads * self.dim_head)
            return value.reshape(b, self.heads, value.shape[1],
                                 self.dim_head).permute(0, 2, 1, 3).reshape(
                                     b, value.shape[1], self.heads * self.dim_head)

        q = split_heads(q)
        if k is not None:
            k, v = map(split_heads, (k, v))
            out = self._efficient_attention(
                q, k, v, use_torch=(self._torch_cross_attention_enabled and
                                    not spatial_self_attn))
            out = merge_heads(out)

        if k_ip is not None:
            # For image cross-attention
            k_ip, v_ip = map(split_heads, (k_ip, v_ip))
            out_ip = self._efficient_attention(
                q, k_ip, v_ip, use_torch=self._torch_cross_attention_enabled)
            out_ip = merge_heads(out_ip)

        if k_as is not None:
            # For agent state cross-attention
            k_as, v_as = map(split_heads, (k_as, v_as))
            out_as = self._efficient_attention(
                q, k_as, v_as, use_torch=self._torch_cross_attention_enabled)
            out_as = merge_heads(out_as)
        if k_aa is not None:
            # For agent action cross-attention
            k_aa, v_aa = map(split_heads, (k_aa, v_aa))
            if use_bmhk:
                attn_mask_aa = attn_mask_aa.reshape(
                    b, self.heads, q.shape[1], k_aa.shape[1])

            out_aa = self._efficient_attention(
                q, k_aa, v_aa, attn_bias=attn_mask_aa,
                use_torch=self._torch_cross_attention_enabled)

            out_aa = merge_heads(out_aa)
        if exists(mask):
            raise NotImplementedError

        if (self._inference_fused_mix and not self.training and
                not torch.is_grad_enabled() and out is not None and
                not self.cross_attention_scale_learnable):
            from unifolm_wma.utils.triton_attention_mix import fused_attention_mix
            out = fused_attention_mix(
                out, out_ip, out_as, out_aa,
                (self.image_cross_attention_scale,
                 self.agent_state_cross_attention_scale,
                 self.agent_action_cross_attention_scale))
        else:
            out = 0.0 if out is None else out
            out_ip = 0.0 if out_ip is None else out_ip
            out_as = 0.0 if out_as is None else out_as
            out_aa = 0.0 if out_aa is None else out_aa

            if self.cross_attention_scale_learnable:
                out = out + \
                    self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha_ctx) + 1) + \
                    self.agent_state_cross_attention_scale * out_as * (torch.tanh(self.alpha_cas) + 1) + \
                    self.agent_action_cross_attention_scale * out_aa * (torch.tanh(self.alpha_caa) + 1)
            else:
                out = out + \
                    self.image_cross_attention_scale * out_ip + \
                    self.agent_state_cross_attention_scale * out_as + \
                    self.agent_action_cross_attention_scale * out_aa

        if self._inference_projection_dtype == torch.float32:
            return self.to_out(out)
        return self._apply_projection(self.to_out[0], out).to(input_dtype)

    def _efficient_attention(self, q, k, v, attn_bias=None, use_torch=False):
        use_torch = (use_torch or
                     (self._torch_small_k_enabled and attn_bias is None and
                      q.shape[1] >= 2048 and k.shape[1] <= 16))
        if not use_torch:
            attention_dtype = self._inference_attention_dtype
            if (self._inference_attention_self_only and
                    q.shape[1] != k.shape[1]):
                attention_dtype = q.dtype
            if attention_dtype != q.dtype:
                q_low = q.to(attention_dtype)
                k_low = k.to(attention_dtype)
                v_low = v.to(attention_dtype)
                if isinstance(attn_bias, torch.Tensor):
                    attn_bias = attn_bias.to(attention_dtype)
                return xformers.ops.memory_efficient_attention(
                    q_low, k_low, v_low, attn_bias=attn_bias,
                    op=None).to(q.dtype)
            return xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=attn_bias, op=None)
        scores = torch.bmm(q, k.transpose(1, 2))
        scores.mul_(self.scale)
        if attn_bias is not None:
            scores.add_(attn_bias)
        probabilities = scores.softmax(dim=-1)
        return torch.bmm(probabilities, v)

    def _project_self_qkv(self, value):
        """Combine three fixed inference projections into one larger GEMM."""
        weights = (self.to_q.weight, self.to_k.weight, self.to_v.weight)
        key = tuple((weight.data_ptr(), weight._version, tuple(weight.shape),
                     weight.dtype, weight.device) for weight in weights)
        cached = self._fused_self_qkv_cache
        if cached is None or cached[0] != key:
            cached = (key, torch.cat(weights, dim=0).contiguous())
            self._fused_self_qkv_cache = cached
        return F.linear(value, cached[1])

    def _apply_projection(self, projection, value):
        """Apply a fixed inference Linear in the configured tensor-core dtype."""
        target_dtype = self._inference_projection_dtype
        if target_dtype == torch.float32:
            return projection(value)
        weight = projection.weight
        bias = projection.bias
        key = (weight.data_ptr(), weight._version, tuple(weight.shape),
               weight.device, target_dtype,
               None if bias is None else (bias.data_ptr(), bias._version))
        cached = self._lowp_projection_cache.get(id(projection))
        if cached is None or cached[0] != key:
            cached = (key, weight.to(target_dtype),
                      None if bias is None else bias.to(target_dtype))
            self._lowp_projection_cache[id(projection)] = cached
        if value.dtype != target_dtype:
            value = value.to(target_dtype)
        return F.linear(value, cached[1], cached[2])

    def _project_condition(self, name, projection, value):
        """Reuse an exact fixed-condition projection during inference.

        A strong reference to the source view prevents allocator address reuse.
        On a new generation, refresh the original output storage in place so
        an already captured CUDA Graph continues to read the correct buffers.
        """
        if (not self._condition_cache_enabled or self.training or
                torch.is_grad_enabled()):
            return self._apply_projection(projection, value)
        target_dtype = (torch.float32
                        if self._inference_attention_self_only else
                        self._inference_attention_dtype)
        key = (value.data_ptr(), value._version, tuple(value.shape),
               tuple(value.stride()), value.dtype, value.device,
               target_dtype)
        cached = self._condition_projection_cache.get(name)
        if cached is not None and cached["key"] == key:
            return cached["output"]
        current = self._apply_projection(projection, value)
        if target_dtype != current.dtype:
            current = current.to(target_dtype)
        if (cached is not None and cached["output"].shape == current.shape and
                cached["output"].dtype == current.dtype and
                cached["output"].device == current.device):
            cached["output"].copy_(current)
            output = cached["output"]
        else:
            output = current
        self._condition_projection_cache[name] = {
            "key": key, "source": value, "output": output,
        }
        return output

    def prepare_condition_cache(self, context):
        """Refresh fixed cross-attention projections without running queries."""
        if not self.image_cross_attention:
            return
        length = context.shape[1]
        if length == self.text_context_len + self.video_length:
            context_image = context[:, self.text_context_len:, :]
            self._project_condition("k", self.to_k, context)
            self._project_condition("v", self.to_v, context)
            self._project_condition("k_ip", self.to_k_ip, context_image)
            self._project_condition("v_ip", self.to_v_ip, context_image)
        elif length == (self.agent_state_context_len + self.text_context_len +
                        self.video_length):
            state_end = self.agent_state_context_len
            text_end = state_end + self.text_context_len
            context_agent_state = context[:, :state_end, :]
            context_ins = context[:, state_end:text_end, :]
            context_image = context[:, text_end:, :]
            self._project_condition("k", self.to_k, context_ins)
            self._project_condition("v", self.to_v, context_ins)
            self._project_condition("k_ip", self.to_k_ip, context_image)
            self._project_condition("v_ip", self.to_v_ip, context_image)
            self._project_condition("k_as", self.to_k_as, context_agent_state)
            self._project_condition("v_as", self.to_v_as, context_agent_state)
        else:
            state_end = self.agent_state_context_len
            action_end = state_end + self.agent_action_context_len
            text_end = action_end + self.text_context_len
            context_agent_state = context[:, :state_end, :]
            context_agent_action = context[:, state_end:action_end, :]
            context_ins = context[:, action_end:text_end, :]
            context_image = context[:, text_end:, :]
            self._project_condition("k", self.to_k, context_ins)
            self._project_condition("v", self.to_v, context_ins)
            self._project_condition("k_ip", self.to_k_ip, context_image)
            self._project_condition("v_ip", self.to_v_ip, context_image)
            self._project_condition("k_as", self.to_k_as, context_agent_state)
            self._project_condition("v_as", self.to_v_as, context_agent_state)
            self._project_condition("k_aa", self.to_k_aa, context_agent_action)
            self._project_condition("v_aa", self.to_v_aa, context_agent_action)

    def _get_cached_attn_mask_aa(self, b, l1, l2, *, device, dtype,
                                 block_size=16):
        key = (b, l1, l2, self.heads, block_size, device, dtype)
        if key not in self._action_mask_cache:
            # Preserve the original mask values and batch/head order. CUTLASS
            # accepts a zero query stride, so storage is O(batch*heads*keys).
            mask = self._get_attn_mask_aa(b, 1, l2, block_size)
            mask = mask.to(device=device, dtype=dtype)
            mask = mask.repeat_interleave(self.heads, dim=0)
            self._action_mask_cache[key] = mask.expand(-1, l1, -1)
        return self._action_mask_cache[key]

    def _get_attn_mask_aa(self, b, l1, l2, block_size=16):
        num_token = l2 // block_size
        start_positions = ((torch.arange(b) % block_size) + 1) * num_token
        col_indices = torch.arange(l2)
        mask_2d = col_indices.unsqueeze(0) >= start_positions.unsqueeze(1)
        mask = mask_2d.unsqueeze(1).expand(b, l1, l2)
        attn_mask = torch.zeros_like(mask, dtype=torch.float)
        attn_mask[mask] = float('-inf')
        return attn_mask


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False,
                 attention_cls=None,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 image_cross_attention_scale=1.0,
                 cross_attention_scale_learnable=False,
                 text_context_len=77):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            video_length=video_length,
            agent_state_context_len=agent_state_context_len,
            agent_action_context_len=agent_action_context_len,
            image_cross_attention=image_cross_attention,
            image_cross_attention_scale=image_cross_attention_scale,
            cross_attention_scale_learnable=cross_attention_scale_learnable,
            text_context_len=text_context_len)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self._inference_fused_residual_norm = False

    def forward(self, x, context=None, mask=None, **kwargs):
        # implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  # should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x, ), self.parameters(),
                              self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        if self._inference_fused_residual_norm and not self.training:
            from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm
            attended = self.attn1(self.norm1(x),
                context=context if self.disable_self_attn else None, mask=mask)
            x, normalized = fused_residual_layer_norm(attended, x, self.norm2)
            attended = self.attn2(normalized, context=context, mask=mask)
            x, normalized = fused_residual_layer_norm(attended, x, self.norm3)
            return self.ff(normalized) + x
        x = self.attn1(self.norm1(x),
                       context=context if self.disable_self_attn else None,
                       mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 disable_self_attn=False,
                 use_linear=False,
                 video_length=None,
                 agent_state_context_len=2,
                 agent_action_context_len=16,
                 image_cross_attention=False,
                 cross_attention_scale_learnable=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                agent_state_context_len=agent_state_context_len,
                agent_action_context_len=agent_action_context_len,
                image_cross_attention=image_cross_attention,
                cross_attention_scale_learnable=cross_attention_scale_learnable,
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self._channels_last_output = False

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if self._channels_last_output:
            x = x.contiguous(memory_format=torch.channels_last)
        else:
            x = x.contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 use_linear=False,
                 only_self_att=True,
                 causal_attention=False,
                 causal_block_size=1,
                 relative_position=False,
                 temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert (temporal_length is not None)
            attention_cls = partial(CrossAttention,
                                    relative_position=True,
                                    temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention,
                                    temporal_length=temporal_length)
        if self.causal_attention:
            assert (temporal_length is not None)
            self.mask = torch.tril(
                torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim,
                                  attention_cls=attention_cls,
                                  checkpoint=use_checkpoint)
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv1d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear
        self._channels_last_output = False
        self._inference_direct_layout = False

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if self.use_linear and self._inference_direct_layout and not self.training:
            # The intermediate contiguous BHW,C,T tensor is immediately
            # transposed again. Copy directly to the projection's BHW,T,C.
            x = rearrange(x, 'b c t h w -> (b h w) t c').contiguous()
            x = self.proj_in(x)
        else:
            x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
            if not self.use_linear:
                x = self.proj_in(x)
            x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
            if self.use_linear:
                x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # Slice the from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b * h * w)
        else:
            mask = None

        if self.only_self_att:
            # NOTE: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con',
                                t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # Calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(context[j],
                                       't l con -> (t r) l con',
                                       r=(h * w) // t,
                                       t=t).contiguous()
                    # Note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w)
            if self._channels_last_output:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            else:
                x = x.contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h,
                          w=w)
            if self._channels_last_output:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            else:
                x = x.contiguous()

        return x + x_in


class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self._inference_fused_geglu = False

    def forward(self, x):
        return self.activate(self.proj(x))

    def activate(self, projected):
        if self._inference_fused_geglu and not self.training:
            from unifolm_wma.utils.triton_geglu import fused_geglu
            return fused_geglu(projected)
        x, gate = projected.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))
        self._inference_dtype = torch.float32
        self._lowp_weight_cache = {}

    def forward(self, x):
        if self._inference_dtype == torch.float32 or self.training:
            return self.net(x)
        input_dtype = x.dtype
        x = x.to(self._inference_dtype)
        project_in = self.net[0]
        if isinstance(project_in, GEGLU):
            projected = self._linear(project_in.proj, x)
            x = project_in.activate(projected)
        else:
            x = F.gelu(self._linear(project_in[0], x))
        x = self.net[1](x)
        return self._linear(self.net[2], x).to(input_dtype)

    def _linear(self, projection, value):
        weight = projection.weight
        bias = projection.bias
        key = (weight.data_ptr(), weight._version, tuple(weight.shape),
               weight.device, self._inference_dtype,
               None if bias is None else (bias.data_ptr(), bias._version))
        cached = self._lowp_weight_cache.get(id(projection))
        if cached is None or cached[0] != key:
            cached = (key, weight.to(self._inference_dtype),
                      None if bias is None else bias.to(self._inference_dtype))
            self._lowp_weight_cache[id(projection)] = cached
        return F.linear(value, cached[1], cached[2])

class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv,
                            'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads=self.heads,
                            qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out,
                        'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads,
                        h=h,
                        w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_
