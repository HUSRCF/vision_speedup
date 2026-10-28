"""Balance only the INT8 residual; keep the original Half main product."""
import triton
import triton.language as tl
from fp16_grouped_residual209_split_candidate import (
    allocate_group_pack,allocate_pack,pack,pack_channels,launch,launch_group,
)


@triton.jit
def _pack_balanced(X,Channel,Q,S,R:tl.constexpr,K:tl.constexpr,
                   GK:tl.constexpr,P:tl.constexpr,INVERSE:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1)
    index=tl.arange(0,P);ks=group*GK+index
    mask=(ks<K)&(index<GK)
    value=tl.load(X+row*K+ks,mask,other=0).to(tl.float32)
    channel=tl.load(Channel+ks,mask,other=1.)
    if INVERSE:value=value/channel
    else:value=value*channel
    maximum=tl.max(tl.abs(value),0)
    scale=tl.where(maximum>0,maximum/127.,1.)
    quantized=tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;',
        constraints='=r,f',args=[value/scale],dtype=tl.int32,is_pure=True,pack=1)
    tl.store(Q+row*K+ks,quantized,mask)
    tl.store(S+row*tl.cdiv(K,GK)+group,scale)


def pack_balanced(value,out,gk,channel,*,inverse=False):
    rows,k=value.shape
    assert channel.shape==(k,) and channel.is_contiguous()
    _pack_balanced[(rows,triton.cdiv(k,gk))](value,channel,*out,rows,k,gk,
        triton.next_power_of_2(gk),inverse,num_warps=4,enable_fp_fusion=False)
