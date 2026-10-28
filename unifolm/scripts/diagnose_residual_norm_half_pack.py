"""Locate compiler-induced bit differences; no model installation or scoring."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import torch
import triton
from unifolm_wma.utils.triton_residual_layer_norm import fused_residual_layer_norm,_residual_layer_norm
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.residual_norm_half_pack import residual_norm_half_pack
from unifolm_wma.utils.triton_residual_norm_half_pack_kernel import _residual_norm_half_pack


def difference(a,b):
    dtype=torch.int32 if a.dtype==torch.float32 else torch.int16
    aa=a.contiguous().view(dtype);bb=b.contiguous().view(dtype)
    mask=aa!=bb
    indices=torch.nonzero(mask.flatten())[:4].flatten()
    return dict(dtype=str(a.dtype),unequal_bits=int(mask.sum()),max_abs=float((a.float()-b.float()).abs().max()),
                first_indices=indices.tolist(),first_bits_a=aa.flatten()[indices].tolist(),first_bits_b=bb.flatten()[indices].tolist())


@torch.no_grad()
def main():
    torch.manual_seed(725);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(UNIFOLM_NORM_BUFFER_NOFILL='all',UNIFOLM_HALF_PACK_NOFILL='1')
    output=Path.home()/'benchmark-runs/diagnostics'/('residual-norm-half-pack-numeric-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));output.mkdir()
    print('DIAGNOSTIC_DIR='+str(output),flush=True)
    records=[]
    for index,(rows,channels) in enumerate([(40960,320),(10240,640),(2560,1280),(640,1280),(73,321),(17,96),(1,320),(2,640)]):
        norm=torch.nn.LayerNorm(channels,eps=1e-5 if index%2 else 1e-6).cuda().eval()
        norm.weight.uniform_(.7,1.3);norm.bias.uniform_(-.2,.2)
        left=torch.randn(rows,channels,device='cuda')*.7;right=torch.randn_like(left)*.3
        r,n=fused_residual_layer_norm(left,right,norm);parts=pack_half_weight(n,weight_scale=1.)
        reference=(r,*parts,n)
        actual=residual_norm_half_pack(left,right,norm,debug=True)
        fast=residual_norm_half_pack(left,right,norm)
        diagnostic=[difference(a,b) for a,b in zip(reference,actual)]
        fast_diagnostic=[difference(a,b) for a,b in zip(reference[:3],fast)]
        kernels={}
        original=_residual_layer_norm[(rows,)](left,right,norm.weight,norm.bias,r,n,channels,norm.eps,triton.next_power_of_2(channels),num_warps=4,enable_fp_fusion=False)
        for debug in [False,True]:
            fused=_residual_norm_half_pack[(rows,)](left,right,norm.weight,norm.bias,actual[0],actual[3],actual[1],actual[2],channels,norm.eps,triton.next_power_of_2(channels),debug,num_warps=4,enable_fp_fusion=False)
            kernels['fused_'+str(debug)]=fused
        kernels['original']=original
        for label,kernel in kernels.items():
            for name in ['ttgir','llir','ptx']:
                (output/(str(index)+'-'+label+'.'+name)).write_text(kernel.asm[name])
        row=dict(index=index,shape=[rows,channels],debug_differences=diagnostic,fast_differences=fast_diagnostic,
                 resources={label:dict(registers=k.n_regs,shared=k.metadata['shared']) for label,k in kernels.items()})
        records.append(row);print('RESIDUAL_NORM_HALF_PACK_DIAG '+json.dumps(row),flush=True)
        del left,right,norm,r,n,parts,reference,actual,fast
        torch.cuda.empty_cache()
    (output/'numeric-differences.json').write_text(json.dumps(records,indent=2)+'\n')
    print('RESIDUAL_NORM_HALF_PACK_DIAG_CHECK passed=true fixtures=8 no_precision_claim=true',flush=True)


if __name__=='__main__':main()
