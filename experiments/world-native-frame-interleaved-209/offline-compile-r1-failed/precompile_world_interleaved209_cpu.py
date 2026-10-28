"""Compile the nine selected new SM80 kernels with no visible CUDA device.

Explicit target/signatures match the pinned Triton2.3.1 JIT source. No device
query, kernel load, launch or allocation is needed. Full GPU witnesses and
numerical gates remain mandatory; compiling is not an execution speed result.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch,triton
from triton.compiler import ASTSource,AttrsDescriptor
from world_native_frame_interleaved209_kernel import (
    transformed_gemm_fp16_packed_frame_interleaved,
    transformed_gemm_fp16_kahan_frame_interleaved)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    assert torch.__version__=='2.3.1+cu121' and triton.__version__=='2.3.1'
    assert 'TRITON_CACHE_MANAGER' not in os.environ
    scope=json.loads((args.root/'world-interleaved-owner-scope.json').read_text())
    selected={tuple(row['geometry']) for row in scope['rows'] if row['within_qualified_ResBlock']}
    assert len(selected)==9
    specs=json.loads((args.root/'matrix-geometries.json').read_text());rows=[]
    for spec in specs:
        geometry=tuple(spec['geometry'])
        if geometry not in selected:continue
        packed=spec['family']=='packed';np=4 if packed else 3
        function=transformed_gemm_fp16_packed_frame_interleaved if packed else transformed_gemm_fp16_kahan_frame_interleaved
        co,cols,k=geometry;bm,bn,bk,warps,stages=spec['config']
        constants=dict(enumerate([co,cols,k,bm,bn,bk,cols//2],start=np))
        assert len(function.params)==np+7 and all(not p.do_not_specialize for p in function.params)
        assert all(v%16==0 and v!=1 for v in constants.values())
        # Torch CUDA allocations and the original validated owner pointers
        # are aligned to16. All seven constexpr integers also divide by16;
        # only integer arguments enter the JIT's divisible_by_8 set.
        attrs=AttrsDescriptor(tuple(range(np+7)),(),(),tuple(range(np,np+7)))
        signature={i:('*fp16' if packed and i<2 else '*fp32') for i in range(np)}
        start=time.perf_counter()
        kernel=triton.compile(ASTSource(function,signature,constants,attrs),target=('cuda',80),
            options=dict(num_warps=warps,num_stages=stages,enable_fp_fusion=False,debug=function.debug))
        assert kernel.function is None and kernel.module is None and not torch.cuda.is_initialized()
        rows.append(dict(**spec,seconds=time.perf_counter()-start,CUBIN_SHA256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),
            shared_bytes=kernel.shared,signature=signature,constants=constants,
            target=['cuda',80],no_device_kernel_load_or_launch=True))
        Path('progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('WORLD_INTERLEAVED_OFFLINE_COMPILE '+json.dumps(rows[-1]),flush=True)
    assert len(rows)==9 and not torch.cuda.is_initialized()
    report=dict(complete=True,CPU_only=True,CUDA_VISIBLE_DEVICES='',Torch_CUDA_not_initialized=True,
        kernels=9,rows=rows,cache=os.environ['TRITON_CACHE_DIR'],
        original_default_cache_not_used_or_changed=True,
        original_Kahan_CUBIN_gates_not_modified=True,
        runtime_captured_pointer_ABI_and_full_model_numerics_still_required=True,
        no_GPU_speed_or_official_result=True)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('WORLD_INTERLEAVED_OFFLINE_COMPILE_CHECK passed=true CUDA_initialized=false',flush=True)


if __name__=='__main__':main()
