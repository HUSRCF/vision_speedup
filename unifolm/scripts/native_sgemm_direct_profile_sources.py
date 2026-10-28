"""Replace only template construction with capture-witnessed direct parameters."""
import ast
from pathlib import Path


def profile():
    t=Path(__file__).with_name('profile_native_sgemm_pair.py').read_text()
    def one(old,new):
        nonlocal t
        assert t.count(old)==1,old
        t=t.replace(old,new)
    one('from native_winograd_sgemm_swap import NativeSgemmSwap',
        'from native_sgemm_direct_params import DirectSgemmSwap')
    start=t.index('                        dw=torch.zeros(')
    end=t.index('                    swaps.append(swap)',start)
    t=t[:start]+'''                        dummy=torch.empty((1,),device=w.device)
                        kernel=transformed_gemm_fp16.warmup(dummy,dummy,dummy,co,m,ci,bm,bn,bk,
                            num_warps=warps,num_stages=stages,enable_fp_fusion=False,
                            grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36))
                        kernel._init_handles()
                        templates[geometry]=(None,dummy,kernel)
                    kernel=templates[geometry][-1]
                    swap=DirectSgemmSwap(native,consumer,kernel,geometry,configs[geometry],extension,api)
'''+t[end:]
    one('template_bytes=sum(t.numel()*4 for row in templates.values() for t in row[1:4]),',
        'direct_params=True,template_bytes=sum(row[1].numel()*4 for row in templates.values()),')
    one('                for cg,*_ in templates.values():cg.reset()',
        '                for cg,*_ in templates.values():\n                    if cg is not None:cg.reset()')
    ast.parse(t);return t
