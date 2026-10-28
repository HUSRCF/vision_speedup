"""Keep the full old real-QKV guard and independently check schedule selection."""
import json
import os
import torch
from check_world_zero_weight_low_qkv import main as previous_guard
from unifolm_wma.utils import triton_zero_weight_low_linear as helper
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight

def exact(a,b):assert a.shape==b.shape and torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))

@torch.no_grad()
def main():
    os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']='3-large'
    original=helper._zero_low_gemm;launches=[]
    positive={(r['geometry']['M'],r['geometry']['N'],r['geometry']['K']) for r in json.load(open('zero-low-geometry.json'))['world'] if r['geometry']['M']>=2560}
    class Observed:
        def __getitem__(self,grid):
            launch=original[grid]
            def run(*args,**kwargs):
                p=dict(zip(original.arg_names,args));p.update(kwargs)
                wanted=(3 if os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']=='3-large'
                    and (p['M'],p['N'],p['K']) in positive and p['OUTPUT_GROUPS']==3
                    and not p['HAS_BIAS'] and not torch.backends.cudnn.allow_tf32
                    and not torch.backends.cuda.matmul.allow_tf32 else 2)
                assert kwargs['num_stages']==wanted
                launches.append(dict(M=p['M'],N=p['N'],K=p['K'],depth=wanted))
                return launch(*args,**kwargs)
            return run
    helper._zero_low_gemm=Observed()
    try:
        previous_guard()
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        x=torch.randn(2560,1280,device='cuda')*.2
        w=(torch.randn(3840,1280,device='cuda')*.02).half().float()
        parts=pack_half_weight(w);count=helper.weight_low_count(parts)
        bias=torch.randn(3840,device='cuda')*.02
        cases=['selected','explicit2','decision','matmul_tf32','bias','groups1','other_shape']
        for case in cases:
            value=x[:2].contiguous() if case=='other_shape' else x
            b=bias if case=='bias' else None;groups=1 if case=='groups1' else 3
            torch.backends.cudnn.allow_tf32=case=='decision';torch.backends.cuda.matmul.allow_tf32=case=='matmul_tf32'
            os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']='2'
            expected=helper.zero_weight_low_linear(value,parts,count,b,output_groups=groups)
            os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']='2' if case=='explicit2' else '3-large'
            before=len(launches);actual=helper.zero_weight_low_linear(value,parts,count,b,output_groups=groups)
            exact(actual,expected);assert len(launches)==before+1
            assert launches[-1]['depth']==(3 if case=='selected' else 2)
        os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']='invalid'
        try:helper.zero_weight_low_linear(x[:2],parts,count,output_groups=3)
        except ValueError as error:assert 'schedule' in str(error)
        else:raise AssertionError('invalid policy accepted')
        assert any(r['depth']==3 for r in launches) and any(r['M']==640 and r['depth']==2 for r in launches)
    finally:
        helper._zero_low_gemm=original
        os.environ['UNIFOLM_WORLD_ZERO_QKV_STAGES']='3-large'
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    print('ZERO_QKV_SCHEDULE_RUNTIME_CHECK '+json.dumps(dict(passed=True,real_QKV_configs=4,
        previous_real_QKV_guard_retained=True,policy_cases=8,selected_large3_small2=True,
        explicit2_decision_matmul_bias_group_shape_fallbacks=True,invalid_policy_rejected=True,
        all_shader_launches_observed=True)),flush=True)

if __name__=='__main__':main()
