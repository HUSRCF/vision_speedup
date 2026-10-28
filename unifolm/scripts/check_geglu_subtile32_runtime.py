"""Verify real host selection, original caches, and explicit source fallbacks."""
import json
import os
import torch
from check_world_geglu_hidden_parts import main as c640_guards
from check_world_ffn320_zero import main as c320_guards
from geglu_subtile32_production_hooks import installed,MODE,GEOMETRIES,c320,c640
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight
from unifolm_wma.utils.triton_zero_weight_low_geglu import residual_count


@torch.no_grad()
def main():
    os.environ[MODE]='1'
    with installed() as launches:
        c640_guards();c320_guards()
        assert launches and {(r['M'],r['C'],r['K']) for r in launches}==GEOMETRIES
        assert all(r['subtile32'] for r in launches)
        assert all(r['group']==(16 if r['K']==320 else int(os.environ.get('UNIFOLM_WORLD_GEGLU_C640_GROUP','32'))) for r in launches)
        eligible=len(launches)
        for width,fn in [(320,c320.geglu_hidden_parts_c320),(640,c640.geglu_hidden_parts)]:
            xp=pack_half_weight(torch.randn(13,width,device='cuda')*.1,weight_scale=1.)
            wp=pack_half_weight(torch.randn(width*8,width,device='cuda')*.01)
            count=residual_count(wp,refresh=True)
            reference=None
            for mode in ['0','1']:
                os.environ[MODE]=mode
                output=fn(xp,wp,count)
                assert not launches[-1]['subtile32']
                if reference is None:reference=output
                else:assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(reference,output))
            for backend in [torch.backends.cudnn,torch.backends.cuda.matmul]:
                old=backend.allow_tf32
                try:
                    backend.allow_tf32=True;output=fn(xp,wp,count)
                    assert not launches[-1]['subtile32']
                    assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(reference,output))
                finally:backend.allow_tf32=old
            os.environ[MODE]='invalid';before=len(launches)
            try:fn(xp,wp,count)
            except ValueError as exc:assert str(exc)=='Invalid world GEGLU K32 mode'
            else:raise AssertionError('invalid mode accepted')
            assert len(launches)==before
            os.environ[MODE]='1'
    print('GEGLU_SUBTILE32_RUNTIME_CHECK '+json.dumps(dict(passed=True,real_block_fixtures=4,
        eligible_launches=eligible,actual_both_widths_selected=True,original_cache_and_fallback_guards=True,
        original_FP32_masters=True,production_host_controls_dispatch=True,tail_mode_and_backend_fallbacks=True,
        invalid_flag_rejected=True)),flush=True)
    print('GEGLU_SUBTILE32_MODEL_GUARD_CHECK passed=true fixtures=4',flush=True)


if __name__=='__main__':main()
