"""Check role factories, then reuse the common kernel/cache/graph validation."""
import torch
from unifolm_wma.utils.initial_decision_linears import install_initial_decision_linears
from unifolm_wma.utils.initial_world_linears import install_initial_world_linears
from unifolm_wma.utils.decision_initial_primary import install_decision_initial_primary
from unifolm_wma.utils.decision_initial_projection_roles import install_decision_initial_projection_roles
from check_decision_initial_primary import main as check_common_forward


def main():
    for role, ffn_primary in [('qk',False),('vo',False),('qk',True),('vo',True)]:
        root = torch.nn.Module()
        block = torch.nn.Module()
        root.init_attn = torch.nn.ModuleList([block])
        for label in ['attn1','attn2']:
            attention = torch.nn.Module()
            setattr(block,label,attention)
            for projection in ['to_q','to_k','to_v']:
                setattr(attention,projection,torch.nn.Linear(512,512,bias=False))
            attention.to_out = torch.nn.Sequential(torch.nn.Linear(512,512))
        block.ffn = torch.nn.ModuleList([torch.nn.Linear(512,4096),torch.nn.Linear(2048,512)])
        root.unrelated = torch.nn.Linear(512,512)
        assert install_initial_decision_linears(root) == 10
        assert install_initial_world_linears(root) == 10
        if ffn_primary:
            assert install_decision_initial_primary(root,'ffn') == 2
        assert install_decision_initial_projection_roles(root,role) == 4
        for name,module in root.named_modules():
            if not isinstance(module,torch.nn.Linear):
                continue
            expected = name.startswith('init_attn.') and (name.endswith(('.to_q','.to_k'))
                if role == 'qk' else name.endswith(('.to_v','.to_out.0')))
            assert (getattr(module,'_decision_initial_primary_role',None) == role) == expected
            expected_ffn = ffn_primary and name.startswith('init_attn.') and '.ffn.' in name
            assert bool(getattr(module,'_decision_initial_primary_installed',False)) == expected_ffn
        try:
            install_decision_initial_projection_roles(root,role)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate initial role installation accepted')
    check_common_forward()
    print('INITIAL_PROJECTION_ROLES_CHECK passed=true qk_modules=4 vo_modules=4 '
          'other_roles_ffn_excluded=true ffn_composition_membership=true duplicate_rejected=true '
          'common_kernel_cache_graph_guards=passed full_video_quality=unverified',flush=True)


if __name__ == '__main__':
    main()
