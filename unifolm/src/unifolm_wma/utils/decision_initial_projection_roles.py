"""Narrow the initial attention experiment by projection role."""
import torch
from .decision_initial_primary import make_forward


def install_decision_initial_projection_roles(diffusion, role):
    if role not in {'qk', 'vo'}:
        raise ValueError('Initial projection role must be qk or vo')
    count = 0
    for name, module in diffusion.named_modules():
        selected = (name.endswith(('.to_q','.to_k')) if role == 'qk' else
                    name.endswith(('.to_v','.to_out.0')))
        if (not name.startswith('init_attn.') or not selected
                or not isinstance(module, torch.nn.Linear)
                or (module.in_features,module.out_features) != (512,512)):
            continue
        if (not getattr(module,'_initial_decision_linear_installed',False)
                or not getattr(module,'_initial_world_linear_installed',False)
                or getattr(module,'_decision_initial_primary_installed',False)
                or getattr(module,'_decision_initial_primary_role',None) is not None):
            raise ValueError('Initial role experiment requires unmodified compensated wrappers')
        module.forward = make_forward(module)
        module._decision_initial_primary_role = role
        count += 1
    return count
