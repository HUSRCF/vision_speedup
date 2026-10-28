"""Compensated initial linears only in the existing decision-cudnn stage."""
import torch
from .triton_compensated_linear import make_forward


def install_initial_decision_linears(diffusion):
    count = 0
    for name, module in diffusion.named_modules():
        if (not name.startswith('init_attn.') or not isinstance(module, torch.nn.Linear)
                or (module.in_features, module.out_features) not in
                {(512, 4096), (2048, 512), (512, 512)}):
            continue
        if (getattr(module, '_initial_decision_linear_installed', False)
                or getattr(module, '_tf32x3_installed', False)):
            raise ValueError('Initial Linear already wrapped')
        original = module.forward
        compensated = make_forward(module, min_matmul_macs=0,
            backend='scaled-fp16', cache_half_weights=True, group_m=32,
            block_k=32, pack_inputs=False)

        def forward(value, module=module, original=original, compensated=compensated):
            if (not torch.backends.cudnn.allow_tf32
                    or torch.backends.cuda.matmul.allow_tf32
                    or value.shape[-1] != module.in_features
                    or value.numel() // module.in_features != 40960):
                return original(value)
            return compensated(value)

        module.forward = forward
        module._initial_decision_linear_installed = True
        count += 1
    return count
