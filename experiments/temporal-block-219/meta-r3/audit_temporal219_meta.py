"""Instantiate the configured backbone on meta; no checkpoint, CUDA or data."""
import json,os
from pathlib import Path
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf
from unifolm_wma.modules.networks.wma_model import WMAModel
from unifolm_wma.modules.attention import TemporalTransformer


def main():
    assert not torch.cuda.is_initialized()
    base=Path(os.environ['CANONICAL_ROOT'])
    config=OmegaConf.load(base/'configs/inference/world_model_interaction.yaml')
    parameters=OmegaConf.to_container(config.model.params.wma_config.params,resolve=True)
    assert config.model.params.wma_config.target.endswith('.WMAModel')
    with torch.device('meta'):model=WMAModel(**parameters)
    rows=[]
    for name,module in model.named_modules():
        if type(module) is not TemporalTransformer:continue
        assert module.only_self_att and not module.causal_attention
        rows.append(dict(name=name,input_channels=module.in_channels,
            use_linear=module.use_linear,proj_in_class=type(module.proj_in).__name__,
            proj_out_class=type(module.proj_out).__name__,proj_in_weight=list(module.proj_in.weight.shape),
            proj_out_weight=list(module.proj_out.weight.shape),norm_groups=module.norm.num_groups,
            norm_channels=module.norm.num_channels,transformer_blocks=len(module.transformer_blocks),
            block_channels=module.transformer_blocks[0].attn1.to_q.in_features,
            FFN_in=list(module.transformer_blocks[0].ff.net[0].proj.weight.shape),
            FFN_out=list(module.transformer_blocks[0].ff.net[2].weight.shape),
            temporal_length=module.transformer_blocks[0].attn1.temporal_length))
    assert len(rows)==17 and sum(r['use_linear'] for r in rows)==16
    initial=next(r for r in rows if r['name'].startswith('init_attn.'))
    assert initial['proj_in_class']==initial['proj_out_class']=='Conv1d'
    assert initial['input_channels']==320 and initial['block_channels']==512
    assert not torch.cuda.is_initialized()
    out=dict(passed=True,source='actual configured WMAModel instantiated on meta device',
        checkpoint_loaded=False,input_data_loaded=False,no_GPU_initialized=True,
        temporal_modules=rows,regular_linear_blocks=16,initial_Conv1d_projection_blocks=1,
        global5D_GroupNorm_must_remain_intact=True,
        dynamic_strides_JIT_launches_and_performance_not_measured=True)
    Path(os.environ['META_REPORT_PATH']).write_text(json.dumps(out,indent=2)+'\n')
    print('TEMPORAL219_META '+json.dumps(dict(passed=True,modules=17,regular_linear=16,initial_Conv1d=1,no_GPU_initialized=True)))
if __name__=='__main__':main()
