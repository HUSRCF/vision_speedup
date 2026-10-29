"""Measure selected channel-split native convolutions on the adopted223 model."""
from contextlib import contextmanager
import json
from pathlib import Path
import torch
from unifolm_wma.utils import decision_temporal223_mode as adopted
from unifolm_wma.utils.nccl_geglu_209_mode import measure
from .world_co_split225 import parameters,gather_channels
import torch.distributed as dist

active=None
enabled=False


class Controller:
    def __init__(self,model):
        from unifolm_wma.utils.native_graph_sgemm import NativeSgemmModel
        self.base=adopted.install(model);self.rank=dist.get_rank()%2;self.saved=[];self.rows=[];self.owners=[];self.replacements=[];self.native_reports=[]
        scope={'baseline_source_count': 223, 'selected_owner_count': 9, 'selected': [{'local_weight': 'input_blocks.7.0.in_layers.2.weight', 'geometry': [1280, 192, 640], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.09420798222223917}, {'local_weight': 'output_blocks.5.0.in_layers.2.weight', 'geometry': [1280, 192, 1920], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.5773652791976928}, {'local_weight': 'output_blocks.5.3.conv.weight', 'geometry': [1280, 640, 1280], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.6400000254313151}, {'local_weight': 'output_blocks.7.0.in_layers.2.weight', 'geometry': [640, 640, 1280], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.5367467403411865}, {'local_weight': 'output_blocks.8.0.in_layers.2.weight', 'geometry': [640, 640, 960], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.3353599707285564}, {'local_weight': 'output_blocks.8.3.conv.weight', 'geometry': [640, 2560, 640], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.7307947476704917}, {'local_weight': 'output_blocks.9.0.in_layers.2.weight', 'geometry': [320, 2560, 960], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.3543039162953693}, {'local_weight': 'output_blocks.10.0.in_layers.2.weight', 'geometry': [320, 2560, 640], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.32785073916117347}, {'local_weight': 'output_blocks.11.0.in_layers.2.weight', 'geometry': [320, 2560, 640], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': 0.32785073916117347}], 'excluded': [{'local_weight': 'input_blocks.1.0.in_layers.2.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.1.0.out_layers.3.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.2.0.in_layers.2.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.2.0.out_layers.3.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.4.0.in_layers.2.weight', 'geometry': [640, 640, 320], 'config': [32, 64, 32, 4, 1], 'minimum_geometry_net_saved_ms': -0.05768533547719318, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.4.0.out_layers.3.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.5.0.in_layers.2.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'input_blocks.5.0.out_layers.3.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.6.0.out_layers.3.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.7.0.out_layers.3.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.8.0.out_layers.3.weight', 'geometry': [640, 640, 640], 'config': [64, 64, 64, 4, 1], 'minimum_geometry_net_saved_ms': 0.01945598920186359, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.9.0.out_layers.3.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.10.0.out_layers.3.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}, {'local_weight': 'output_blocks.11.0.out_layers.3.weight', 'geometry': [320, 2560, 320], 'config': [32, 128, 64, 4, 1], 'minimum_geometry_net_saved_ms': -0.21725869178771973, 'reason': 'negative_or_less_than_50us_minimum_whole_conv_net_saving'}], 'minimum_net_gain_threshold_ms': 0.05, 'selection_uses_both_ranks_and_layouts_and_all6_pairs': True, 'sum_minimum_synthetic_net_saved_ms': 3.924480140209198, 'synthetic_budget_not_full_model_result': True};assert scope['selected_owner_count']==9
        self.expected={r['local_weight']:(tuple(r['geometry']),r['config'])for r in scope['selected']}
        modules={n+'.weight':m for n,m in model.named_modules()if isinstance(m,torch.nn.Conv2d)}
        assert set(self.expected)<=set(modules)
        self.weights={id(modules[n].weight):n for n in self.expected}
        for name,(geometry,config)in self.expected.items():
            conv=modules[name];assert tuple(conv.weight.shape)==(geometry[0],geometry[2],3,3)
            before=conv.forward
            def forward(value,*args,_before=before,_name=name,**kwargs):
                result=_before(value,*args,**kwargs)
                if not enabled or torch.backends.cudnn.allow_tf32:return result
                assert not torch.is_grad_enabled()and result.dtype==torch.float32 and result.shape[0]==16
                co,cols,ci=self.expected[_name][0];assert result.shape[1]==co
                full=gather_channels(result,self.rank)
                if torch.cuda.is_current_stream_capturing():self.rows.append(dict(weight=_name,geometry=[co,cols,ci],input_stride=list(value.stride()),output_stride=list(result.stride()),collectives=1,local_MiB=result.numel()*result.element_size()/2**21))
                return full
            conv.forward=forward;self.saved.append((conv,before))
        self.owner_class=NativeSgemmModel;self.old_init=NativeSgemmModel.__init__
        def initialize(owner,graph,model):
            self.old_init(owner,graph,model)
            if not enabled:return
            assert owner.report['selected']==len(owner.swaps)==23 and owner.report['numerical_selected']==15 and owner.report['kahan_selected']==8
            assert len(owner._world_head_filter_cache.caches)==45
            self.owners.append(owner);records=[]
            for swap,plan in zip(owner.swaps,owner.report['plans']):
                name=self.weights.get(id(swap.cache.weight))
                if name is None:continue
                geometry,config=self.expected[name];assert tuple(plan['geometry'])==geometry
                original=swap.replacement;replacement=parameters(swap,geometry,config,self.rank)
                assert replacement.params.func==original.params.func and replacement.params.shared==original.params.shared
                assert replacement.params.grid_x*2==original.params.grid_x
                self.replacements.append((swap,original,replacement));swap.replacement=replacement;swap.set_enabled(True)
                records.append(dict(weight=name,geometry=list(geometry),family='packed'if hasattr(swap,'wh')else'kahan',original_CUBIN_and_kernel_function=True,original_K_constexpr_batch_strides=True,grid_rows_halved=True,original_owner_close_params_preserved=True))
            assert {r['weight']for r in records}==set(self.expected)and len(records)==9
            self.native_reports.append(dict(selected=9,original_owners=23,original_head_caches=45,original_strict_CUBIN_and_cache_checks_completed_first=True,rows=records))
        self.initialize=initialize;NativeSgemmModel.__init__=initialize

    def report(self):
        return dict(captured=len(self.rows),rows=self.rows.copy(),selected_convolutions=len(self.expected),native_cache_coverage=self.native_reports.copy(),adopted223=self.base.report(),
            original_JITs_constexpr_and_K_order=True,new_kernels=0,full5D_GroupNorm_and16T_preserved=True,extra_world_collectives_per_forward=9,
            pack_gather_and_layout_restoration_in_timing=True,full_model_and_video_quality_unverified=True)



class ProductionController(Controller):
    @property
    def captures(self):return self.base.captures
    def report(self):
        report=super().report()
        return dict(report['adopted223'],world_CO={k:v for k,v in report.items()if k!='adopted223'})


def install(model):
    global active
    assert active is None;active=ProductionController(model);return active


def uninstall():
    global active,enabled
    if active is not None:
        for owner in active.owners:
            if not owner.closed:owner.close()
        for swap,old,new in active.replacements:
            assert swap.closed and swap.replacement is new;swap.replacement=old
        for module,forward in reversed(active.saved):module.forward=forward
        assert active.owner_class.__init__ is active.initialize;active.owner_class.__init__=active.old_init
        adopted.uninstall()
    active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:
        with adopted.mode(True):yield
    finally:enabled=previous
