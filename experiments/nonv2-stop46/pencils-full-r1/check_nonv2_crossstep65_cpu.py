"""Verify module contracts and that reuse calls retain the parallel executor."""
import json,torch
from pathlib import Path
from types import MethodType,SimpleNamespace
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.crossstep_feature_runtime import Runtime
from unifolm_wma.utils.nccl_nonv2_all_blocks_runtime63 import expected_modules

class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.8));self.calls=0
    def forward(self,x):self.calls+=1;return torch.tanh(x*self.weight+.01)

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.middle_block=Block();self.output_blocks=torch.nn.ModuleList([Block()for _ in range(3)]);self.head=Block()
    def forward(self,x):
        for block in self.input_blocks:x=block(x)
        x=self.middle_block(x)
        for block in self.output_blocks:x=block(x)
        return self.head(x)

class Executor:
    def __init__(self,model):
        def original(owner,*args,**kwargs):return owner(args[0])
        self.denoiser_graph=DenoiserGraph(MethodType(original,model));self.calls=0
    def __call__(self,*args,**kwargs):
        self.calls+=1;assert '_crossstep_feature_reuse'in kwargs
        return self.denoiser_graph(*args,**kwargs)

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    proof=json.loads((Path(__file__).parent/'module-coverage.json').read_text())
    names=tuple(sorted(proof['full_module_names']));assert len(names)==33
    for row in proof['candidates'].values():
        result=expected_modules(names,tuple(row['selected']),row['scope'],True)
        assert result==tuple(sorted(row['retained']))and len(result)==row['count']
        assert expected_modules(names,tuple(row['selected']),row['scope'],False)==names
    try:expected_modules(names[:-1]+(names[0],),(),'deep',False)
    except AssertionError:pass
    else:raise AssertionError('Duplicate module name accepted')
    backbone=Backbone().eval();executor=Executor(backbone)
    outer=SimpleNamespace(model=SimpleNamespace(diffusion_model=backbone),apply_model=executor)
    runtime=Runtime(outer,input_start=1,output_end=1,start=20,stop=40,stages=('decision','world'))
    assert runtime.executor is executor and runtime.graph is executor.denoiser_graph and runtime.graph.max_graphs==4
    rng=torch.get_rng_state().clone();x=torch.ones(1,4)
    for generation in range(12):
        for stage in ('decision','world'):
            condition={'stage':stage,'generation':generation}
            for position in range(50):
                runtime.set_step(stage,position)
                outer.apply_model(x+position*.001,None,None,torch.tensor([49-position]),condition,fs=None)
                runtime.finish_step()
    report=runtime.report()
    assert executor.calls==1200
    assert report['counts']==dict(decision_calls=600,decision_refresh=480,decision_reuse=120,world_calls=600,world_refresh=480,world_reuse=120)
    assert len(report['completed_generations'])==24 and all(g['steps']==50 for g in report['completed_generations'])
    assert backbone.input_blocks[0].calls==backbone.output_blocks[-1].calls==backbone.head.calls==1200
    assert backbone.middle_block.calls==960
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    runtime.controller.close()
    print('NONV2_CROSSSTEP65_CPU passed=true all1200_calls_through_parallel_executor=true role_module_sets=true duplicate_rejected=true no_GPU=true full_video_quality_not_inferred=true')

if __name__=='__main__':main()

"""Two-refresh predictor: affine fidelity, real step gaps and stale rejection."""
import copy
import torch
from unifolm_wma.utils.crossstep_feature_reuse import Controller,predict

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5));self.position=0;self.calls=0
    def forward(self,x):
        self.calls+=1
        return x*self.weight+.125+self.position/64

class SpatialTransformer(Block):pass

class Model(torch.nn.Module):
    def __init__(self,spatial=False):
        super().__init__();kind=SpatialTransformer if spatial else Block
        self.input_blocks=torch.nn.ModuleList([kind()for _ in range(3)]);self.middle_block=kind();self.output_blocks=torch.nn.ModuleList([kind()for _ in range(3)]);self.head=Block();self.position=0
    def forward(self,x):
        for module in [*self.input_blocks,self.middle_block,*self.output_blocks,self.head]:module.position=self.position;x=module(x)
        return x

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    for scope,boundary in(('deep',False),('deep',True),('spatial',False)):
        original=Model(scope=='spatial').eval();model=copy.deepcopy(original)
        c=Controller(model,input_start=1,output_end=1,scope=scope,boundary_residual=boundary,stages=('decision','world'))
        rng=torch.get_rng_state().clone()
        for generation in range(12):
            for stage in('decision','world'):
                for position in range(50):
                    model.position=original.position=position
                    x=torch.full((1,17),generation/8+(stage=='world')/8+position/64)
                    reuse=c.begin(stage=stage,generation=generation,position=position,total_steps=50,condition_key=(stage,generation),input_key=(1,17))
                    assert reuse==(10<=position<40 and position%2==1)
                    actual=model(x);expected=original(x)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    if reuse:assert float(c.alpha)==(1. if position==11 else .5)
                    c.end()
        r=c.report();assert r['alpha_histogram']=={'1.0':24,'0.5':336}
        assert r['counts']==dict(decision_calls=600,decision_refresh=420,decision_reuse=180,world_calls=600,world_refresh=420,world_reuse=180)
        assert len(r['completed_generations'])==24 and model.head.calls==1200
        if scope=='deep':assert model.input_blocks[0].calls==model.output_blocks[-1].calls==1200
        assert r['predictive_feature_reuse']and r['residual_cache_FP32']and torch.equal(rng,torch.get_rng_state())
        c.close()
    for changed in('master','condition'):
        model=Model().eval();c=Controller(model,input_start=1,output_end=1)
        for position in range(17):
            model.position=position
            if position==12 and changed=='master':model.middle_block.weight.add_(.125)
            condition='new'if changed=='condition'and position>=12 else'old'
            use=c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=condition,input_key=(1,17))
            if position in(12,13):assert not use,'Need two fresh full evaluations after invalidation'
            if position==15:assert use and float(c.alpha)==1.
            model(torch.full((1,17),position/64));c.end()
        c.close()
    # Repeated warm-up at a single position must not count as two past steps.
    model=Model().eval();c=Controller(model,input_start=1,output_end=1)
    for position in range(50):
        model.position=position
        c.begin(stage='world',generation=0,position=position,total_steps=50,condition_key=0,input_key=(1,17))
        value=model(torch.full((1,17),position/64))
        if position==0:
            for _ in range(4):model(torch.full((1,17),position/64))
        c.end()
    assert c.report()['counts']['world_reuse']==15;c.close()
    for dtype in(torch.float16,torch.float32):
        for residual in(False,True):
            x=torch.arange(257,dtype=torch.float32)/256;previous=x-.125;alpha=torch.tensor([.5]);live=x.to(dtype)if residual else None
            expected=x+(x-previous)*alpha
            if residual:expected=expected+live.float()
            assert torch.equal(predict(x,previous,alpha,live,output_dtype=dtype),expected.to(dtype))
    # A generation may not skip any sampler position.
    model=Model().eval();c=Controller(model,input_start=1,output_end=1)
    c.begin(stage='world',generation=0,position=0,total_steps=50,condition_key=0,input_key=(1,17));model(torch.zeros(1,17));c.end()
    try:c.begin(stage='world',generation=0,position=2,total_steps=50,condition_key=0,input_key=(1,17))
    except AssertionError:pass
    else:raise AssertionError('Skipped position accepted')
    c.close();assert not torch.cuda.is_initialized()
    print('CROSSSTEP_PREDICT_CPU passed=true affine_complete_outputs=3600 alpha_first1_steadyHalf=true two_fresh_after_invalidation=true warmup_not_extra_step=true FP32_residual=true RNG_unchanged=true no_GPU=true')

if __name__=='__main__':main()

@torch.no_grad()
def check_cache_scope():
    # New kernels may use a custom manager only within their lexical scope.
    # Triton2.3 caches the class globally even when the env var is removed.
    import os,sys,tempfile
    import unifolm_wma.utils.crossstep_feature_reuse as cache_module
    sys.path.insert(0,str(Path(cache_module.__file__).resolve().parents[3]/'scripts/evaluation'))
    import triton.runtime.cache as cache
    from unifolm_wma.utils.crossstep_feature_reuse import kernel_cache_scope
    from unifolm_wma.utils.nonv2_ffn59_mode import candidate_cache
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as temporary:
        root=Path(temporary);(root/'rank0').mkdir();(root/'rank0/gpu').mkdir();(root/'rank0/exec').mkdir();(root/'ffn').mkdir()
        old=dict(os.environ);old_class=cache.__dict__['__cache_cls'];old_name=cache.__dict__['__cache_cls_nme']
        try:
            os.environ.update(TEAM07_PREDICT_CACHE=str(root),TEAM07_PARALLEL_RANK='0',NONV2_FFN_CACHE_DIR=str(root/'ffn'))
            os.environ.pop('TRITON_CACHE_MANAGER',None)
            before={key:os.environ.get(key)for key in('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','EXPERIMENT_EXEC_CACHE')}
            with kernel_cache_scope():
                obj=cache.get_cache_manager('team07-static-time-lexical-CPU-check')
                assert type(obj).__name__=='ReadWarmSplitCache'and root/'rank0/gpu'in Path(obj.cache_dir).parents
            assert {key:os.environ.get(key)for key in before}==before
            assert cache.__dict__['__cache_cls']is old_class and cache.__dict__['__cache_cls_nme']==old_name
            with candidate_cache(0):
                obj=cache.get_cache_manager('team07-original-FFN-CPU-check');assert type(obj)is cache.FileCacheManager
        finally:
            os.environ.clear();os.environ.update(old);cache.__dict__['__cache_cls']=old_class;cache.__dict__['__cache_cls_nme']=old_name
    print('PREDICT_CACHE_SCOPE_CPU passed=true environment_and_Triton_class_restored=true original_FFN_assertion_retained=true no_GPU=true')

if __name__=='__main__':check_cache_scope()

print('PENCILS_WINDOW65_CPU window20_40=true all1200_calls=true no_GPU=true')

@torch.no_grad()
def check_actual_late23():
    import copy
    class Affine(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.5));self.position=0;self.calls=0
        def forward(self,x):self.calls+=1;return x*self.weight+.125+self.position/64
    class Actual(torch.nn.Module):
        def __init__(self):
            super().__init__();self.input_blocks=torch.nn.ModuleList([Affine()for _ in range(12)]);self.middle_block=Affine();self.output_blocks=torch.nn.ModuleList([Affine()for _ in range(12)]);self.head=Affine();self.position=0
        def forward(self,x):
            for m in [*self.input_blocks,self.middle_block,*self.output_blocks,self.head]:m.position=self.position;x=m(x)
            return x
    backbone=Actual().eval();reference=copy.deepcopy(backbone);executor=Executor(backbone)
    outer=SimpleNamespace(model=SimpleNamespace(diffusion_model=backbone),apply_model=executor)
    runtime=Runtime(outer,input_start=1,output_end=11,start=20,stop=46,stages=('decision','world'))
    assert len(runtime.controller.saved)==23
    rng=torch.get_rng_state().clone()
    for generation in range(8):
        for stage in('decision','world'):
            for position in range(50):
                backbone.position=reference.position=position
                x=torch.full((1,17),generation/8+(stage=='world')/8+position/64)
                runtime.set_step(stage,position)
                actual=outer.apply_model(x,None,None,torch.tensor([49-position]),dict(stage=stage,generation=generation),fs=None)
                torch.testing.assert_close(actual,reference(x),rtol=0,atol=0)
                if 20<=position<46 and position%2==1:assert float(runtime.controller.alpha)==(1. if position==21 else .5)
                runtime.finish_step()
    r=runtime.report()
    assert executor.calls==800 and r['counts']==dict(decision_calls=400,decision_refresh=296,decision_reuse=104,world_calls=400,world_refresh=296,world_reuse=104)
    assert r['alpha_histogram']=={'1.0': 16, '0.5': 192}
    assert backbone.input_blocks[0].calls==backbone.output_blocks[-1].calls==backbone.head.calls==800
    assert backbone.middle_block.calls==592
    assert len(r['completed_generations'])==16 and all(g['steps']==50 for g in r['completed_generations'])
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    runtime.controller.close()
    print('NONV2_STOP46_CPU passed=true actual23_parents=true all800_affine_outputs_exact=true window20_46=true full50_steps=true shallow_and_head_live=true RNG_unchanged=true no_GPU=true')
if __name__=='__main__':check_actual_late23()
