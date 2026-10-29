"""Verify module contracts and that reuse calls retain the parallel executor."""
import json,torch
from pathlib import Path
from types import MethodType,SimpleNamespace
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.crossstep_feature_runtime import Runtime
from unifolm_wma.utils.nccl_nonv2_native_shared_runtime68 import partition_enabled

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
        self.denoiser_graph=DenoiserGraph(MethodType(original,model));self.calls=0;self.partition_flags=[]
    def __call__(self,*args,**kwargs):
        self.calls+=1;assert '_crossstep_feature_reuse'in kwargs
        self.partition_flags.append(partition_enabled(kwargs['_crossstep_feature_reuse']))
        return self.denoiser_graph(*args,**kwargs)

@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
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
    assert sum(executor.partition_flags)==960 and len(executor.partition_flags)==1200
    assert report['counts']==dict(decision_calls=600,decision_refresh=480,decision_reuse=120,world_calls=600,world_refresh=480,world_reuse=120)
    assert len(report['completed_generations'])==24 and all(g['steps']==50 for g in report['completed_generations'])
    assert backbone.input_blocks[0].calls==backbone.output_blocks[-1].calls==backbone.head.calls==1200
    assert backbone.middle_block.calls==960
    assert torch.equal(rng,torch.get_rng_state())and not torch.cuda.is_initialized()
    runtime.controller.close()
    print('G1_EXECUTOR70_CPU passed=true all1200_calls_through_parallel_executor=true partition_modes_checked=true no_GPU=true full_video_quality_not_inferred=true')

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
            os.environ['TRITON_CACHE_DIR']=str(root/'ffn')
            obj=cache.get_cache_manager('team07-original-native-CPU-check');assert type(obj)is cache.FileCacheManager
        finally:
            os.environ.clear();os.environ.update(old);cache.__dict__['__cache_cls']=old_class;cache.__dict__['__cache_cls_nme']=old_name
    print('PREDICT_CACHE_SCOPE_CPU passed=true environment_and_Triton_class_restored=true original_native_cache_class_restored=true no_GPU=true')

if __name__=='__main__':check_cache_scope()

print('G1_WINDOW70_CPU window20_40=true all1200_calls=true no_GPU=true')

print("G1_PREDICTION70_CPU passed=true reuse_partitions_disabled=true full960_reuse240_executor_calls=true no_new_native_inspection=true")

@torch.no_grad()
def check_refresh_control():
    backbone=Backbone().eval();executor=Executor(backbone)
    outer=SimpleNamespace(model=SimpleNamespace(diffusion_model=backbone),apply_model=executor)
    runtime=Runtime(outer,input_start=1,output_end=1,start=20,stop=40,stages=())
    assert runtime.graph.max_graphs==2
    for generation in range(12):
        for stage in('decision','world'):
            for position in range(50):
                runtime.set_step(stage,position)
                outer.apply_model(torch.ones(1,4),None,None,torch.tensor([49-position]),{'generation':generation,'stage':stage},fs=None)
                runtime.finish_step()
    report=runtime.report()
    assert report['counts']==dict(decision_calls=600,decision_refresh=600,world_calls=600,world_refresh=600)
    assert executor.calls==sum(executor.partition_flags)==backbone.middle_block.calls==backbone.head.calls==1200
    runtime.controller.close();assert not torch.cuda.is_initialized()
    print('G1_FULL_REFRESH_CONTROL_CPU passed=true full1200_calls=true prediction_calls0=true max_graphs2=true no_GPU=true')

if __name__=='__main__':check_refresh_control()

"""Independent polynomial and dynamic-slot checks without CUDA initialization."""
import torch
from unifolm_wma.utils.feature_trajectory_observer71 import record,estimate,positions,scalar_error


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    for dtype in(torch.float16,torch.float32):
        value=torch.arange(257,dtype=torch.float32).to(dtype);table=torch.full((100,257),float('nan'),dtype=dtype);cursor=torch.tensor([0],dtype=torch.int64)
        for position in(0,49,50,99):
            cursor.fill_(position);record(value,table,cursor);assert torch.equal(table[position],value)
        before=table.view(torch.int16 if dtype==torch.float16 else torch.int32).clone()
        for position in(-1,100):cursor.fill_(position);record(value,table,cursor)
        assert torch.equal(before,table.view(before.dtype))
    assertions=0
    for start,stop in((10,40),(20,40),(30,40)):
        rows=positions(start,stop);assert len(rows)==(stop-start)//2
        for degree in(0,1,2):
            data=torch.stack([torch.arange(17)/16+(float(t)**degree)/2048 for t in range(50)])
            for position,history in rows:
                for order in range(degree,3):
                    actual=estimate(data,position,history,order)
                    torch.testing.assert_close(actual,data[position],rtol=0,atol=0)
                    assert scalar_error(actual,data[position])['mse']==0
                    assertions+=1
    error=scalar_error(torch.ones(17),torch.zeros(17));assert error['mse']==1 and error['max_abs']==1 and error['elements']==17
    assert not torch.cuda.is_initialized()
    print('FEATURE_TRAJECTORY71_CPU passed=true exact_polynomial_assertions='+str(assertions)+' mutable_slots_and_bounds=true no_GPU=true')


if __name__=='__main__':main()
