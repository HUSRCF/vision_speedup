"""Declared-Tensor graph-safe feature taps; export scalar errors only."""
import json
from pathlib import Path
import torch
import triton
import triton.language as tl
from .crossstep_feature_reuse import kernel_cache_scope


@triton.jit
def _record_feature(X,T,Cursor,N:tl.constexpr,ROWS:tl.constexpr,BLOCK:tl.constexpr,TAG:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.load(Cursor)
    value=tl.load(X+i,i<N,other=0)
    tl.store(T+row*N+i,value,(i<N)&(row>=0)&(row<ROWS))


def record(value,table,cursor):
    assert value.device==table.device==cursor.device and value.dtype==table.dtype
    assert table.ndim==2 and table.is_contiguous()and table.shape[1]==value.numel()
    assert cursor.dtype==torch.int64 and cursor.numel()==1
    flat=value.contiguous().view(-1)
    if not value.is_cuda:
        row=int(cursor.item())
        if 0<=row<table.shape[0]:table[row].copy_(flat)
        return
    with kernel_cache_scope():
        _record_feature[(triton.cdiv(flat.numel(),256),)](flat,table,cursor,flat.numel(),table.shape[0],256,1793277001,num_warps=4)


def positions(start,stop):
    history=[];rows=[]
    for position in range(50):
        if start<=position<stop and(position-start)%2:
            assert len(history)==3
            rows.append((position,tuple(history)))
        else:history=(history+[position])[-3:]
    return rows


def estimate(table,position,history,order):
    old,previous,last=history;assert old<previous<last<position and order in(0,1,2)
    latest=table[last].float()
    if order==0:return table[last]
    first=latest-table[previous].float()
    alpha=(position-last)/(last-previous)
    result=latest+first*alpha
    if order==2:
        second=table[previous].float()-table[old].float()
        ratio=(last-previous)/(previous-old)
        beta=(position-last)*(position-previous)/((last-previous)*(last-old))
        result=result+(first-second*ratio)*beta
    return result.to(table.dtype)


def scalar_error(predicted,actual):
    assert predicted.shape==actual.shape and predicted.dtype==actual.dtype
    delta=predicted.double()-actual.double()
    squared=delta.square().sum()
    energy=actual.double().square().sum()
    values=torch.stack((squared,energy,delta.abs().max(),torch.tensor(actual.numel(),device=actual.device,dtype=torch.float64)))
    squared,energy,maximum,count=values.tolist()
    return dict(elements=int(count),mse=squared/count,relative_l2=(squared/max(energy,1e-300))**.5,max_abs=maximum)


@torch.no_grad()
def gpu_selfcheck(device):
    rows=[]
    for dtype in(torch.float16,torch.float32):
        for size in(257,1025):
            source=(torch.arange(size,device=device,dtype=torch.float32)/32).to(dtype)
            source[0]=-0.;source[1]=float('nan')
            table=torch.full((4,size),float('nan'),device=device,dtype=dtype);cursor=torch.tensor([0],device=device,dtype=torch.int64)
            stream=torch.cuda.Stream(device=device);parent=torch.cuda.current_stream(device);stream.wait_stream(parent)
            with torch.cuda.stream(stream):record(source,table,cursor)
            parent.wait_stream(stream);torch.cuda.synchronize(device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):record(source,table,cursor)
            for position in(0,3,1):
                source.add_(.125);source[0]=-0.;source[1]=float('nan');cursor.fill_(position);graph.replay();torch.cuda.synchronize(device)
                width=torch.int16 if dtype==torch.float16 else torch.int32
                assert torch.equal(table[position].view(width),source.view(width))
            cursor.fill_(-1);before=table.clone();graph.replay();torch.cuda.synchronize(device)
            width=torch.int16 if dtype==torch.float16 else torch.int32
            assert torch.equal(table.view(width),before.view(width))
            graph.reset();rows.append(dict(dtype=str(dtype),elements=size,mutable_cursor_and_source=True,tail_exact=True,NaN_and_signed_zero_bits_exact=True,invalid_cursor_no_write=True))
    return dict(passed=True,cases=rows)


class Observer:
    def __init__(self,model,runtime):
        assert runtime.controller.stages==()and runtime.graph.max_graphs==2
        self.runtime=runtime;self.tables={};self.shapes={};self.saved=[];self.statistics=[];self.generations=[]
        self.cursor=torch.tensor([-1],device=next(model.parameters()).device,dtype=torch.int64)
        self.selfcheck=gpu_selfcheck(self.cursor.device);self.bytes=0
        names={'output_blocks.2':'deep7','output_blocks.5':'deep13','output_blocks.10':'deep23'}
        modules=dict(model.named_modules());assert names.keys()<=modules.keys()
        for name,label in names.items():
            module=modules[name];before=module.forward
            def forward(*args,_before=before,_name=name,_label=label,**kwargs):
                value=_before(*args,**kwargs)
                assert isinstance(value,torch.Tensor)and value.dtype==torch.float16 and value.is_cuda
                if _name not in self.tables:
                    assert not torch.cuda.is_current_stream_capturing()
                    amount=100*value.numel()*value.element_size()
                    assert self.bytes+amount<=16*2**30,'feature-table diagnostic budget exceeded'
                    free,_=torch.cuda.mem_get_info(value.device);assert free>amount+8*2**30
                    self.tables[_name]=torch.full((100,value.numel()),float('nan'),device=value.device,dtype=value.dtype)
                    self.shapes[_name]=dict(label=_label,shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),bytes=amount)
                    self.bytes+=amount
                record(value,self.tables[_name],self.cursor)
                return value
            module.forward=forward;self.saved.append((module,before))
        old_set=runtime.set_step;old_finish=runtime.finish_step
        def set_step(stage,position):
            old_set(stage,position)
            self.cursor.fill_((0 if stage=='decision'else 50)+position)
        def finish_step():
            stage,generation,position=runtime.step
            old_finish()
            if position==49:self.analyze(stage,generation)
        runtime.set_step=set_step;runtime.finish_step=finish_step

    @torch.no_grad()
    def analyze(self,stage,generation):
        assert len(self.tables)==3
        offset=0 if stage=='decision'else 50
        for name,storage in self.tables.items():
            table=storage[offset:offset+50]
            assert torch.isfinite(table).all().item(),'not all50 feature slots written with finite values'
            for start,stop in((10,40),(20,40),(30,40)):
                for position,history in positions(start,stop):
                    for order in(0,1,2):
                        error=scalar_error(estimate(table,position,history,order),table[position])
                        self.statistics.append(dict(module=name,scope=self.shapes[name]['label'],stage=stage,generation=generation,window=[start,stop],position=position,history=list(history),order=order,**error))
        self.generations.append(dict(stage=stage,generation=generation,slots=50,all_finite=True))

    def report(self):
        assert self.runtime.step is None and len(self.generations)==22
        assert len(self.statistics)==5940
        return dict(read_only_feature_trajectory_observer=True,full_refresh_reference_only=True,
            no_features_or_weights_exported=True,no_new_native_argument_or_address_inspection=True,
            all50_steps_and11_rounds_preserved=True,inference_outputs_returned_unchanged=True,
            offline_errors_do_not_include_approximate_feedback=True,full_video_SHA_validation_required=True,
            primitive_GPU_selfcheck=self.selfcheck,persistent_feature_bytes=self.bytes,tables=self.shapes,
            completed_generations=self.generations,statistics=self.statistics)


active=None


def install(model):
    global active
    from . import crossstep_feature_runtime as feature
    assert active is None and feature.active is not None
    active=Observer(model.model.diffusion_model,feature.active)


def finish(args):
    from . import nccl_nonv2_native_shared_runtime68 as parallel
    assert active is not None
    parallel.write_json('rank'+str(parallel._rank)+'-feature-trajectory.json',active.report())
