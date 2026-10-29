"""Move a CPU model once over H2D, then broadcast exact typed CUDA buckets.

This changes initial transport only. Parameter objects, aliases, dtypes, shapes
and contiguous strides are preserved. Every subsequent inference stays live.
No weights or tensor contents are exported. All allocation and communication
belongs inside the original model_to_gpu stage.
"""
import hashlib,json,time
import torch
import torch.distributed as dist


def collect(model):
    entries={}
    for module in model.modules():
        for kind,slots in (('parameter',module._parameters),('buffer',module._buffers)):
            for name,tensor in slots.items():
                if tensor is None:continue
                key=id(tensor)
                if key not in entries:entries[key]=dict(tensor=tensor,slots=[])
                entries[key]['slots'].append((module,kind,name))
    return list(entries.values())


def buckets(entries,limit):
    groups={}
    for entry in entries:groups.setdefault(entry['tensor'].dtype,[]).append(entry)
    for dtype,group in groups.items():
        batch=[];size=0
        for entry in group:
            n=entry['tensor'].numel()*entry['tensor'].element_size()
            if batch and size+n>limit:yield dtype,batch;batch=[];size=0
            batch.append(entry);size+=n
        if batch:yield dtype,batch


def layout(batch):
    offset=0;rows=[]
    for entry in batch:
        tensor=entry['tensor'];alignment=256//tensor.element_size()
        offset=((offset+alignment-1)//alignment)*alignment
        rows.append((entry,offset));offset+=tensor.numel()
    return rows,offset


@torch.no_grad()
def move_model(model,*,bucket_bytes=256*2**20):
    assert dist.is_initialized()and dist.get_world_size()==2 and not torch.cuda.is_current_stream_capturing()
    entries=collect(model);rank=dist.get_rank()
    supported={torch.float16,torch.float32,torch.float64,torch.bfloat16,torch.int8,torch.uint8,torch.int32,torch.int64,torch.bool}
    eligible=all(e['tensor'].device.type=='cpu'and e['tensor'].layout==torch.strided and e['tensor'].is_contiguous()and e['tensor'].dtype in supported for e in entries)
    identity=[(str(e['tensor'].dtype),list(e['tensor'].shape),list(e['tensor'].stride()),len(e['slots']))for e in entries]
    digest=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()
    checks=[None,None];dist.all_gather_object(checks,(eligible,digest));assert checks[0]==checks[1]
    if not eligible:
        result=model.cuda();torch.cuda.synchronize()
        return result,dict(root_only_H2D=False,fallback_native_cuda=True,no_precision_change=True)
    start=time.perf_counter();count=0;total=0;largest=0
    for dtype,batch in buckets(entries,bucket_bytes):
        spans,size=layout(batch)
        flat=torch.empty(size,device='cuda',dtype=dtype)
        for entry,offset in spans:
            tensor=entry['tensor'];view=flat[offset:offset+tensor.numel()].view(tensor.shape).as_strided(tensor.shape,tensor.stride())
            if rank==0:view.copy_(tensor)
        dist.broadcast(flat,src=0)
        for entry,offset in spans:
            tensor=entry['tensor'];view=flat[offset:offset+tensor.numel()].view(tensor.shape).as_strided(tensor.shape,tensor.stride())
            assert view.stride()==tensor.stride()or tensor.numel()==0
            for module,kind,name in entry['slots']:
                if kind=='parameter':
                    parameter=module._parameters[name];parameter.data=view
                else:module._buffers[name]=view
        n=flat.numel()*flat.element_size();total+=n;largest=max(largest,n);count+=1
    torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
    report=dict(root_only_H2D=True,fallback_native_cuda=False,no_precision_change=True,original_parameter_objects_and_aliases=True,original_contiguous_strides=True,all_tensor_starts_256_byte_aligned=True,tensors=len(entries),buckets=count,total_bytes=total,max_bucket_bytes=largest,transport_wall_s=time.perf_counter()-start,no_operand_export=True)
    return model,report
