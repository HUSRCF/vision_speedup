"""Synthetic-only Half row partition with owned padded Torch inputs/outputs.

Driver argument buffers remain in process. No argument bytes, addresses or
parameter offsets are printed or serialized. Reject unbound/workspace plans.
"""
import ctypes as C
import hashlib,importlib.util,json,os,statistics,struct
from pathlib import Path
import torch
import torch.distributed as dist
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI,OwnedParams


def bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


class RowOwner:
    def __init__(self,graph,x,w,y,extension,api,rank):
        self.api=api;self.rank=rank;self.raw,self.exe=extension.get_handles(graph)
        inventory=api.read(self.raw,dict(input=x,weight=w,output=y))
        def flatten(tree):
            for node in tree['nodes']:
                if 'child'in node:yield from flatten(node['child'])
                else:yield node
        selected=[n for n in flatten(inventory)if n['type']==0 and {'input','weight','output'}<=set(n.get('matched_tensors',[]))]
        assert len(selected)==1,'No unique kernel bound to all owned GEMM operands'
        self.node=selected[0];self.original=OwnedParams(self.node);self.closed=False
        assert x.ndim==w.ndim==y.ndim==2 and x.is_contiguous()and w.is_contiguous()and y.is_contiguous()
        assert x.dtype==w.dtype==y.dtype==torch.float16 and x.shape[0]==y.shape[0]and w.shape==(y.shape[1],x.shape[1])
        m=x.shape[0];assert m%2==0
        self.replacements={};seen=set()
        # Bind only exact starts of live owned input/output views, not guessed
        # workspace addresses or arbitrary offsets into device allocations.
        for argument in self.node['arguments']:
            data=bytes.fromhex(argument['bytes_hex'])
            for byte in range(0,len(data)-7,8):
                value=struct.unpack_from('<Q',data,byte)[0]
                for name,tensor in [('input',x),('output',y)]:
                    if value!=tensor.data_ptr():continue
                    seen.add(name);offset=self.rank*(m//2)*tensor.shape[1]*tensor.element_size()
                    assert tensor.storage_offset()==0 and tensor.untyped_storage().nbytes()>=2*tensor.numel()*tensor.element_size()
                    assert offset+tensor.numel()*tensor.element_size()<=tensor.untyped_storage().nbytes()
                    self.replacements[(argument['index'],byte)]=(value,value+offset)
        assert seen=={'input','output'}
    def enable(self,axis):
        assert not self.closed and self.node['grid'][axis]>1 and self.node['grid'][axis]%2==0
        self.modified=OwnedParams(self.node,self.replacements)
        field=['grid_x','grid_y','grid_z'][axis]
        setattr(self.modified.params,field,getattr(self.modified.params,field)//2)
        self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.exe,int(self.node['node'],16),C.byref(self.modified.params)),'set owned Half row candidate')
    def restore(self):
        if not self.closed:self.api.check(self.api.lib.cuGraphExecKernelNodeSetParams(self.exe,int(self.node['node'],16),C.byref(self.original.params)),'restore owned Half original')
    def close(self):self.restore();self.closed=True
    def public_metadata(self):
        return dict(kernel_name=self.node['name'],original_grid=self.node['grid'],block=self.node['block'],original_function_and_all_descriptor_bytes_except_owned_starts_preserved=True,only_owned_padded_Torch_views_bound=True,no_argument_bytes_addresses_or_offsets_exported=True)


def geometry(shape,extension,api):
    rank=dist.get_rank();m,n,k=shape
    xpad=torch.randn((2*m,k),device='cuda',dtype=torch.float16)*.1;x=xpad[:m]
    w=torch.randn((n,k),device='cuda',dtype=torch.float16)*.1
    bias=torch.randn(n,device='cuda',dtype=torch.float16)*.01
    ypad=torch.empty((2*m,n),device='cuda',dtype=torch.float16);y=ypad[:m]
    x0,w0,b0=x.clone(),w.clone(),bias.clone()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):torch.addmm(bias,x,w.T,out=y)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
    with torch.cuda.graph(graph):torch.addmm(bias,x,w.T,out=y)
    graph.replay();torch.cuda.synchronize();reference=torch.nn.functional.linear(x,w,bias)
    assert bits(y,reference),'Owned-output addmm differs from original F.linear'
    owner=None
    try:
        bind_error=None
        try:owner=RowOwner(graph,x,w,y,extension,api,rank)
        except AssertionError as error:bind_error=str(error)
        readiness=[None,None];dist.all_gather_object(readiness,owner is not None)
        if not all(readiness):return dict(shape=shape,supported=False,reason='No validated owned-operand kernel on both ranks',local_binding_error=bind_error,no_argument_export=True)
        begin,end=rank*m//2,(rank+1)*m//2
        def poison():
            ypad.fill_(float('nan'));y.copy_(bias.expand_as(y))
        def check_owned(expected):
            local=bits(y[begin:end],expected[begin:end])
            other=y[m//2:]if rank==0 else y[:m//2]
            untouched=bits(other,bias.expand_as(other))
            tail=bool(torch.isnan(ypad[m:]).all())
            return local and untouched and tail
        trials=[];axis=None
        for candidate_axis,size in enumerate(owner.node['grid']):
            if size<=1 or size%2:continue
            owner.enable(candidate_axis);poison();graph.replay();torch.cuda.synchronize();passed=check_owned(reference)
            trials.append(dict(axis=candidate_axis,owned_half_and_other_rows_and_padding_passed=passed))
            owner.restore()
            if passed:axis=candidate_axis;break
        readiness=[None,None];dist.all_gather_object(readiness,axis is not None)
        if not all(readiness):return dict(shape=shape,supported=False,reason='No halved grid axis preserves exact owned row domain on both ranks',axis_trials=trials,metadata=owner.public_metadata())
        owner.enable(axis);cases=[];rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
        for case in('initial','changed_inputs','changed_weights_and_bias','zero_weights','zero_inputs','restored'):
            x.copy_(x0);w.copy_(w0);bias.copy_(b0)
            if case=='changed_inputs':x.mul_(.75).add_(.0137)
            elif case=='changed_weights_and_bias':w.add_(.000137);bias.add_(.00071)
            elif case=='zero_weights':w.zero_()
            elif case=='zero_inputs':x.zero_()
            expected=torch.nn.functional.linear(x,w,bias)
            owner.restore();graph.replay();torch.cuda.synchronize();assert bits(y,expected)
            owner.enable(axis);poison();graph.replay();torch.cuda.synchronize();domain_ok=check_owned(expected);assert domain_ok,'Owned row or padded domain mismatch after mutation'
            full=torch.empty_like(y);dist.all_gather_into_tensor(full,y[begin:end].contiguous());torch.cuda.synchronize();assert bits(full,expected)
            cases.append(dict(case=case,full_reconstruction_signedzero_bits=True,owned_rows_and_unowned_rows_and_padding_passed=True,owned_addmm_and_F_linear_bitwise=True))
            del full,expected
        assert torch.equal(rng,torch.cuda.get_rng_state())and torch.equal(cpu_rng,torch.get_rng_state())
        pairs=[];start,finish=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for index in range(6):
            pair={}
            for label in(['baseline','candidate']if index%2==0 else['candidate','baseline']):
                owner.restore()if label=='baseline'else owner.enable(axis)
                dist.barrier();torch.cuda.synchronize()
                for _ in range(10):graph.replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(7):
                    start.record()
                    for _ in range(5):graph.replay()
                    finish.record();finish.synchronize();samples.append(start.elapsed_time(finish)/5)
                pair[label+'_samples_ms']=samples;pair[label+'_ms']=statistics.median(samples)
            pair['saved_ms']=pair['baseline_ms']-pair['candidate_ms'];pairs.append(pair)
        owner.restore();graph.replay();torch.cuda.synchronize();assert bits(y,torch.nn.functional.linear(x,w,bias))
        return dict(shape=shape,supported=True,row_axis=axis,axis_trials=trials,cases=cases,pairs=pairs,median_saved_ms=statistics.median(p['saved_ms']for p in pairs),RNG_unchanged=True,restored_original_bitwise=True,metadata=owner.public_metadata(),timing_is_GEMM_graph_only_excludes_gather=True,full_FFN_and_model_and_video_unverified=True)
    except AssertionError as error:
        if owner is None:return dict(shape=shape,supported=False,reason=str(error),no_argument_export=True)
        raise
    finally:
        if owner is not None:owner.close()
        graph.reset()


def main():
    torch.set_grad_enabled(False);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    canonical=Path(os.environ['CANONICAL_ROOT']);so=canonical/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so);extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension);api=GraphAPI()
    shapes=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'shapes.json').read_text());rows=[]
    for shape in shapes:
        rows.append(geometry(shape,extension,api));Path('rank'+str(dist.get_rank())+'-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('HALF_NATIVE_ROW_GRID '+json.dumps({k:rows[-1].get(k)for k in('shape','supported','reason','row_axis','median_saved_ms')}),flush=True)
    report=dict(synthetic_only=True,no_official_score=True,rows=rows,supported_shapes=sum(r['supported']for r in rows),all_supported_cases_bitwise=all(all(c['full_reconstruction_signedzero_bits']for c in r['cases'])for r in rows if r['supported']),full_FFN_and_model_and_video_unverified=True,no_kernel_argument_or_address_exports=True)
    Path('rank'+str(dist.get_rank())+'-report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
