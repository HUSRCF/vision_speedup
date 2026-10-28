"""Read-only cross-attention operands and query sharing on actual stages."""
import hashlib,json
import torch
from unifolm_wma.modules.attention import CrossAttention

class Audit:
    def __init__(self,model,stage):
        assert stage in ('decision','world')
        self.stage=stage
        self.names={id(module):name for name,module in model.named_modules() if isinstance(module,CrossAttention)}
        self.original=CrossAttention._efficient_attention
        self.first_queries={};self.ordinals={};self.records=[];self.payload={};self.signatures=set();self.queries={}
        self.snapshot_bytes=0;self.skipped=[];self.byte_limit=256*1024**2
        self.active=set();self.done=set();self.hooks=[]
        def begin(module,args):
            name=self.names[id(module)]
            if (not torch.cuda.is_current_stream_capturing() and name not in self.done and
                    bool(torch.backends.cudnn.allow_tf32)==(self.stage=='decision')):self.active.add(name)
        def end(module,args,output):
            name=self.names[id(module)]
            if name in self.active:self.active.remove(name);self.done.add(name)
        for module in model.modules():
            if id(module) in self.names:
                self.hooks.extend((module.register_forward_pre_hook(begin),module.register_forward_hook(end)))
        original=self.original
        def observed(module,q,k,v,attn_bias=None,use_torch=False):
            output=original(module,q,k,v,attn_bias=attn_bias,use_torch=use_torch)
            if (torch.cuda.is_current_stream_capturing() or q.ndim!=4 or q.shape==k.shape or
                    q.dtype!=torch.float32 or id(module) not in self.names):return output
            name=self.names[id(module)]
            if name not in self.active:return output
            identity=(name,q.data_ptr(),tuple(q.shape),tuple(q.stride()))
            if name not in self.first_queries:self.first_queries[name]=identity;self.ordinals[name]=0
            if self.first_queries[name]!=identity:return output
            ordinal=self.ordinals[name];self.ordinals[name]+=1
            signature=(tuple(q.shape),tuple(q.stride()),tuple(k.shape),tuple(k.stride()),tuple(v.stride()),attn_bias is not None,ordinal)
            if signature in self.signatures:return output
            self.signatures.add(signature)
            required=(0 if identity in self.queries else q.numel()*q.element_size())+sum(t.numel()*t.element_size() for t in (k,v))
            if isinstance(attn_bias,torch.Tensor):required+=attn_bias.numel()*attn_bias.element_size()
            if self.snapshot_bytes+required>self.byte_limit:
                self.skipped.append(dict(module=name,branch_ordinal=ordinal,required_bytes=required,budget_exceeded=True))
                return output
            self.snapshot_bytes+=required
            index=len(self.records);label='cross_'+str(index)
            if identity not in self.queries:self.queries[identity]=q.detach().clone()
            snapshot=dict(q=self.queries[identity],k=k.detach().clone(),v=v.detach().clone(),
                bias=attn_bias.detach().clone() if isinstance(attn_bias,torch.Tensor) else None)
            assert attn_bias is None or isinstance(attn_bias,torch.Tensor)
            self.payload[label]=snapshot
            self.records.append(dict(label=label,module=name,branch_ordinal=ordinal,
                q_shape=list(q.shape),k_shape=list(k.shape),v_shape=list(v.shape),
                q_stride=list(q.stride()),k_stride=list(k.stride()),v_stride=list(v.stride()),
                snapshot_q_stride=list(snapshot['q'].stride()),snapshot_k_stride=list(snapshot['k'].stride()),
                snapshot_v_stride=list(snapshot['v'].stride()),bias_shape=list(attn_bias.shape) if attn_bias is not None else None,
                bias_stride=list(attn_bias.stride()) if attn_bias is not None else None,
                scale=module.scale,use_torch=use_torch,query_identity_group=list(self.queries).index(identity),
                all_public_dtypes='float32',output_shape=list(output.shape)))
            return output
        CrossAttention._efficient_attention=observed;self.installed=observed

    def finish(self,directory):
        assert CrossAttention._efficient_attention is self.installed
        CrossAttention._efficient_attention=self.original
        for hook in self.hooks:hook.remove()
        assert self.records
        rng=torch.cuda.get_rng_state()
        # Preserve sharing of each cloned query when moving the snapshot to CPU.
        cpu_queries={identity:value.cpu() for identity,value in self.queries.items()}
        cpu={}
        for row in self.records:
            source=self.payload[row['label']]
            cpu[row['label']]={key:(cpu_queries[list(self.queries)[row['query_identity_group']]] if key=='q'
                else value.cpu() if value is not None else None) for key,value in source.items()}
        path=directory/'cross-context-operands.pt';torch.save(cpu,path)
        assert torch.equal(rng,torch.cuda.get_rng_state())
        report=dict(records=self.records,selected=len(self.records),query_groups=len(self.queries),
            captured_stage=self.stage,
            first_query_branch_counts=self.ordinals,source='actual eager operands before graph capture',
            original_function_restored=True,original_outputs_returned_without_changes=True,
            snapshot_query_identity_includes_module_name=True,only_first_eager_module_invocation=True,
            snapshot_tensor_byte_limit=self.byte_limit,snapshot_tensor_bytes=self.snapshot_bytes,budget_skipped=self.skipped,
            no_graph_or_weight_or_rng_mutations=True,no_performance_or_quality_claim=True,
            snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot_bytes=path.stat().st_size)
        (directory/'cross-context-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        self.queries.clear();self.payload.clear()
        print('CROSS_CONTEXT_209_AUDIT '+json.dumps(dict(selected=report['selected'],query_groups=report['query_groups'],bytes=report['snapshot_bytes'])),flush=True)
