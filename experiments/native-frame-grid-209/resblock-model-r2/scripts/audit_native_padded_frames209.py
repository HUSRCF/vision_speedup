"""Synthetic native Conv2d inventory with full N16 descriptors and eight live frames.

Read-only CUDA Graph inspection: no node parameters or launch grids are changed.
Export only shapes, names, launch dimensions and numerical aggregate statistics.
"""
from datetime import datetime,timezone
import hashlib,importlib.util,json
from pathlib import Path
import torch
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI,eligible_plan
import unifolm_wma.utils.native_graph_weight_cache as production


def bits(left,right):
    return torch.equal(left.contiguous().view(torch.int32),right.contiguous().view(torch.int32))


def layout(value,kind):
    if kind=='NCHW':return value.contiguous()
    if kind=='NC_permuted':return value.transpose(0,1).contiguous().transpose(0,1)
    assert kind=='channels_last'
    return value.contiguous(memory_format=torch.channels_last)


def run(extension,api,co,ci,h,w,rank,kind):
    shape=(16,ci,h,w)
    module=torch.nn.Conv2d(ci,co,3,padding=1,bias=True,device='cuda',dtype=torch.float32).eval()
    module.requires_grad_(False)
    original=torch.randn(shape,device='cuda',dtype=torch.float32)*.1
    original+=torch.arange(16,device='cuda',dtype=torch.float32).view(16,1,1,1)*.013
    original=layout(original,kind)
    padded=layout(torch.zeros(shape,device='cuda',dtype=torch.float32),kind)
    padded[:8].copy_(original[rank*8:(rank+1)*8])
    reference=module(original).clone()
    eager=module(padded).clone()
    original_stride=tuple(original.stride());padded_stride=tuple(padded.stride())
    assert original_stride==padded_stride
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):module(padded)
    torch.cuda.current_stream().wait_stream(stream)
    pool=torch.cuda.graph_pool_handle();graph=torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph,pool=pool):output=module(padded)
    graph.replay();torch.cuda.synchronize()
    assert output.data_ptr()!=reference.data_ptr() and output.data_ptr()!=eager.data_ptr()
    raw,executable=extension.get_handles(graph)
    inventory=api.read(raw,dict(weight=module.weight));plan=eligible_plan(inventory,module.weight)
    consumer=None
    if plan is not None:
        transform,node,pointer_byte,old_filter,passthrough=plan
        consumer=dict(transform_name=transform['name'],name=node['name'],grid=node['grid'],
            block=node['block'],shared_bytes=node['shared_bytes'],
            argument_sizes=[a['size'] for a in node['arguments']],intervening_memset_nodes=len(passthrough),
            eligible_plan_validated=True)
    target=reference[rank*8:(rank+1)*8]
    same=bits(output[:8],target);captured=bits(output,eager)
    delta=output[:8]-target
    row=dict(co=co,ci=ci,height=h,width=w,rank=rank,layout=kind,
        logical_batch=16,live_frames=8,strides=list(padded_stride),
        full_descriptor_padding_preserved_original_stride=True,
        padded_eager_first8_signedzero_bits=bits(eager[:8],target),
        graph_matches_padded_eager_all16_signedzero_bits=captured,
        padded_first8_matches_original_frame_half_signedzero_bits=same,
        first8_max_abs=float(delta.abs().max()),
        first8_relative_l2=float(delta.double().norm()/target.double().norm().clamp_min(1e-30)),
        graph_nodes=len(inventory['nodes']),consumer=consumer,
        kernel_launches=[dict(name=n['name'],grid=n['grid'],block=n['block'],shared_bytes=n['shared_bytes'],
            argument_sizes=[a['size'] for a in n['arguments']])
            for n in inventory['nodes'] if n['type']==0],
        graph_parameter_writes=False,grid_shrink_unverified=True)
    graph.reset();return row


def main():
    torch.set_grad_enabled(False)
    assert torch.__version__=='2.3.1+cu121'
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    cudnn_tf32=torch.backends.cudnn.allow_tf32;matmul_tf32=torch.backends.cuda.matmul.allow_tf32
    deterministic=torch.backends.cudnn.deterministic;benchmark=torch.backends.cudnn.benchmark
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory=Path.home()/'benchmark-runs/diagnostics'/('native-padded-frame-inventory-'+stamp)
    directory.mkdir(exist_ok=False);print('DIAGNOSTIC_DIR='+str(directory),flush=True)
    so=Path(production.__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'
    digest=hashlib.sha256(so.read_bytes()).hexdigest()
    assert digest=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    api=GraphAPI();rows=[]
    try:
        for co,ci,h,w in ((320,320,40,64),(640,640,20,32),(1280,1280,10,16),
                         (1280,1280,5,8),(640,1920,20,32),(320,960,40,64)):
            for kind in ('NCHW','NC_permuted','channels_last'):
                for rank in (0,1):
                    row=run(extension,api,co,ci,h,w,rank,kind);rows.append(row)
                    print('NATIVE_PADDED_FRAME '+json.dumps(row),flush=True)
                    (directory/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        torch.cuda.synchronize();torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
        torch.backends.cudnn.allow_tf32=cudnn_tf32;torch.backends.cuda.matmul.allow_tf32=matmul_tf32
        torch.backends.cudnn.deterministic=deterministic;torch.backends.cudnn.benchmark=benchmark
        torch.use_deterministic_algorithms(deterministic_algorithms)
    assert len(rows)==36
    result=dict(diagnostic_complete=True,synthetic_only=True,rows=rows,
        all36_padded_first8_signedzero_bit_exact=all(r['padded_first8_matches_original_frame_half_signedzero_bits'] for r in rows),
        all36_native_capture_signedzero_bit_exact=all(r['graph_matches_padded_eager_all16_signedzero_bits'] for r in rows),
        eligible_consumer_plans=sum(r['consumer'] is not None for r in rows),
        original_extension_SHA256=digest,graph_parameter_writes=False,
        no_raw_operands_or_addresses_exported=True,no_official_score=True,
        grid_shrink_or_parallel_speedup_unverified=True)
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE_PADDED_FRAME_INVENTORY_CHECK passed=true',flush=True)


if __name__=='__main__':main()
