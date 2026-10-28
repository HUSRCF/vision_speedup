"""Fresh synthetic native Winograd launch geometry: no graph edits or JIT.

Collect only names/grids/blocks/argument sizes, never argument bytes, pointers,
operands or weights. N8/N16 calibration is an observation, not a proof that a
reduced native grid is valid. A subsequent poisoned-workspace test is required.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from guard_world_transform209 import announce_worker
from unifolm_wma.utils.native_graph_weight_cache_probe import GraphAPI
from unifolm_wma.utils.native_graph_world_filter_cache_probe import eligible_world_plan


def arrange(value, kind):
    if kind=='NCHW':return value.contiguous()
    return value.permute(1,0,2,3).contiguous().permute(1,0,2,3)


def capture(module, value, extension, api):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):module(value)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph();graph.enable_debug_mode()
    with torch.cuda.graph(graph,pool=torch.cuda.graph_pool_handle()):output=module(value)
    expected=module(value).clone();graph.replay();torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert torch.equal(output.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    raw,_=extension.get_handles(graph)
    inventory=api.read(raw,dict(input=value,weight=module.weight,output=output))
    plan=eligible_world_plan(inventory,module.weight)
    nodes=[]
    for n in inventory['nodes']:
        if 'name' not in n:continue
        nodes.append(dict(name=n['name'],grid=n['grid'],block=n['block'],
            shared_bytes=n['shared_bytes'],argument_sizes=[a['size'] for a in n.get('arguments',[])]))
    graph.reset()
    return dict(batch=value.shape[0],input_shape=list(value.shape),input_stride=list(value.stride()),
                native_Winograd_plan_eligible=plan is not None,nodes=nodes,
                captured_output_signedzero_bits=True)


@torch.no_grad()
def main():
    hold=announce_worker()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(2091616)
    canonical=Path(os.environ['CANONICAL_ROOT'])
    so=canonical/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec=importlib.util.spec_from_file_location('team07_cuda_graph_handles',so)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    api=GraphAPI();rows=[]
    specs=json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'matrix-geometries.json').read_text())
    for geometry in specs:
        co,cols,ci=geometry['geometry'];h,w={2560:(40,64),640:(20,32),192:(10,16)}[cols]
        module=torch.nn.Conv2d(ci,co,3,padding=1,bias=True,device='cuda').eval().requires_grad_(False)
        source=torch.randn(16,ci,h,w,device='cuda')*.1
        for kind in ('NCHW','NC_permuted'):
            values={batch:arrange(source[:batch].clone(),kind) for batch in (16,8)}
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            samples=[capture(module,values[batch],extension,api) for batch in (16,8)]
            assert samples[0]['native_Winograd_plan_eligible'],(geometry,kind)
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            calibration=[]
            for n16 in samples[0]['nodes']:
                if 'winogradForwardData4x4' not in n16['name'] and 'winogradForwardOutput4x4' not in n16['name']:continue
                matching=[n for n in samples[1]['nodes'] if n['name']==n16['name'] and n['block']==n16['block'] and n['argument_sizes']==n16['argument_sizes']]
                if len(matching)!=1:continue
                n8=matching[0]
                axes=[axis for axis in range(3) if n16['grid'][axis]==2*n8['grid'][axis] and
                      all(n16['grid'][j]==n8['grid'][j] for j in range(3) if j!=axis)]
                calibration.append(dict(name=n16['name'],full_grid=n16['grid'],batch8_grid=n8['grid'],
                                        observed_halving_axes=axes,grid_reduction_validity_unproven=True))
            row=dict(geometry=geometry['geometry'],layout=kind,samples=samples,calibration=calibration,RNG_unchanged=True)
            rows.append(row)
            Path('progress.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
            print('NATIVE_TRANSFORM_GEOMETRY '+json.dumps(dict(geometry=row['geometry'],layout=kind,calibration=calibration)),flush=True)
            del values
        del module,source
    assert len(rows)==28
    report=dict(diagnostic_complete=True,all_dynamic_gates_passed=True,synthetic_only=True,
                no_grid_or_graph_edits=True,no_new_Triton_kernels_compiled=True,
                no_argument_bytes_addresses_weights_or_operands_exported=True,
                reduced_grid_numerics_unverified=True,model_or_video_quality_verified=False,
                rows=rows,native_capture_checks=56)
    Path('comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_TRANSFORM_GEOMETRY_CHECK complete=true',flush=True)


if __name__=='__main__':main()
