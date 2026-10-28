"""Time real graph trunk/head boundaries with explicit driver event nodes.

Only adds timing nodes to a second diagnostic graph. All kernels, master
weights, cache owners and three output tensors are checked against the original
209-source production graph. No raw tensor values are written.
"""
import ctypes as C
import importlib.util
import json
from pathlib import Path
import runpy
import statistics
import sys
import torch
import torch.utils.checkpoint
from torch.utils._pytree import tree_flatten
import unifolm_wma.utils.inference_runtime as runtime


class Finished(Exception):
    pass


class Events:
    def __init__(self):
        self.lib = C.CDLL('libcuda.so.1')
        specs = {
            'cuStreamGetCaptureInfo_v2': [C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_ulonglong), C.POINTER(C.c_void_p), C.POINTER(C.POINTER(C.c_void_p)), C.POINTER(C.c_size_t)],
            'cuStreamUpdateCaptureDependencies': [C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.c_uint],
            'cuEventCreate': [C.POINTER(C.c_void_p), C.c_uint],
            'cuEventDestroy_v2': [C.c_void_p],
            'cuEventElapsedTime': [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            'cuGraphAddEventRecordNode': [C.POINTER(C.c_void_p), C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.c_void_p],
        }
        for name, args in specs.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = C.c_int
        self.records = []
        self.spans = []

    @staticmethod
    def check(status, operation):
        assert status == 0, (operation, status)

    def mark(self, name):
        stream = torch.cuda.current_stream().cuda_stream
        status, capture_id, graph = C.c_int(), C.c_ulonglong(), C.c_void_p()
        deps, count = C.POINTER(C.c_void_p)(), C.c_size_t()
        self.check(self.lib.cuStreamGetCaptureInfo_v2(stream, C.byref(status), C.byref(capture_id), C.byref(graph), C.byref(deps), C.byref(count)), 'capture info')
        assert status.value == 1 and graph.value and count.value <= 32
        event, node = C.c_void_p(), C.c_void_p()
        self.check(self.lib.cuEventCreate(C.byref(event), 0), 'create timing event')
        self.check(self.lib.cuGraphAddEventRecordNode(C.byref(node), graph, deps, count, event), 'add timing node')
        new_deps = (C.c_void_p * 1)(node.value)
        self.check(self.lib.cuStreamUpdateCaptureDependencies(stream, new_deps, 1, 1), 'advance capture dependency')
        self.records.append(dict(name=name, event=event, graph=graph.value, capture_id=capture_id.value))

    def elapsed(self, first, last):
        by_name = {record['name']: record['event'] for record in self.records}
        assert len(by_name) == len(self.records)
        result = C.c_float()
        self.check(self.lib.cuEventElapsedTime(C.byref(result), by_name[first], by_name[last]), 'elapsed time')
        assert 0 <= result.value < 2000
        return result.value

    def close(self):
        for record in self.records:
            self.check(self.lib.cuEventDestroy_v2(record['event']), 'destroy timing event')
        self.records.clear()


def close_graph(wrapper):
    torch.cuda.synchronize()
    for entry in wrapper.entries.values():
        for key in ('numerical_sgemm', 'native_weight_cache'):
            if entry.get(key) is not None:
                entry[key].close()
        entry['graph'].reset()
    wrapper.entries.clear()


def exact(actual, expected):
    left, spec = tree_flatten(actual)
    right, other = tree_flatten(expected)
    assert spec == other and len(left) == len(right) == 3
    for a, b in zip(left, right):
        assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def main():
    root = Path.cwd()
    configured = runtime.configure_denoiser
    saved = []
    originals = []
    diagnostic = [None]
    active = [None]
    softmax_shapes = {}
    reports = {}

    def configure(model, *args, **kwargs):
        configured(model, *args, **kwargs)
        from unifolm_wma.utils.cuda_graph import DenoiserGraph
        network = model.model.diffusion_model
        original = model.apply_model
        assert isinstance(original, DenoiserGraph)
        originals.append(original)

        def wrap(module, name, kind):
            previous = module.forward
            def forward(*values, **options):
                events = active[0]
                tracing = events is not None and torch.cuda.is_current_stream_capturing()
                if tracing:
                    ticket = name + '@' + str(len(events.records))
                    events.mark(ticket + ':begin')
                result = previous(*values, **options)
                if tracing:
                    events.mark(ticket + ':end')
                    events.spans.append(dict(name=name,ticket=ticket,kind=kind))
                    if kind == 'softmax':
                        assert isinstance(result, torch.Tensor) and result.dtype == torch.float32
                        softmax_shapes[ticket] = dict(input=list(values[0].shape), output=list(result.shape), output_bytes=result.numel()*result.element_size())
                return result
            module.forward = forward
            saved.append((module, previous))

        wrap(network, 'trunk_and_head', 'network')
        for label in ('action', 'state'):
            head = getattr(network, label + '_unet')
            wrap(head, label + '_head', 'head')
            seen_softmax = set()
            for index, softmax in enumerate(head.spatial_softmax_blocks):
                # Symmetric scales reuse the same module object; instrument
                # each object once and give each actual invocation a ticket.
                if id(softmax) in seen_softmax:
                    continue
                seen_softmax.add(id(softmax))
                wrap(softmax, label + '_softmax_' + str(index), 'softmax')

        def apply(*values, **options):
            stage = 'decision' if torch.backends.cudnn.allow_tf32 else 'world'
            if stage in reports:
                return original(*values, **options)
            rng = torch.cuda.get_rng_state()
            versions = {name: parameter._version for name, parameter in model.named_parameters()}
            expected = original(*values, **options)
            entry = next(reversed(original.entries.values()))
            if stage == 'world':
                owner = entry['numerical_sgemm']
                assert len(owner.swaps) == 23 and len(owner._world_head_filter_cache.caches) == 45
            else:
                owner = entry['native_weight_cache']
                assert len(owner.caches) == 138 and len(owner._native_conv1d_weight_cache.caches) == 43 and len(owner._native_remaining_filter_cache.caches) == 8
            softmax_shapes.clear()
            events = Events()
            wrapper = DenoiserGraph(original.original, max_graphs=1,
                condition_refresher=original.condition_refresher,
                skip_unchanged_inputs=original.skip_unchanged_inputs)
            diagnostic[0] = wrapper
            active[0] = events
            try:
                actual = wrapper(*values, **options)
                active[0] = None
                torch.cuda.synchronize()
                exact(actual, expected)
                spans = events.spans
                (root/('head-pipeline-'+stage+'-spans.json')).write_text(json.dumps(dict(spans=spans,softmax_shapes=softmax_shapes),indent=2)+'\n')
                head = 'action' if stage == 'decision' else 'state'
                networks = [span for span in spans if span['kind']=='network']
                heads = [span for span in spans if span['kind']=='head']
                assert len(networks)==len(heads)==1,(networks,heads)
                assert heads[0]['name']==head+'_head'
                network_ticket,head_ticket = networks[0]['ticket'],heads[0]['ticket']
                assert len(softmax_shapes) == 9 and all(name.startswith(head+'_softmax_') for name in softmax_shapes),list(softmax_shapes)
                assert len({record['graph'] for record in events.records}) == 1
                candidate_entry = next(iter(wrapper.entries.values()))
                candidate_owner = candidate_entry['numerical_sgemm'] if stage == 'world' else candidate_entry['native_weight_cache']
                if stage == 'world':
                    assert len(candidate_owner.swaps) == 23 and len(candidate_owner._world_head_filter_cache.caches) == 45
                else:
                    assert len(candidate_owner.caches) == 138 and len(candidate_owner._native_conv1d_weight_cache.caches) == 43 and len(candidate_owner._native_remaining_filter_cache.caches) == 8
                for _ in range(12):
                    wrapper(*values, **options)
                torch.cuda.synchronize()
                samples = []
                for _ in range(18):
                    result = wrapper(*values, **options)
                    torch.cuda.synchronize()
                    total = events.elapsed(network_ticket+':begin', network_ticket+':end')
                    trunk = events.elapsed(network_ticket+':begin', head_ticket+':begin')
                    entire_head = events.elapsed(head_ticket+':begin', head_ticket+':end')
                    parts = {name: events.elapsed(name+':begin',name+':end') for name in softmax_shapes}
                    softmax = sum(parts.values())
                    assert abs(total-trunk-entire_head) < .1
                    assert 0 < softmax < entire_head and 0 < entire_head < total
                    samples.append(dict(total_ms=total, trunk_ms=trunk, spatial_softmax_ms=softmax,
                        head_remaining_ms=entire_head-softmax, entire_head_ms=entire_head,
                        spatial_softmax_parts_ms=parts))
                exact(result, expected)
                restored = original(*values, **options)
                exact(restored, expected)
                assert torch.equal(rng, torch.cuda.get_rng_state())
                assert versions == {name: parameter._version for name, parameter in model.named_parameters()}
                medians = {name: statistics.median(row[name] for row in samples) for name in ('total_ms','trunk_ms','spatial_softmax_ms','head_remaining_ms','entire_head_ms')}
                bound = 11*49*min(medians['trunk_ms']+medians['spatial_softmax_ms'],medians['head_remaining_ms'])/1000
                report = dict(stage=stage, samples=samples, medians=medians,
                    maximum_ideal_pipeline_saving_s=bound, ideal_bound_ignores_transfer_and_graph_restructure=True,
                    packed_feature_bytes=sum(row['output_bytes'] for row in softmax_shapes.values()),
                    softmax_shapes=dict(softmax_shapes), explicit_timing_event_nodes=len(events.records),
                    original209_sources_fixed=True, original_native_cache_owners_retained=True,
                    original_three_output_signedzero_bits=True, baseline_restoration_bits=True,
                    RNG_unchanged=True, master_versions_unchanged=True,
                    sampling_instrumented=True, no_official_score=True)
                reports[stage] = report
                (root/('head-pipeline-'+stage+'.json')).write_text(json.dumps(report,indent=2)+'\n')
                print('HEAD_PIPELINE_STAGE '+json.dumps(dict(stage=stage,medians=medians,ideal_saving_s=bound,feature_bytes=report['packed_feature_bytes'])),flush=True)
            finally:
                active[0] = None
                close_graph(wrapper)
                diagnostic[0] = None
                events.close()
            if len(reports) == 2:
                summary = dict(stages=reports, no_official_score=True, original_three_output_bits_checked=True,
                    maximum_combined_ideal_saving_s=sum(row['maximum_ideal_pipeline_saving_s'] for row in reports.values()))
                (root/'head-pipeline-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('HEAD_PIPELINE_209 passed=true stages=2',flush=True)
                raise Finished()
            return expected
        model.apply_model = apply

    runtime.configure_denoiser = configure
    script = root/'scripts/evaluation/world_model_interaction.py'
    sys.path.insert(0,str(script.parent))
    sys.argv = [str(script),'--seed','123','--ckpt_path','/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',
        '--config','configs/inference/world_model_interaction.yaml','--dataset','unitree_z1_dual_arm_stackbox_v2',
        '--savedir',str(root/'unused-head-profile-output'),'--bs','1','--height','320','--width','512',
        '--unconditional_guidance_scale','1.0','--ddim_steps','50','--ddim_eta','1.0','--video_length','16',
        '--frame_stride','4','--exe_steps','16','--n_iter','1','--timestep_spacing','uniform_trailing',
        '--guidance_rescale','.7','--perframe_ae','--input_cache_dir',
        '/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs','--require_input_cache']
    try:
        try:
            runpy.run_path(str(script),run_name='__main__')
        except Finished:
            pass
        else:
            raise AssertionError('Both stages were not measured')
    finally:
        if diagnostic[0] is not None:
            close_graph(diagnostic[0])
        for wrapper in originals:
            close_graph(wrapper)
        for module, previous in reversed(saved):
            module.forward = previous
        runtime.configure_denoiser = configured


if __name__ == '__main__':
    main()
