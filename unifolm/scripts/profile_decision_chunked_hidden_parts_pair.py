"""Compare two complete graphs of the same model/input with norm prefill on/off."""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import torch
import torch.utils.checkpoint
from torch.utils._pytree import tree_flatten
import unifolm_wma.utils.inference_runtime as runtime
from unifolm_wma.utils.cuda_graph import DenoiserGraph


class DiagnosticComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--profile-stage', choices=['world','decision'], required=True)
    options, inference_args = parser.parse_known_args()
    options.diagnostic_dir.mkdir(parents=True, exist_ok=False)
    configure_original = runtime.configure_denoiser

    def configure(model, *args, **kwargs):
        configure_original(model, *args, **kwargs)
        original = model.apply_model
        assert isinstance(original, DenoiserGraph)

        def diagnose(*values, **keywords):
            if torch.backends.cudnn.allow_tf32 != (options.profile_stage == 'decision'):
                return original(*values, **keywords)
            previous = os.environ.get('UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS')
            graphs = []
            from unifolm_wma.utils import decision_chunked_hidden_parts as fusion_helper
            original_fused=fusion_helper.chunked_hidden_residual
            fused_calls={}
            def counted(value,*args,**kwargs):
                width=str(args[1].shape[-1]);fused_calls[width]=fused_calls.get(width,0)+1
                return original_fused(value,*args,**kwargs)
            fusion_helper.chunked_hidden_residual=counted
            try:
                results, entries = {}, {}
                shared_before=None
                for label, mode in [('baseline','0'),('candidate','1')]:
                    os.environ['UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS'] = mode
                    if mode!='0':
                        if options.profile_stage=='world':
                            shared_before={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr(),tuple(t.data_ptr() for t in m.ff.net[2]._ffn640_residual_weight_cache[1]))
                                for name,m in model.model.diffusion_model.named_modules()
                                if getattr(m,'_world_geglu_hidden_parts_installed',False)}
                            assert len(shared_before)==10
                        assert fusion_helper.install_decision_chunked_hidden_parts(model.model.diffusion_model)==10
                    graph = DenoiserGraph(original.original,
                        condition_refresher=original.condition_refresher,
                        skip_unchanged_inputs=False)
                    graphs.append(graph)
                    results[label] = graph(*values, **keywords)
                    assert len(graph.entries) == 1
                    entries[label] = next(iter(graph.entries.values()))
                shared_after={name:(tuple(t.data_ptr() for t in m.ff.net[0]._norm3_geglu_weight_cache[1]),m.ff.net[0]._norm3_geglu_weight_cache[2].data_ptr(),tuple(t.data_ptr() for t in m.ff.net[2]._ffn640_residual_weight_cache[1]))
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_chunked_hidden_parts_installed',False)}
                assert len(shared_after)==10
                if shared_before is not None:assert shared_before==shared_after
                reference, spec = tree_flatten(results['baseline'])
                actual, actual_spec = tree_flatten(results['candidate'])
                assert spec == actual_spec
                tensor_count = 0
                for left, right in zip(reference, actual):
                    if isinstance(left, torch.Tensor):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                        assert left.dtype == right.dtype and left.shape == right.shape
                        if left.dtype in (torch.float32, torch.float16):
                            bits = torch.int32 if left.dtype == torch.float32 else torch.int16
                            assert torch.equal(left.contiguous().view(bits), right.contiguous().view(bits))
                        tensor_count += 1
                    else:
                        assert left == right
                assert tensor_count > 0
                module_dispatch={name:getattr(m,'_world_norm3_packed_geglu_calls',0)
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_world_norm3_packed_geglu_installed',False)}
                assert len(module_dispatch)==32
                tail_dispatch={name:getattr(m,'_decision_chunked_hidden_parts_calls',0)
                    for name,m in model.model.diffusion_model.named_modules()
                    if getattr(m,'_decision_chunked_hidden_parts_installed',False)}
                assert len(tail_dispatch)==10
                if options.profile_stage=='decision':
                    assert all(n>=3 for n in tail_dispatch.values()),tail_dispatch
                else:
                    assert not any(tail_dispatch.values()),tail_dispatch
                if options.profile_stage=='world':
                    assert all(n>=3 for n in module_dispatch.values()),module_dispatch
                else:
                    assert not any(module_dispatch.values()),module_dispatch
                if options.profile_stage=='decision':
                    assert all(fused_calls.get(str(width),0)>=30 for width in [640]),fused_calls
                else:
                    assert sum(fused_calls.values())==0,fused_calls
                for _ in range(3):
                    for entry in entries.values():
                        entry['graph'].replay()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                pairs = []
                for index in range(4):
                    order = ['baseline','candidate'] if index % 2 == 0 else ['candidate','baseline']
                    sample = {}
                    for label in order:
                        times = []
                        for _ in range(3):
                            start.record()
                            entries[label]['graph'].replay()
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end))
                        sample[label] = statistics.median(times)
                    sample['saved_ms'] = sample['baseline'] - sample['candidate']
                    sample['order'] = order
                    pairs.append(sample)
                summary = dict(stage=options.profile_stage, pairs=pairs,
                    median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
                    shared_GE_and_final_parts_unchanged_in_world=(shared_before==shared_after if shared_before is not None else None),
                    shared_cache_count=len(shared_after), fused_hidden_pipeline_calls_per_width=dict(fused_calls), hidden_parts_dispatch_per_module=tail_dispatch, norm3_dispatch_per_module=module_dispatch, same_process=True, same_gpu=True, same_model_weights=True,
                    same_actual_first_denoiser_input=True, exact_output_tensors=tensor_count,
                    output_signed_zero_bitwise=True, full_video_quality='not_tested_here',
                    alternating_order=True, no_external_event_nodes=True, no_official_score=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('DECISION_CHUNKED_HIDDEN_PARTS_MODEL_ACTUAL_GRAPH_PAIR '+json.dumps(summary),flush=True)
                print('DECISION_CHUNKED_HIDDEN_PARTS_MODEL_ACTUAL_GRAPH_PAIR_CHECK passed=true same_process=true '
                      'same_gpu=true same_actual_input=true output_bitwise=true '
                      'alternating_pairs=4 no_official_score=true',flush=True)
            finally:
                fusion_helper.chunked_hidden_residual=original_fused
                if previous is None:
                    os.environ.pop('UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS',None)
                else:
                    os.environ['UNIFOLM_DECISION_CHUNKED_HIDDEN_PARTS'] = previous
                torch.cuda.synchronize()
                for wrapper in [original,*graphs]:
                    for entry in wrapper.entries.values():
                        entry['graph'].reset()
                    wrapper.entries.clear()
            raise DiagnosticComplete

        model.apply_model = diagnose

    runtime.configure_denoiser = configure
    script = Path(__file__).parent/'evaluation/world_model_interaction.py'
    sys.path.insert(0,str(script.parent.resolve()))
    sys.argv = [str(script),*inference_args]
    try:
        runpy.run_path(str(script),run_name='__main__')
    except DiagnosticComplete:
        pass
    finally:
        runtime.configure_denoiser = configure_original


if __name__ == '__main__':
    main()
