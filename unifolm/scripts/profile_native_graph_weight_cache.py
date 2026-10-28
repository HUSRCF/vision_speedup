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
            previous = os.environ.get('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE')
            graphs = []
            try:
                results, entries = {}, {}
                for label, mode in [('baseline','0'),('candidate','1')]:
                    os.environ['UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE'] = mode
                    graph = DenoiserGraph(original.original,
                        condition_refresher=original.condition_refresher,
                        skip_unchanged_inputs=False)
                    graphs.append(graph)
                    results[label] = graph(*values, **keywords)
                    assert len(graph.entries) == 1
                    entries[label] = next(iter(graph.entries.values()))
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
                for _ in range(3):
                    for entry in entries.values():
                        (entry.get('native_weight_cache') or entry['graph']).replay()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                pairs = []
                for index in range(6):
                    order = ['baseline','candidate'] if index % 2 == 0 else ['candidate','baseline']
                    sample = {}
                    for label in order:
                        times = []
                        for _ in range(3):
                            start.record()
                            for _ in range(10):
                                (entries[label].get('native_weight_cache') or entries[label]['graph']).replay()
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end)/10)
                        sample[label] = statistics.median(times)
                    sample['saved_ms'] = sample['baseline'] - sample['candidate']
                    sample['order'] = order
                    pairs.append(sample)
                reports = {label:entry['native_weight_cache'].report if entry.get('native_weight_cache') else None
                           for label,entry in entries.items()}
                assert reports['baseline'] is None
                assert (reports['candidate'] is not None) == (options.profile_stage=='decision')
                changed = tuple(v+.0001 if isinstance(v,torch.Tensor) and v.is_floating_point() else v for v in values)
                def exact_trees(left,right):
                    aa,sa=tree_flatten(left);bb,sb=tree_flatten(right);assert sa==sb
                    for a,b in zip(aa,bb):
                        if isinstance(a,torch.Tensor):
                            assert a.dtype==b.dtype and a.shape==b.shape
                            dtype=torch.int32 if a.dtype==torch.float32 else torch.int16 if a.dtype==torch.float16 else None
                            assert torch.equal(a.contiguous().view(dtype),b.contiguous().view(dtype)) if dtype else torch.equal(a,b)
                        else:assert a==b
                exact_trees(graphs[0](*changed,**keywords),graphs[1](*changed,**keywords))
                manager=entries['candidate'].get('native_weight_cache')
                if manager is not None:
                    masters={c.weight.data_ptr():c.weight for _,c in manager.caches}
                    saved={ptr:w.clone() for ptr,w in masters.items()}
                    for _,c in manager.caches:c.cache.fill_(float('nan'))
                    for w in masters.values():w.add_(.00013)
                    exact_trees(graphs[0](*changed,**keywords),graphs[1](*changed,**keywords))
                    for ptr,w in masters.items():w.copy_(saved[ptr])
                    exact_trees(graphs[0](*values,**keywords),graphs[1](*values,**keywords))
                    for _,c in manager.caches:c.verify_canaries()
                    del saved
                summary = dict(stage=options.profile_stage, pairs=pairs,
                    cache_reports=reports, actual_host_inputs_and_poisoned_master_refresh_bits=True,
                    guard_canaries_intact=True, median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
                    same_process=True, same_gpu=True, same_model_weights=True,
                    same_actual_first_denoiser_input=True, exact_output_tensors=tensor_count,
                    output_signed_zero_bitwise=True, full_video_quality='not_tested_here',
                    alternating_order=True, no_external_event_nodes=True, no_official_score=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('NATIVE_WEIGHT_CACHE_ACTUAL_GRAPH_PAIR '+json.dumps(summary),flush=True)
                print('NATIVE_WEIGHT_CACHE_ACTUAL_GRAPH_PAIR_CHECK passed=true same_process=true '
                      'same_gpu=true same_actual_input=true output_bitwise=true '
                      'alternating_pairs=6 no_official_score=true',flush=True)
            finally:
                if previous is None:
                    os.environ.pop('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE',None)
                else:
                    os.environ['UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE'] = previous
                torch.cuda.synchronize()
                for wrapper in [original,*graphs]:
                    for entry in wrapper.entries.values():
                        if entry.get('native_weight_cache') is not None:
                            entry['native_weight_cache'].close()
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
