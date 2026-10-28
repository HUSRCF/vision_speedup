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
    from collections import Counter
    from geglu_c1280_subtile32_tiles_hooks import installed
    hooks=installed();launches=hooks.__enter__()
    configure_original = runtime.configure_denoiser

    def configure(model, *args, **kwargs):
        configure_original(model, *args, **kwargs)
        original = model.apply_model
        assert isinstance(original, DenoiserGraph)

        def diagnose(*values, **keywords):
            if torch.backends.cudnn.allow_tf32 != (options.profile_stage == 'decision'):
                return original(*values, **keywords)
            previous = os.environ.get('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES')
            graphs = []
            try:
                results, entries, dispatch = {}, {}, {}
                for label, mode in [('baseline','0'),('candidate','1')]:
                    os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES'] = mode
                    graph = DenoiserGraph(original.original,
                        condition_refresher=original.condition_refresher,
                        skip_unchanged_inputs=False)
                    graphs.append(graph)
                    before=len(launches)
                    results[label] = graph(*values, **keywords)
                    dispatch[label]=launches[before:]
                    if options.profile_stage=='world':
                        expected=Counter({(2560,5120,1280,label=='candidate'):40,(640,5120,1280,label=='candidate'):8})
                        assert Counter((r['M'],r['C'],r['K'],r['tuned_tiles']) for r in dispatch[label])==expected
                        assert all(r['subtile32'] and r['stages']==2 for r in dispatch[label])
                    else:assert not dispatch[label]
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
                        entry['graph'].replay()
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
                            entries[label]['graph'].replay()
                            end.record()
                            torch.cuda.synchronize()
                            times.append(start.elapsed_time(end))
                        sample[label] = statistics.median(times)
                    sample['saved_ms'] = sample['baseline'] - sample['candidate']
                    sample['order'] = order
                    pairs.append(sample)
                summary = dict(stage=options.profile_stage, pairs=pairs,dispatch=dispatch,
                    median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
                    same_process=True, same_gpu=True, same_model_weights=True,
                    same_actual_first_denoiser_input=True, exact_output_tensors=tensor_count,
                    output_signed_zero_bitwise=True, full_video_quality='not_tested_here',
                    alternating_order=True, no_external_event_nodes=True, no_official_score=True)
                (options.diagnostic_dir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
                print('GEGLU_C1280_SUBTILE32_TILES_ACTUAL_GRAPH_PAIR '+json.dumps(summary),flush=True)
                print('GEGLU_C1280_SUBTILE32_TILES_ACTUAL_GRAPH_PAIR_CHECK passed=true same_process=true '
                      'same_gpu=true same_actual_input=true output_bitwise=true '
                      'alternating_pairs=6 no_official_score=true',flush=True)
            finally:
                if previous is None:
                    os.environ.pop('UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES',None)
                else:
                    os.environ['UNIFOLM_WORLD_GEGLU_C1280_SUBTILE32_TILES'] = previous
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
        hooks.__exit__(None,None,None)


if __name__ == '__main__':
    main()
