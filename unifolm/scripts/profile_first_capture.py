"""Attribute first decision/world capture in a fresh process; diagnostic only."""
import argparse
import os
from collections import Counter
import unifolm_wma.utils.inference_runtime as runtime
from unifolm_wma.utils import ffn1280_zero_residual as consumer
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import torch
from torch.utils._pytree import tree_flatten
from unifolm_wma.utils.cuda_graph import DenoiserGraph


class Complete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--diagnostic-dir', required=True, type=Path)
    options, inference_args = parser.parse_known_args()
    options.diagnostic_dir.mkdir(exist_ok=False, parents=True)
    original_capture = DenoiserGraph.capture
    measurements = []

    original_configure=runtime.configure_denoiser
    original_kernel=consumer._zero_low_gemm_residual
    launches=[];narrow_launches=[];wide={};narrow={}
    depth=int(os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2'));assert depth in {2,3}
    class ObservedKernel:
        def __getitem__(self,grid):
            launch=original_kernel[grid]
            def observed(*args,**kwargs):
                assert kwargs['num_warps']==4
                assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
                record=dict(rows=args[8],channels=args[9],inner=args[10],num_stages=kwargs['num_stages'])
                if args[10]==5120:
                    assert kwargs['num_stages']==depth
                    launches.append(record)
                else:
                    assert (args[8],args[9],args[10])==(40960,320,1280) and kwargs['num_stages']==3
                    narrow_launches.append(record)
                return launch(*args,**kwargs)
            return observed
    consumer._zero_low_gemm_residual=ObservedKernel()
    def configure(model,*args,**kwargs):
        original_configure(model,*args,**kwargs)
        wide.update((n,m) for n,m in model.model.diffusion_model.named_modules() if getattr(m,'_world_geglu_hidden_parts_c1280_zero_installed',False))
        narrow.update((n,m) for n,m in model.model.diffusion_model.named_modules() if getattr(m,'_world_ffn320_zero_installed',False))
        assert len(wide)==12 and len(narrow)==10
    runtime.configure_denoiser=configure
    def counts(modules,name):return {n:getattr(m,name,0) for n,m in modules.items()}
    def delta(modules,name,before):return {n:getattr(m,name,0)-before[n] for n,m in modules.items()}
    def capture(self, *args, **kwargs):
        stage = 'decision' if torch.backends.cudnn.allow_tf32 else 'world'
        launch_before=len(launches);narrow_launch_before=len(narrow_launches)
        beforew=counts(wide,'_world_geglu_hidden_parts_c1280_zero_calls')
        befored=counts(wide,'_decision_primary_hidden_parts_calls')
        beforen=counts(narrow,'_world_ffn320_zero_calls')
        original_forward = self.original
        forwards = []

        def forward(*values, **keywords):
            capturing = torch.cuda.is_current_stream_capturing()
            # Never insert events or synchronization into the captured graph.
            if not capturing:
                torch.cuda.synchronize()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
            before = time.perf_counter()
            result = original_forward(*values, **keywords)
            enqueue_wall = time.perf_counter() - before
            if not capturing:
                end.record()
                end.synchronize()
                cuda_ms = start.elapsed_time(end)
            else:
                cuda_ms = None
            forwards.append(dict(index=len(forwards), graph_capturing=capturing,
                                 enqueue_wall_s=enqueue_wall, cuda_elapsed_ms=cuda_ms))
            return result

        self.original = forward
        before = time.perf_counter()
        try:
            entry = original_capture(self, *args, **kwargs)
        finally:
            self.original = original_forward
        capture_wall = time.perf_counter() - before
        outputs, _ = tree_flatten(entry['outputs'])
        hashes = []
        for value in outputs:
            if isinstance(value, torch.Tensor):
                raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                hashes.append(dict(shape=list(value.shape), dtype=str(value.dtype),
                                   sha256=hashlib.sha256(raw).hexdigest()))
        replay_ms = []
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(12):
            start.record()
            entry['graph'].replay()
            end.record()
            end.synchronize()
            replay_ms.append(start.elapsed_time(end))
        current_launches=launches[launch_before:]
        current_narrow_launches=narrow_launches[narrow_launch_before:]
        if stage=='world':
            assert len(current_launches)==48
            shapes=Counter((r['rows'],r['channels'],r['inner']) for r in current_launches)
            assert shapes==Counter({(2560,1280,5120):40,(640,1280,5120):8})
            assert all(r['num_stages']==depth for r in current_launches)
            assert len(current_narrow_launches)==40
        else:assert not current_launches and not current_narrow_launches
        world=delta(wide,'_world_geglu_hidden_parts_c1280_zero_calls',beforew)
        decision=delta(wide,'_decision_primary_hidden_parts_calls',befored)
        narrow_dispatch=delta(narrow,'_world_ffn320_zero_calls',beforen)
        assert all(v==(4 if stage=='world' else 0) for v in [*world.values(),*narrow_dispatch.values()])
        assert all(v==(4 if stage=='decision' else 0) for v in decision.values())
        measurements.append(dict(FFN1280_launches=current_launches,retained_C320_FFN_launches=current_narrow_launches,world_C1280_dispatch=world,
                                 decision_C1280_dispatch=decision,world_C320_dispatch=narrow_dispatch,
                                 stage=stage, capture_wall_s=capture_wall,
                                 forwards=forwards, replay_ms=replay_ms, outputs=hashes))
        (options.diagnostic_dir/'capture.json').write_text(json.dumps(measurements, indent=2)+'\n')
        print('FIRST_CAPTURE_STAGE '+json.dumps(measurements[-1]), flush=True)
        if stage == 'world':
            assert [m['stage'] for m in measurements] == ['decision', 'world']
            print('FIRST_CAPTURE_CHECK passed=true stages=2 diagnostic_only=true no_official_score=true', flush=True)
            raise Complete
        return entry

    DenoiserGraph.capture = capture
    script = Path(__file__).parent/'evaluation/world_model_interaction.py'
    sys.path.insert(0, str(script.parent.resolve()))
    sys.argv = [str(script), *inference_args]
    try:
        runpy.run_path(str(script), run_name='__main__')
    except Complete:
        pass
    finally:
        DenoiserGraph.capture = original_capture
        runtime.configure_denoiser=original_configure
        consumer._zero_low_gemm_residual=original_kernel


if __name__ == '__main__':
    main()
