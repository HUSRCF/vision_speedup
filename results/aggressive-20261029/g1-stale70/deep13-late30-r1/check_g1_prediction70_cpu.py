"""Check actual13-parent stale reuse against an independent module interpreter."""
import copy
import torch
from types import MethodType, SimpleNamespace
from unifolm_wma.utils.cuda_graph import DenoiserGraph
from unifolm_wma.utils.crossstep_feature_runtime import Runtime
from unifolm_wma.utils.crossstep_feature_reuse import Controller, predict


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.75))
        self.position = 0
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.tanh(x*self.weight+.03+self.position/128)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_blocks = torch.nn.ModuleList([Block() for _ in range(12)])
        self.middle_block = Block()
        self.output_blocks = torch.nn.ModuleList([Block() for _ in range(12)])
        self.head = Block()

    def modules_in_order(self):
        return [*self.input_blocks, self.middle_block, *self.output_blocks, self.head]

    def forward(self, x):
        for block in self.modules_in_order():
            x = block(x)
        return x


class Executor:
    def __init__(self, model):
        def original(owner, *args, **kwargs):
            return owner(args[0])
        self.denoiser_graph = DenoiserGraph(MethodType(original, model))
        self.calls = 0
        self.reuse_calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.reuse_calls += int(kwargs['_crossstep_feature_reuse'])
        return self.denoiser_graph(*args, **kwargs)


@torch.no_grad()
def main():
    assert not torch.cuda.is_initialized()
    model = Model().eval()
    reference = copy.deepcopy(model)
    executor = Executor(model)
    outer = SimpleNamespace(model=SimpleNamespace(diffusion_model=model), apply_model=executor)
    runtime = Runtime(outer, input_start=6, output_end=6, start=30, stop=40,
                      stages=('decision', 'world'))
    assert len(runtime.controller.saved) == 13
    selected = set(range(6, 19))
    rng = torch.get_rng_state().clone()
    for generation in range(11):
        for stage in ('decision', 'world'):
            history = {}
            for position in range(50):
                for block in model.modules_in_order()+reference.modules_in_order():
                    block.position = position
                x = torch.full((1, 257), generation/16+(stage == 'world')/8+position/64)
                reuse = 30 <= position < 40 and position % 2 == 1
                expected = x
                for index, block in enumerate(reference.modules_in_order()):
                    if reuse and index in selected:
                        expected = history[index].clone()
                    else:
                        expected = block(expected)
                        if index in selected:
                            history[index] = expected.clone()
                runtime.set_step(stage, position)
                actual = outer.apply_model(x, None, None, torch.tensor([49-position]),
                    dict(stage=stage, generation=generation), fs=None)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert float(runtime.controller.alpha) == 0.
                runtime.finish_step()
    report = runtime.report()
    assert report['counts'] == dict(decision_calls=550, decision_refresh=495, decision_reuse=55,
                                    world_calls=550, world_refresh=495, world_reuse=55)
    assert report['alpha_histogram'] == {'0.0': 110}
    assert report['stale_refresh_reuse'] and report['history_complete_refreshes'] == 2
    assert executor.calls == 1100 and executor.reuse_calls == 110
    assert model.input_blocks[0].calls == model.output_blocks[-1].calls == model.head.calls == 1100
    assert model.middle_block.calls == 990
    assert len(report['completed_generations']) == 22
    assert all(g['steps'] == 50 for g in report['completed_generations'])
    runtime.controller.close()
    assert torch.equal(rng, torch.get_rng_state())
    # Keep two-real-refresh invalidation semantics and reject skipped steps.
    for changed in ('master', 'condition'):
        model = Model().eval()
        controller = Controller(model, input_start=6, output_end=6, start=30, stop=40)
        for position in range(36):
            if position == 32 and changed == 'master':
                model.middle_block.weight.add_(.125)
            condition = 'new' if changed == 'condition' and position >= 32 else 'old'
            reuse = controller.begin(stage='world', generation=0, position=position,
                total_steps=50, condition_key=condition, input_key=(1, 257))
            if position in (32, 33):
                assert not reuse
            if position == 35:
                assert reuse and float(controller.alpha) == 0.
            model(torch.full((1, 257), position/64))
            controller.end()
        controller.close()
    model = Model().eval()
    controller = Controller(model, input_start=6, output_end=6)
    controller.begin(stage='world', generation=0, position=0, total_steps=50,
                     condition_key=0, input_key=(1, 257))
    model(torch.ones(1, 257))
    controller.end()
    try:
        controller.begin(stage='world', generation=0, position=2, total_steps=50,
                         condition_key=0, input_key=(1, 257))
    except AssertionError:
        pass
    else:
        raise AssertionError('Skipped sampler step accepted')
    controller.close()
    for dtype in (torch.float16, torch.float32):
        x = torch.arange(257, dtype=torch.float32)/256
        for residual in (False, True):
            live = x.to(dtype) if residual else None
            expected = x+live.float() if residual else x
            torch.testing.assert_close(predict(x, x-.125, torch.tensor([0.]), live,
                output_dtype=dtype), expected.to(dtype), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    print('G1_STALE70_CPU passed=true actual13_parents=true all1100_outputs_match_independent_stale_interpreter=true full50_steps=true shallow_and_head_live=true two_refresh_invalidation=true skipped_step_rejected=true no_GPU=true')


if __name__ == '__main__':
    main()

from pathlib import Path
@torch.no_grad()
def check_cache_scope():
    # New kernels may use a custom manager only within their lexical scope.
    # Triton2.3 caches the class globally even when the env var is removed.
    import os,sys,tempfile
    import unifolm_wma.utils.crossstep_feature_reuse as cache_module
    sys.path.insert(0,str(Path(cache_module.__file__).resolve().parents[3]/'scripts/evaluation'))
    import triton.runtime.cache as cache
    from unifolm_wma.utils.crossstep_feature_reuse import kernel_cache_scope
    with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])as temporary:
        root=Path(temporary);(root/'rank0').mkdir();(root/'rank0/gpu').mkdir();(root/'rank0/exec').mkdir();(root/'ffn').mkdir()
        old=dict(os.environ);old_class=cache.__dict__['__cache_cls'];old_name=cache.__dict__['__cache_cls_nme']
        try:
            os.environ.update(TEAM07_PREDICT_CACHE=str(root),TEAM07_PARALLEL_RANK='0',NONV2_FFN_CACHE_DIR=str(root/'ffn'))
            os.environ.pop('TRITON_CACHE_MANAGER',None)
            before={key:os.environ.get(key)for key in('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','EXPERIMENT_EXEC_CACHE')}
            with kernel_cache_scope():
                obj=cache.get_cache_manager('team07-static-time-lexical-CPU-check')
                assert type(obj).__name__=='ReadWarmSplitCache'and root/'rank0/gpu'in Path(obj.cache_dir).parents
            assert {key:os.environ.get(key)for key in before}==before
            assert cache.__dict__['__cache_cls']is old_class and cache.__dict__['__cache_cls_nme']==old_name
            os.environ['TRITON_CACHE_DIR']=str(root/'ffn')
            obj=cache.get_cache_manager('team07-original-native-CPU-check');assert type(obj)is cache.FileCacheManager
        finally:
            os.environ.clear();os.environ.update(old);cache.__dict__['__cache_cls']=old_class;cache.__dict__['__cache_cls_nme']=old_name
    print('PREDICT_CACHE_SCOPE_CPU passed=true environment_and_Triton_class_restored=true original_native_cache_class_restored=true no_GPU=true')


if __name__=='__main__':check_cache_scope()
