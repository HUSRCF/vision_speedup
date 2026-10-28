"""Checkpoint-backed decode timing, never a complete inference/quality score."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils import nonv2_nonfeedback_vae as baseline


def load_candidate():
    spec = importlib.util.spec_from_file_location('all_feedback_candidate', Path(__file__).with_name('candidate.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    candidate = load_candidate()
    os.environ['UNIFOLM_CASE'] = 'unitree_z1_stackbox'
    torch.manual_seed(1568)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tf32 = os.environ.get('VAE_MICRO_TF32', '0') == '1'
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    cfg = OmegaConf.load('configs/inference/world_model_interaction.yaml').model.params
    vae = instantiate_from_config(cfg.first_stage_config).eval()
    raw = torch.load('/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt', map_location='cpu', mmap=True)
    state = raw.get('state_dict', raw)
    prefix = 'first_stage_model.'
    vae.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    del raw, state
    vae.cuda()
    model = SimpleNamespace(encoder_type='2d', perframe_ae=True, n_obs_steps_imagen=2,
                            scale_factor=cfg.scale_factor, first_stage_model=vae)
    # Use the same qualified owner for both timings: there is one persistent
    # Half shadow in production, and cache construction is already paid there.
    baseline.owners[id(vae)] = candidate.owners[id(vae)] = baseline.Decoder(vae)
    owner = baseline.owners[id(vae)]
    video = torch.randn(1, 4, 16, 40, 64, device='cuda') * .2
    saved_input = video.clone()
    saved_weights = [p.clone() for p in vae.parameters()]
    pointers = [p.data_ptr() for p in vae.parameters()]
    versions = [p._version for p in vae.parameters()]
    keys = list(vae.state_dict())
    rng = torch.cuda.get_rng_state().clone()

    def call(name, value=video):
        os.environ['UNIFOLM_NONV2_VAE_SCOPE'] = 'all-display' if name == 'baseline' else 'all-feedback'
        module = baseline if name == 'baseline' else candidate
        result = module.decode(model, value, stage='simulation', execution_steps=16)
        assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == (tf32, tf32)
        return result

    checks = []
    for label, value in [('initial', video), ('changed', video + .00021), ('zero', torch.zeros_like(video)), ('restored', video)]:
        expected = call('baseline', value)
        actual = call('candidate', value)
        assert actual.shape == expected.shape == (1, 3, 16, 320, 512)
        assert actual.dtype == expected.dtype == torch.float32
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        retained = actual.clone()
        other = call('candidate', value + .00017)
        assert other.data_ptr() != actual.data_ptr()
        assert torch.equal(actual.view(torch.int32), retained.view(torch.int32))
        diff = actual - expected
        checks.append(dict(case=label, max_abs=float(diff.abs().max()), mse=float(diff.square().mean()),
            feedback_max_abs=float(diff[:, :, 14:16].abs().max()), independent_outputs=True))
        del actual, expected, other, retained, diff
    pairs = []
    for name in ('baseline', 'candidate', 'candidate', 'baseline'):
        out = call(name)
        del out
    for i in range(6):
        pair = {}
        for name in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
            samples = []
            for _ in range(3):
                torch.cuda.synchronize()
                started = time.perf_counter()
                out = call(name)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
                del out
            pair[name + '_samples_ms'] = samples
            pair[name + '_ms'] = statistics.median(samples)
        pair['saved_ms'] = pair['baseline_ms'] - pair['candidate_ms']
        pairs.append(pair)
    assert torch.equal(video.view(torch.int32), saved_input.view(torch.int32))
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert pointers == [p.data_ptr() for p in vae.parameters()]
    assert versions == [p._version for p in vae.parameters()]
    assert keys == list(vae.state_dict())
    assert all(torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(vae.parameters(), saved_weights))
    result = dict(passed=True, checks=checks, pairs=pairs, median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
        all6_pairs_positive=all(p['saved_ms'] > 0 for p in pairs), owner_builds=owner.builds,
        FP32_masters_input_RNG_flags_unchanged=True, shared_qualified_half_owner=True,
        checkpoint_backed=True, synthetic_latents=True, cudnn_tf32=tf32, matmul_tf32=tf32, full_closed_loop_quality_unverified=True,
        numerical_candidate=True, no_official_score=True, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    Path(os.environ['EXPERIMENT_REPORT']).write_text(json.dumps(result, indent=2) + '\n')
    print('ALL_FEEDBACK_VAE_MICRO passed=true', flush=True)


if __name__ == '__main__':
    main()
