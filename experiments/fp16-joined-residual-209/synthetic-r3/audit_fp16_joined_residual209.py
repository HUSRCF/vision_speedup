"""Synthetic local precision/timing screen. No model quality claim or tensors exported."""
import json
import statistics
from pathlib import Path
import torch
import triton
from guard_fp16_joined209 import announce_worker
from fp16_joined_residual209_candidate import launch_fused
from unifolm_wma.utils.ffn1280_zero_residual import _zero_low_gemm_residual
from unifolm_wma.utils.triton_half_compensated_linear import pack_half_weight


def bits(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def error(a, b):
    delta = a.double() - b.double()
    return dict(relative_l2=float(torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(b.double()).clamp_min(1e-30)),
                max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(a).all()))


def graph(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    out = torch.cuda.CUDAGraph()
    with torch.cuda.graph(out): fn()
    return out


def measure(a, b):
    for _ in range(20): a.replay(); b.replay()
    torch.cuda.synchronize()
    pairs = []
    for pair in range(6):
        times = {}
        order = [('baseline', a), ('candidate', b)]
        for label, g in (order if pair % 2 == 0 else order[::-1]):
            samples = []
            for _ in range(7):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10): g.replay()
                end.record(); end.synchronize()
                samples.append(start.elapsed_time(end) / 10)
            times[label] = statistics.median(samples)
        pairs.append(dict(**times, saved_ms=times['baseline'] - times['candidate']))
    return dict(pairs=pairs, median_saved_ms=statistics.median(p['saved_ms'] for p in pairs),
                median_speedup=statistics.median(p['baseline'] / p['candidate'] for p in pairs),
                all6_pairs_faster=all(p['saved_ms'] > 0 for p in pairs))


@torch.inference_mode()
def screen(m, n, k, baseline_bm, stages):
    torch.manual_seed(12805120 + m)
    x = torch.randn(m, k, device='cuda') * 1.5
    # Trained weights here are represented by Half with zero low parts, but
    # these are fresh synthetic distributions, not captured model operands.
    w = (torch.randn(n, k, device='cuda') * .01).half().float()
    xh, xl = pack_half_weight(x, weight_scale=1.)
    wh, wl = pack_half_weight(w)
    count = torch.count_nonzero(wl)
    assert int(count) == 0
    bias = torch.randn(n, device='cuda') * .01
    residual = torch.randn(m, n, device='cuda') * .1
    baseline = torch.empty_like(residual)
    storage = torch.full((m * n + 512,), 12345., device='cuda')
    out = storage[256:-256].view(m, n)
    options = dict(num_warps=4, num_stages=stages, enable_fp_fusion=False)
    v = dict(XH=xh, XL=xl, WH=wh, WL=wl, LowCount=count, Bias=bias,
             Residual=residual, M=m, N=n, K=k, HAS_BIAS=True, _launch_options=options)
    saved = {name: v[name].clone() for name in ('XH', 'XL', 'WH', 'WL', 'LowCount')}
    def original():
        _zero_low_gemm_residual[(triton.cdiv(m, baseline_bm) * triton.cdiv(n, 64),)](
            xh, xl, wh, wl, count, bias, residual, baseline,
            m, n, k, True, baseline_bm, 32, 1, **options)
    original(); torch.cuda.synchronize()
    initial_reference = baseline.clone()
    base_graph = graph(original)
    ri = torch.linspace(0, m - 1, min(m, 16), device='cuda').long()
    ci = torch.linspace(0, n - 1, min(n, 8), device='cuda').long()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    rows = []
    for method, shift, bm in (('chained', 5, 32), ('chained', 5, 64),
                             ('chained', 11, 32), ('chained', 11, 64)):
            candidate_values = dict(v, _launch_options=dict(options, num_stages=2))
            def call(): launch_fused(candidate_values, out, shift=shift, bm=bm, method=method)
            cases = []
            def check(label, exact=False):
                out.fill_(float('nan'))
                original(); call(); torch.cuda.synchronize()
                e = error(out, baseline)
                denominator = baseline - residual - bias[None, :]
                free_error = float(torch.linalg.vector_norm((out - baseline).double()) /
                                   torch.linalg.vector_norm(denominator.double()).clamp_min(1e-30))
                xd = xh[ri].double() + xl[ri].double() / 2048.
                wd = (wh[ci].double() + wl[ci].double() / 2048.) / 256.
                oracle = xd @ wd.T + residual[ri[:, None], ci[None, :]].double() + bias[ci].double()[None, :]
                canaries = bool((storage[:256] == 12345.).all() and (storage[-256:] == 12345.).all())
                same = bits(out, baseline)
                assert e['finite'] and canaries and (not exact or same)
                cases.append(dict(name=label, **e, residual_free_relative_l2=free_error,
                    baseline_fp64_sample=error(baseline[ri[:, None], ci[None, :]], oracle),
                    candidate_fp64_sample=error(out[ri[:, None], ci[None, :]], oracle),
                    exact_required=exact, signedzero_bits=same, output_canaries=canaries))
            check('initial')
            first = out.clone()
            xl.mul_(.75); check('changed_input_low')
            assert not bits(baseline, initial_reference), 'Missing low-part input sensitivity'
            xl.copy_(saved['XL'])
            xh.add_(.125); check('changed_input_high')
            xh.copy_(saved['XH'])
            wl.fill_(.125); count.fill_(wl.numel()); check('general_weight_low_fallback', exact=True)
            wh.mul_(.5); check('changed_weight_fallback', exact=True)
            wh.copy_(saved['WH']); wl.zero_(); count.zero_()
            xh.fill_(1.); xl.fill_(2. ** -12); check('near_constant')
            xh.zero_(); xl.zero_(); check('zero_inputs', exact=True)
            for name, value in saved.items(): v[name].copy_(value)
            check('restored')
            assert bits(baseline, initial_reference) and bits(out, first), 'Restoration failure'
            assert torch.equal(torch.get_rng_state(), cpu_rng)
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
            candidate_graph = graph(call)
            timing = measure(base_graph, candidate_graph)
            candidate_graph.replay(); torch.cuda.synchronize()
            assert bits(out, first), 'Captured/replayed arithmetic differs'
            precision = all(c['residual_free_relative_l2'] <= 1e-6 and
                c['candidate_fp64_sample']['relative_l2'] <= 1e-5 for c in cases)
            row = dict(shape=[m, n, k], baseline_BM=baseline_bm, candidate_BM=bm,
                       method=method, baseline_stages=stages, stages=2,
                       A_SHIFT=shift, B_SHIFT=11-shift, cases=cases,
                       precision_screen_passed=precision, RNG_unchanged=True,
                       dynamic_restoration_signedzero_bits=True, timing=timing,
                       eligible_for_real_operand_screen=precision and timing['all6_pairs_faster'])
            rows.append(row)
            Path('progress.json').write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
            print('JOINED_SCREEN ' + json.dumps({key: row[key] for key in
                ('shape', 'method', 'candidate_BM', 'A_SHIFT', 'precision_screen_passed', 'timing')}), flush=True)
            del candidate_graph, first
    return rows


def main():
    hold = announce_worker()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for geometry in ((1280, 1280, 5120, 64, 3), (640, 1280, 5120, 32, 3), (67, 79, 127, 32, 2)):
        rows.extend(screen(*geometry))
    report = dict(diagnostic_complete=True, all_dynamic_gates_passed=True,
                  all_precision_screens_passed=all(r['precision_screen_passed'] for r in rows),
                  synthetic_only=True, model_or_video_quality_verified=False,
                  output_cache=False, production_sources_modified=False,
                  original_production_CUBIN_equivalence_claimed=False,
                  logical_K=64, chained_physical_K=64,
                  FP32_outer_accumulation=True,
                  pairs=6, rows=rows, dynamic_checks=sum(len(r['cases']) for r in rows))
    Path('comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FP16_JOINED_SCREEN complete=true', flush=True)


if __name__ == '__main__': main()
