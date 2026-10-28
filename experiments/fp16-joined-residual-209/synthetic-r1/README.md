# Joined FP16 primary/residual: synthetic screen

This is a numerical approximation under test, not an adopted production kernel.
For each logical K64 block it concatenates the primary and correction products
into one physical K128 Half dot, with FP32 outer accumulation. Correction input
and weight use binary shifts 5/6 or 6/5, whose product is 1/2048. This retains
the two products' arithmetic volume and changes the inner FP32 summation order.
Weights with nonzero low parts use the original three-product kernel.

The first three-shape screen compares BM32/BM64, 12 configurations, eight dynamic
states each, output canaries, RNG/restoration, captured graph replay and sampled
full-K FP64 oracles. Six alternating timing pairs include the candidate's
in-kernel scaling. Synthetic precision gates do not establish video PSNR.
The [original209 source](../../../unifolm) is imported without changes. The
reference is freshly compiled in the isolated diagnostic cache: original
production CUBIN identity is not claimed for this synthetic test.

The clean GPU2 attempt completed all96 dynamic checks. All12 configurations
passed the local precision screen, but none was faster in all six timing pairs.
For the two K5120 shapes, initial residual-free relative L2 was about3.35e-7;
candidate speedup ratios were0.35–0.54 (slower). Six earlier foreign-context
attempts are excluded in full; see guard-report.json for the accepted attempt.
This candidate is rejected for performance. There is no model/video quality
result. See [full evidence](../../../results/aggressive-20261028/fp16-joined-residual-209/synthetic-r1).
Current official V2 remains287.745486 seconds; five-case total836.651410.

On a contest host put these files in a fresh private scratch directory, use
original209 on PYTHONPATH, and set CUDA_VISIBLE_DEVICES and MONITORED_GPU to the
same idle physical GPU. The parent guard must acquire both locks and recheck
strict idleness before allocating. For noexec scratch, configure the included
split cache manager with a fresh executable home cache for launcher shared
objects. Run guard_fp16_joined209.py --gpu N --entry audit_fp16_joined_residual209.py.
Do not interpret partial progress from excluded attempts as evidence.
