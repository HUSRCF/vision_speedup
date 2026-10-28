# NonV2 FP16 FFN row sharding — model evidence only

These probes use the exact59-source single-GPU nonV2 VAE baseline and official Z1 inputs/weights, not the V2 FP32 compensated kernels. The common nonV2 flags configure the full denoiser Half; `UNIFOLM_FFN_DTYPE=fp32` is a sentinel selecting the original `self.net(x)` forward, whose actual parameters and input are asserted Half at runtime.

All33 BasicTransformerBlock FFNs are split along their independent leading sample dimension. Spatial FFNs retain their complete spatial sequence per frame; temporal FFNs retain all16 time steps per spatial position. Original FFN projections, GEGLU rounding, master storage and forward are retained. Norm, attention, global action masks, RNG and sampler are unchanged. One raw-FFN output gather per block restores complete data before its original residual addition. This adds33 gathers; all communication is included in complete-forward timing. Actual native GEMM M is halved and its dispatch/rounding can change. It is not claimed bitwise equivalent.

Six dynamic states test original inputs, changed inputs, all132 FFN parameter edits, restored parameters, zeroed parameters and restoration again. The zero state demonstrably changes baseline output, preventing stale-weight false passes. Restored baseline output is bitwise exact and CPU/GPU RNG unchanged. Both ranks use cloned input trees broadcast from rank0; no tensor contents or addresses are exported.

Six alternating pairs with40 balanced warmups,2 settles and7 samples per condition all improve complete denoiser forward time. Decision GPUs0,1 median savings are3.876354/5.427204ms; world GPUs2,3 are5.751297/4.246525ms. Different stages used different pairs; do not add ranks or treat these as a process-level score.

The largest observed video output relative L2 is about0.0525%, maximum absolute error0.00390625. Action/state errors also occur. These satisfy only a deliberately broad diagnostic screen (`relative_l2<=.005 OR max_abs<=.002`), not the official video quality requirement. Full12-round/25-video Z1 and official PSNR have not run. Production/default and the official log/JSON are unchanged.

Candidate FFN JIT shape variants use a new ordinary per-rank executable private cache; original warm caches are preserved. Scripts validate the59-source manifest before and after execution. The small CPU preflight covers36 ownership/layout cases without initializing CUDA. Compact new-attempt logs/reports use ordinary hardlinks after completed writes; no historical evidence is removed or compressed.
