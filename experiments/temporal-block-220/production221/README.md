# Pending temporal221 full-workload candidate

Official default remains adopted220 at277.436969s until this candidate finishes complete V2 and official video validation. This221-source candidate adds one helper and changes runtime routing/coverage only. Original209 sources and110 JIT bodies remain unchanged. The temporal routing functions and Controller except report have identical ASTs to the six-state, six-pair GPU-qualified model-r2. See ../model-r2 for scope and numerical checks.

Materialize: `python materialize.py --output /fresh/private/path`. Provide separate executable per-rank standard cache roots via INIT_FFN_CACHE_DIR and TEMPORAL_BLOCK_CACHE_DIR. Reuse qualified private warm caches if present; preserve original warm CUBIN gates. Sources/outputs may use noexec RAM while the existing native extension remains linked to its executable original. Full workload, helper startup, new compilation, captures, all copies/communication, both video outputs and graph teardown remain inside the evaluator process timer. Nothing preloads model weights or caches video results outside the timer.

The full candidate compares with the historical accepted220 process, not a contemporaneous full-process AB pair. Six alternating pairs were measured for complete world forwards on the same GPUs0,2. No extrapolated number is an official V2 result. Exact preparation flags, guard scripts, source hashes and timing proof are frozen here.


Full attempt r2 failed during video output with ENOSPC on shared RAM storage.
It has no valid full-process score or PSNR and does not replace220. Sources and
timed assets matched the frozen snapshot. Retry awaits sufficient private
output storage; proposed historical evidence compression has not been run
because automatic review requires explicit user approval. See the full-failed-r2
summary for bounded failure evidence and telemetry.
