# Adopted temporal221 full-workload version

Official default is221 at270.825199s /29.675324786dB, with a fresh103-video accepted aggregate of819.731123s. This221-source version adds one helper and changes runtime routing/coverage only. Original209 sources and110 JIT bodies remain unchanged. The temporal routing functions and Controller except report have identical ASTs to the six-state, six-pair GPU-qualified model-r2. See ../model-r2 for numerical checks and accepted-outcome.json for complete-run evidence.

Materialize: `python materialize.py --output /fresh/private/path`. Provide separate executable per-rank standard cache roots via INIT_FFN_CACHE_DIR and TEMPORAL_BLOCK_CACHE_DIR. Reuse qualified private warm caches if present; preserve original warm CUBIN gates. Sources/outputs may use noexec RAM while the existing native extension remains linked to its executable original. Full workload, helper startup, new compilation, captures, all copies/communication, both video outputs and graph teardown remain inside the evaluator process timer. Nothing preloads model weights or caches video results outside the timer.

The full candidate compares with the historical accepted220 process, not a contemporaneous full-process AB pair. Six alternating pairs were measured for complete world forwards on the same GPUs0,2. No extrapolated number is an official V2 result. Exact preparation flags, guard scripts, source hashes and timing proof are frozen here.


Full attempt r2 failed during video output with ENOSPC on shared RAM storage.
It has no valid full-process score or PSNR. Sources and timed assets matched the frozen snapshot. r3 completed inference using split private output storage. Its original validator rejected the symlink video root; the failed raw evidence is preserved. SHA-identical same-run rank1 files were then hardlinked into an ordinary directory and revalidated officially, without GPU rerun or timing change. See ../full-split-r3 and the accepted-outcome for the exact packaging proof. Proposed historical evidence compression was not executed.
