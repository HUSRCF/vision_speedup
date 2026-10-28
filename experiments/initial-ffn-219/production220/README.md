# Experimental initial FFN220 timed entry

The adopted default remains native-attention219 at280.850431s. This candidate
adds one helper and extends runtime routing/coverage checks. All original209
files,110 JIT bodies and four reversible timing hooks remain unchanged.
The initial FFN controller, row routing and temporary cache context have ASTs
identical to the clean GPU-qualified model-r2 script. Only metadata reporting
and composite-mode glue differ. The runtime requires exactly one new world
FFN gather plus the existing spatial/temporal/native-attention capture counts.

Materialize with `python materialize.py --output /fresh/private/path`. Use the
included flags, normal authorized V2 resources and two GPUs. Provide
`INIT_FFN_CACHE_DIR` on an executable mount; only new M20480 variants use
perrank subdirectories there. Keep original warm caches and original CUBIN
gates. The shared native extension remains linked to its original executable
file when the source/output destination is noexec. No preloaded model weights
or outputs are moved outside the process timer. The official PSNR and result
writer and all workload arguments remain unchanged.

CPU lifecycle and row/storage-routing checks passed without CUDA. A matched
whole-process baseline/candidate pair is queued, using private RAM filesystem
outputs because home free space is very small. Full V2 and PSNR are pending.
Do not treat the model-r2 savings3.940857/3.897850ms as a measured process score.
See ../model-r2 for the qualified operation and dynamic-state evidence.
