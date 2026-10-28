# Accepted initial FFN220 timed entry

The adopted default is initial-FFN220 at277.436969s. This implementation
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

CPU lifecycle and row/storage-routing checks passed without CUDA. The complete
GPU0/3, same RAM output comparison was278.896881s for219 and277.436969s for220,
a net1.459912s saving. Model-r2 median savings3.940857/3.897850ms include gather;
do not add ranks or treat those as process timing. World CUDA generation saved
1.965497s, while model transfer grew0.558046s. There is one full process AB pair;
six alternating pairs refer to complete model forwards.

PSNR remains29.675324786dB; all23 videos equal adopted219 and the other rank.
Fresh official49round/103video validation passed; five-case sum826.342893s,
assembled from independent complete cases. The default parallel launcher now
selects220. It creates a private new-variant cache under executable ~/.tmp;
INIT_FFN_CACHE_DIR may point to an existing verified private warm cache. Cold
compilation remains inside the timer. Never share rank cache directories with
another simultaneous new-variant compilation.

See [complete paired evidence](../../../results/aggressive-20261028/initial-ffn-219/full-candidate-r1)
and [model evidence](../model-r2). The historical219 entry remains available.
