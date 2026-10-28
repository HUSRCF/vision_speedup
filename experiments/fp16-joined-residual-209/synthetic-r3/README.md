# Chained Half correction: clean synthetic result

Instead of physical K128 concatenation, this revision directly uses the primary
FP32 dot result as the accumulator operand of the scaled Half correction dot.
Per-logical-K64 outer FP32 accumulation, bias/residual order and the general
nonzero-WL original fallback remain. The zero-WL summation order changes.

Four configurations per shape: BM32/BM64 with binary shifts5/6 or11/0; candidate
depth2 versus recorded original depth3 (odd boundary shape depth2 for both).
The shift11 route avoids rescaling WH but can quantize small correction inputs
into Half subnormals. No static or dynamic operand pack/cache was added.
The old physical-K128 kernel is retained in the file but not executed here.

A clean GPU2 run completed12 configurations and96 dynamic checks, including
nonzero-WL exact fallbacks, restoration/RNG/canaries, graph replay and sampled
full-K FP64 oracles. All local precision screens passed (about3.35e-7 maximum
residual-free relative L2 for the K5120 synthetic distributions), but none met
the six-pair speed gate. On M1280 the best ratio was0.834; on M640 it was0.997,
approximately tied and still not consistently faster. Rejected for performance.
No real-model/video result, production CUBIN identity or full-process gain is
claimed. See [all evidence](../../../results/aggressive-20261028/fp16-joined-residual-209/synthetic-r3).
