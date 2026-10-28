# Whole-world integration r3: separate new-kernel caches by rank

Same21 selected SGEMMs/12 ResBlocks and candidate math as model-r1/r2. New JIT
cache is PREFIX_CACHE_DIR/rank-0 or rank-1 only during new-kernel compile/capture,
then the original cache path is restored. No global custom cache-manager class,
compiler patch or CUBIN-gate relaxation. Every captured ABI witness records its
rank-local cache policy. The [CPU race fixture](../cache-temp-race-cpu-r1) explains
why per-rank directories avoid temporary binary files from the other writer.
The [startup classifier](../startup-guard-r2) is retained.

CPU preflight passed again, including120 block-boundary cases and21 owner
identities/12 selected blocks. The bounded GPU2/3 task is pending; no complete
model numerical/timing or full V2 result yet. All original23+45 owner/old-CUBIN
checks still precede new replacements, and all six dynamic model states, three
FP32 signedzero outputs, RNG and original restoration remain mandatory.

Follow model-r1 staging instructions using a fresh private root/cache; do not
import the CPU-only compiler helper in GPU workers.
[CPU evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r3-preflight).
Official V2 remains287.745486 seconds, five-case total836.651410.
