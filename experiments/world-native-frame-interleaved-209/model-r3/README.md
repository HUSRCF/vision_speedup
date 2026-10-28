# Whole-world integration r3: separate new-kernel caches by rank

Same21 selected SGEMMs/12 ResBlocks and candidate math as model-r1/r2. New JIT
cache is PREFIX_CACHE_DIR/rank-0 or rank-1 only during new-kernel compile/capture,
then the original cache path is restored. No global custom cache-manager class,
compiler patch or CUBIN-gate relaxation. Every captured ABI witness records its
rank-local cache policy. The [CPU race fixture](../cache-temp-race-cpu-r1) explains
why per-rank directories avoid temporary binary files from the other writer.
The [startup classifier](../startup-guard-r2) is retained.

CPU preflight passed again, including120 block-boundary cases and21 owner
identities/12 selected blocks. The GPU2/3 test now completed cleanly: both ranks
passed all six dynamic states and18 FP32 signedzero output checks,570 Linear,
782 norm and192 Conv2d master modifications/zero/restoration, original23+45
owners/old-CUBIN gates, RNG and baseline restoration. Rank-local cache loading
did not reproduce the previous temporary-file error.

All12 blocks together were slower: baseline medians252.367/252.331ms versus
candidate255.275/255.299ms. Median paired savings were-2.898/-2.878ms; do not
sum the two ranks. This candidate is rejected for performance and has no full
V2/video PSNR result. See [complete model evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r3).

Follow model-r1 staging instructions using a fresh private root/cache; do not
import the CPU-only compiler helper in GPU workers.
[CPU evidence](../../../results/aggressive-20261028/world-native-frame-interleaved-209/model-r3-preflight).
Official V2 remains287.745486 seconds, five-case total836.651410.
