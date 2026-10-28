# Native xFormers batch boundary on adopted218

This candidate imports the nine actual adopted218 helper files from a verified
immutable tree, while original209 remains first on the model package path.
Both paired conditions enable adopted spatial/temporal routing and decision
native ResBlock grids. Only the extra native attention wrapper differs.

Inside an active full16 spatial scope, native xFormers receives the rank's
eight Q/K/V frames and the corresponding original global tensor-bias slice.
It copies local outputs into a full16 buffer; following original dispatches
retain full16 shapes, and the existing block-end gather joins valid frames.
Compensated self-attention and temporal attention do not call this wrapper.
No new gather, lower precision, input/output replay cache or JIT kernel.

The included24 CPU cases verify both ranks, four K lengths and no/global-prefix/
broadcast masks, full output shape and RNG preservation. Native CUDA xFormers
precision, actual dispatch, complete model performance and video quality remain
to be established. Model checks retain six dynamic states over570 Linear,782
norm and192 Conv2d masters, three signedzero outputs and all original cache/
CUBIN checks. Six balanced complete-model pairs include output-copy cost.
Additional profiler replays occur only after those timings and export aggregate
actual fmha kernel names/counts/durations, no trace payloads or GPU addresses.

The clean GPU0/1 model test completed. Both ranks passed all six dynamic
states/18 signedzero output checks,570 Linear/782 norm/192 Conv2d master
changes/zero/restoration, all original cache and CUBIN gates, RNG and baseline
restoration. All six full-forward timing pairs were faster. Median paired
savings were4.919807/4.903427ms (do not add the ranks). Actual native
attention used the same FP32 CUTLASS64x64 kernel family,70 calls per condition.
There is no full-process or video PSNR result yet. See
[complete model evidence](../../../results/aggressive-20261028/native-attention-218/model-decision-r2). The inherited diagnostic profiler may
retain a generic header; native_attention218_mode.py and its import in that
profiler define the actual extra operation. Current official results unchanged.

Stage in a fresh private directory, link scripts/evaluation to original209
evaluation, set CANONICAL_ROOT and ADOPTED218_ROOT to the verified respective
trees, use scripts:original209/src:original209/scripts:root on PYTHONPATH,
set MONITORED_GPUS and CUDA_VISIBLE_DEVICES to the same selected pair, and use
the normal original209 environment/flags. The guard rechecks strict idleness
under both locks before creating contexts and excludes whole contaminated
attempts. Run the CPU fixture with CUDA_VISIBLE_DEVICES empty, then
retry_nccl_model_pair.py. Do not enable the adopted production entry lifecycle
concurrently: this diagnostic installs identical routing explicitly in both
paired modes. This is not a timed competition entry or a submission.
