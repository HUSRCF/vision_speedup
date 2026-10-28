# Configured temporal module census on meta

The actual original209 WMAModel constructor was instantiated on the meta
device with the exact inference configuration. No checkpoint, input data or
GPU was loaded. There are17 TemporalTransformer modules:16 regular Linear
projection blocks plus init_attn.0 with Conv1d projections320->512->320.
Every regular block retains a full16 time sequence per spatial position.
This is a structural census, not a GPU profile or a speed/quality result.

A future broader temporal shard must keep global5D GroupNorm intact and
remove the existing inner FFN gather when extending its boundary. Initial
Conv1d projection needs separate handling. Existing norm/layout/half-pack
and fused projection+residual owners must retain their full Python geometry
and original affine/rounding. Actual tensor strides and launched JIT
operators still require a live diagnostic before implementing routing.

Run with CUDA_VISIBLE_DEVICES empty, CANONICAL_ROOT fixed to original209,
its src/scripts on PYTHONPATH, cwd at that root for configs/train/meta.json,
and META_REPORT_PATH pointed to a fresh output in your private diagnostic
directory. Disable Python bytecode writes to preserve original source tree.
The earlier setup attempts lacked an explicit torch.utils.checkpoint import
and then the constructor config working directory; neither used CUDA.
