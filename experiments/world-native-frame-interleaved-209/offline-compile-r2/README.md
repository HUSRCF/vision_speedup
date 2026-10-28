# Compile the nine selected new kernels without a GPU

This completed CPU-only helper compiled the nine geometries actually used by
the 21 selected model SGEMMs for the explicit target `('cuda',80)`. It ran with
`CUDA_VISIBLE_DEVICES=''`; Torch CUDA remained uninitialized. No device kernel
was loaded or launched. Compiler signatures and16-byte alignment attributes
match the pinned Triton2.3.1 JIT source for the full-size Torch allocations used
by the subsequent captured ABI witnesses.

Triton2.3.1 writes compiler/cache/launcher files before constructing its runtime
`CompiledKernel`. With no visible GPU, that final wrapper asks
`UnsupportedDriver` for `binary_ext` and fails. In this **separate CPU process**,
the helper temporarily replaces only the result constructor with a file reader,
`OfflineArtifacts`. All compiler stages and cache-key generation remain
unchanged. The helper is not imported by GPU workers, and the real model's
original Kahan CUBIN gates and new captured-pointer ABI checks remain in place.

The preceding r1 stopped in that wrapper after writing the first CUBIN; r2
reused that first cache entry. This is not a cold-compilation timing benchmark
or a GPU performance/quality result.

For a fresh scratch run on the pinned contest environment:

```bash
export CANONICAL_ROOT=/path/to/original209
export PREFIX_MODEL_ROOT=/absolute/path/to/world-native-frame-interleaved-209/model-r1
export PREFIX_CPU_SCRIPT=/absolute/path/to/this/precompile_world_interleaved209_cpu.py
export PREFIX_OFFLINE_WORK="$(mktemp -d /tmp/team07-prefix-offline-XXXXXX)"
mkdir -p "$HOME/.tmp"
export PREFIX_OFFLINE_CACHE="$(mktemp -d "$HOME/.tmp/prefix-offline-cache-XXXXXX")"
cd "$PREFIX_OFFLINE_WORK"
env -u TRITON_CACHE_MANAGER CUDA_VISIBLE_DEVICES='' \
  TRITON_CACHE_DIR="$PREFIX_OFFLINE_CACHE" \
  PYTHONPATH="$PREFIX_MODEL_ROOT/scripts:$CANONICAL_ROOT/src" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=123 \
  "$HOME/.venv/bin/python" "$PREFIX_CPU_SCRIPT" --root "$PREFIX_MODEL_ROOT"
```

All nine artifact hashes, compiler constants, signatures and raw output are in
[the result directory](../../../results/aggressive-20261028/world-native-frame-interleaved-209/offline-compile-r2).
Precompilation of these static kernels does not evaluate model inputs or replace
the required GPU model checks, full process timing and complete-video quality.
