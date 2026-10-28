# Synthetic half-column GEMM check

These are the exact executed sources from the clean r2 attempt. The original
209 source files are read without modification. This experiment compares the
existing compensated packed/Kahan arithmetic against the same arithmetic over
half the columns; it does not compare either path against native FP32 convolution.

All 14 configurations passed six changing-input/weight cases, including signed
zero bits, output guards, untouched tail and RNG checks. A fresh captured graph
and bounded CUDA parameter metadata verify each new kernel's pointer ABI.
All six timing pairs were faster for each configuration. Static weight packing
is common and excluded from those GEMM timings.

**The subsequent native convolution fixture failed.** Native Winograd columns
cannot simply be assumed to contain the first eight frames as a contiguous
prefix. Do not integrate this prefix mapping into the model based on the matrix
test. The current official V2 remains 287.745486 seconds.

To reproduce on the contest machine, use the original canonical source tree with
the pinned graph-handle extension. Run from a fresh scratch copy of this directory:

```bash
export CANONICAL_ROOT=/path/to/original209
export EXPERIMENT_ROOT="$(mktemp -d /tmp/team07-prefix-matrix-XXXXXX)"
cp ./*.py ./*.json "$EXPERIMENT_ROOT/"
mkdir -p "$HOME/.tmp"
export TRITON_EXECUTABLE_CACHE_DIR="$(mktemp -d "$HOME/.tmp/prefix-launchers-XXXXXX")"
export TRITON_CACHE_DIR="$EXPERIMENT_ROOT/triton-cache"
export TRITON_CACHE_MANAGER=split_experiment_triton_cache209:SplitExperimentCache
export PYTHONPATH="$EXPERIMENT_ROOT:$CANONICAL_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$EXPERIMENT_ROOT/tmp"
export TMPDIR="$EXPERIMENT_ROOT/tmp"
cd "$EXPERIMENT_ROOT"
"$HOME/.venv/bin/python" guard_single_tmp209.py --gpu 3 --entry audit_world_native_frame_prefix209_matrix.py
```

The parent holds both GPU locks, starts only on a fully idle GPU and checks for
foreign CUDA contexts throughout the attempt. It terminates only its own child
if contaminated. `/tmp` is noexec on the contest host: bulk compiler files go
there, while the small generated launcher shared libraries use a new private
home directory. Neither location is the default production compiler cache.

Full samples, exact executed-source hashes and telemetry are in
[`matrix-r2`](../../../results/aggressive-20261028/world-native-frame-prefix-209/matrix-r2).
