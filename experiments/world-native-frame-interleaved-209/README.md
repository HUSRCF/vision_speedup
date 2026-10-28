# Synthetic native Winograd frame sharding

The two supporting `*_sources.py` generators in this directory regenerate both
frozen kernels byte for byte from original209. They are separate from the
remote-executed script sets pinned in each result's `source-proof.json`.

The contiguous half-column prototype failed native convolution. These new
kernels select physical columns `(logical_col // 8) * 16 + logical_col % 8`.
That selects the first eight frames in groups of16 for the tested native
transforms. Each rank's real eight frames are copied into those first eight
positions of a full N16 input. Original full matrix strides, input/output
transforms, compensated arithmetic, bias and native filter cache are retained.

`conv-r2` and `vector8-r1` are separate, immutable executed snapshots. Each
completed 14 matrix configurations plus 28 convolution configurations (two
layouts). Six changing-input/weight cases for each rank produced 336 bit-exact
convolution checks, including signed zero; the separate 84 matrix checks,
bounded captured-pointer ABI witnesses, RNG, cache guards and full16 graph
restoration also passed. Before every candidate case, the native GEMM output
workspace is poisoned to prevent stale values from passing. No native workspace
is wrapped in a Tensor, and no raw operands are exported.

Only **11/14 geometries** were consistently faster in complete single-GPU
convolution fixtures. Three packed geometries regressed. Neither snapshot is a
full-model or complete-video result, and neither is adopted. The vector8 hint
retains numerical behavior but does not demonstrate a meaningful performance
improvement over the unhinted experiment; these are separate runs.

On a contest machine with the pinned original209 source and graph-handle
extension, run a fresh scratch copy of either snapshot:

```bash
export CANONICAL_ROOT=/path/to/original209
export EXPERIMENT_ROOT="$(mktemp -d /tmp/team07-interleaved-XXXXXX)"
cp conv-r2/*.py conv-r2/*.json "$EXPERIMENT_ROOT/"
mkdir -p "$HOME/.tmp"
export TRITON_EXECUTABLE_CACHE_DIR="$(mktemp -d "$HOME/.tmp/interleaved-launchers-XXXXXX")"
export TRITON_CACHE_DIR="$EXPERIMENT_ROOT/triton-cache"
export TRITON_CACHE_MANAGER=split_experiment_triton_cache209:SplitExperimentCache
export PREFIX_COLUMN_LAYOUT=frame_interleaved
export PYTHONPATH="$EXPERIMENT_ROOT:$CANONICAL_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$EXPERIMENT_ROOT/tmp"
export TMPDIR="$EXPERIMENT_ROOT/tmp"
cd "$EXPERIMENT_ROOT"
"$HOME/.venv/bin/python" guard_single_tmp209.py --gpu 3 --entry audit_world_native_frame_prefix209_conv.py
```

For the vector8 snapshot, change the copy source to `vector8-r1`. Run from an
otherwise idle GPU; the parent enforces both locks and a whole-attempt foreign
context guard, and terminates only its own worker on contamination. New compiler
data and launcher caches are isolated from the production cache. The contest
host's `/tmp` is noexec, so small generated launcher libraries use private home
storage while bulk compiler artifacts remain in `/tmp`.

The earlier interleaved harness r1 stopped when it tried to call a nonexistent
`CachedKahanFilterSwap.replay()` method. r2 explicitly refreshes the existing
owner then replays its graph, matching the original model's execution contract.
Only clean completed r2 and vector8 runs supply the results described above.

All raw timing samples, telemetry and exact executed-source hashes:
[`conv-r2`](../../results/aggressive-20261028/world-native-frame-interleaved-209/conv-r2),
[`vector8-r1`](../../results/aggressive-20261028/world-native-frame-interleaved-209/vector8-r1).

Integrating the existing model still requires matching the 21 eligible world
ResBlock SGEMM owners by parameter identity, excluding two outer Upsample
owners, preserving full16 embedding operations and gathering both ranks before
temporal convolution. Original production owner/CUBIN checks must pass first.
Actual model numerical checks, full copy/communication timing and complete V2
quality remain unverified. Official V2 stays **287.745486 seconds**.
