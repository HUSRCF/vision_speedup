# Private tmpfs backing for NCCL control files

NCCL2.20.5's `ncclShmOpen` creates `/dev/shm/nccl-XXXXXX` directly; proxy progress also needs shared control pools when GPU payload traffic uses P2P/CUMEM/read. The original source is [NVIDIA shmutils.cc](https://github.com/NVIDIA/nccl/blob/v2.20.5-1/src/misc/shmutils.cc) and [proxy.cc](https://github.com/NVIDIA/nccl/blob/v2.20.5-1/src/proxy.cc). The competition container has a separate writable1GiB `/var/tmp` tmpfs while `/dev/shm` is almost full.

The opt-in preload library preserves canonical virtual filenames and six-character NCCL suffixes, while backing only this process tree's NCCL-shaped temporary files in one owned0700 tmpfs directory. `mkstemp`, `open` and `unlink` cooperate across the two child processes. Unrelated paths pass through. Disabled mode is transparent; a nonprivate/non-tmpfs root is rejected. Installed NCCL, CUDA kernels, collectives, payload dtype, model and sampling arithmetic are untouched. No old files or caches are removed, compressed, relocated or opened by this adapter.

CPU qualification covers both mkstemp variants, bidirectional cross-process mmap at two offsets, ordinary-file routing, deletion, disabled behavior and invalid-root rejection. The GPU0/1 test checks exact eager and CUDA-Graph all-gather for1/10/25/40MiB, with both reservation/execution locks and whole-attempt foreign-context monitoring. These are communication diagnostics, not inference scores. Scalar reports live under `results/aggressive-20261028/private-nccl-tmpfs`. Raw NCCL debug logs/addresses and shared-library binaries are not exported.

The complete matched-storage221/222 runs additionally observed both live rank processes' mapped NCCL control file descriptors on the `/var/tmp` device. Each rank references the same two5,767,524-byte unlinked control files; summing file sizes across both ranks would double-count shared backing. GPU payload transport is not inferred from the control backing. Source and tested binary hashes are recorded separately.

Build and check using the competition Python environment, choosing fresh directories:

```bash
python experiments/private-nccl-tmpfs/build_and_check.py \
  --workdir /var/tmp/team07-review-storage \
  --exec-dir "$HOME/.tmp/team07-review-storage-exec"
python reproduce/run_v2_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review222 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json
```

Keep the review checkout and host `.so` files on an executable mount. `/var/tmp` and `/dev/shm` on the race machine are noexec; Python source, metadata, videos and shared mmap backing can live there. The storage config is optional for machines with sufficient ordinary `/dev/shm` and output capacity. Compilation and CPU checks perform no GPU/model work. Loader symbol/path initialization occurs before Python's original timer; GPU/NCCL setup, model loading, graph capture, copies, communication, video work and teardown retain the original timing boundary. Both complete AB variants use the identical adapter/guard/storage policy.

The adopted result is documented in `experiments/initial-boundary-221/production222`, with exact matched full guards in `../initial-boundary-221/private-tmpfs-full`. No website submission is performed by these tools.
