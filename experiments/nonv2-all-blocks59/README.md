# Adopted Z1:33 complete inner Transformer blocks,63 sources

The original59 nonV2 Half mathematical source files remain byte-identical. Four additional routing/lifecycle helpers and four additive evaluator hooks partition all16 spatial and17 temporal inner BasicTransformerBlocks. Each rank computes eight complete frames in spatial blocks, or half the independent spatial samples with all16 time steps in temporal blocks. Full five-dimensional GroupNorm, spatial/temporal outer projections, sampler and VAE remain outside these wrappers. The existing FFN-only33 gathers now restore complete block outputs after their residuals; their number and payload shapes do not increase.

The full GPU0/1 Z1 run is **152.203279s / official PSNR46.553398187dB**,12 rounds/25 videos. Both ranks'25 video hashes match. Each phase records600 denoiser calls and33 captured blocks. A fresh official CPU validation of all103 videos is **accepted**; that accepted pair totals **802.134091s** and is preserved in `results/history/official-z1-blocks63-before-other63`; the subsequent dual/pencils adoption totals **782.213299s** and the current root pair totals **778.585395s** after adopting V2 decision-temporal223. V2 was268.844229s at that Z1 adoption; current V2 is265.216325s. There was no website upload. The five-case score is an assembly of separately timed complete cases, not one consecutive five-case process run.

| Full Z1 trial | GPUs | Seconds | PSNR dB |
|---|---:|---:|---:|
| Historical adopted Half-producer60 | 1 | 166.780992 | 40.074914 |
| FFN-only61, GPU0/1 | 2 | 166.190328 | 47.080754 |
| Wider temporal blocks62, GPU0/1 | 2 | 160.689794 | 46.789625 |
| All inner blocks63, GPU0/1 | 2 | 152.203279 | 46.553398 |

The final trial is8.486515s below the preceding temporal62 run and14.577713s below historical adopted60. These are sequential complete trials and a historical single-/two-GPU comparison, not six complete AB pairs. Separately, both real-model stages pass six alternating timing pairs, with complete forward median savings16.5–18.3ms across ranks relative to the original59 model. Model timings include all33 gathers; ranks' savings must not be added.

Correctness boundaries:

- Spatial action bias comes from the original full16-frame mask, then slices each rank's eight frames and associated heads. Rank1 retains frames8–15. Module-specific context splits, including decoder differences, remain original.
- Full and local modes have separate condition-projection cache banks. Condition refresh slices the full context in local mode, updates the same captured storage, and restores the prior bank on exit. Dynamic input and parameter tests cover graph replay.
- The warm attention-mix kernel keeps its original full-size bound and launches exactly the local whole1024-element tiles. All selected local extents are tile-aligned, so that bound is inactive on every launched lane. CPU domain checks and runtime assertions cover this reuse; no pointer or kernel arithmetic is changed. This avoids creating new executable-cache specializations on the full home volume.
- Six dynamic model states mutate, zero and restore **756 master tensors**, including attention and norm parameters. The frozen harness retains older `changed_FFN_masters` labels. All three outputs pass a broad numerical screen; restored baseline is bitwise exact and RNG unchanged. Native GEMM shape changes affect rounding, so complete video quality was independently checked.
- The original process timer includes helper launch, both model setups, NCCL initialization, graph capture, video writes, graph/process-group teardown, helper wait and rank-video comparison. No initialization was moved ahead of that timer.

The raw single-case shell exits1 because four cases are absent; the selected Z1 evaluator exits0 and its profile/quality pass. Its raw log/JSON is not a five-case submission. `official103` publishes the fresh accepted pair, validator report, exact assembly scripts and immutable link proof:25 videos come from this actual timed run and78 are SHA-equal prior accepted videos. All are ordinary files in the new aggregate. Prior accepted816.711804s evidence remains in `results/history/official222-z1-fused-before-blocks63`.

Run from the repository root using the qualified private tmpfs configuration and an executable cache directory:

```bash
python reproduce/verify_snapshot.py
~/.venv/bin/python reproduce/run_z1_parallel.py --gpus 0,1 \
  --workdir /var/tmp/team07-review-z1-63 \
  --storage-config /var/tmp/team07-review-storage/storage-config.json \
  --cache-dir /path/to/executable/FFN-shape-cache
```

Build the storage configuration using `experiments/private-nccl-tmpfs/build_and_check.py`; its host `.so` needs an executable mount. `--prepare-only` materializes and verifies all63 source hashes without Torch/GPU execution. `--variant temporal62` or `ffn61` reproduces preceding candidates. A new cache needs adequate executable space; existing original warm caches are preserved. Source snapshots, two model-stage reports and full process/quality/telemetry evidence are published alongside this README. Cold-start reproduction may be slower than these warm-cache results.
