# Complete Z1 FFN61 trial

Start from `experiments/nonv2-vae-all-display/unifolm`, then apply the three files in `production-overlay`. All59 original Python math sources are unchanged. `production-sources.json` freezes the resulting61 math/runtime files; `timing-asset-hashes.json` separately covers the evaluator, original shell/env and timing proof. The helper is byte-identical to both earlier real-input model tests.

Only33 FFNs are partitioned. Every rank executes the original12-round,50+50-step-per-round Z1 workload. Full spatial/time attention, global GroupNorm, masks, conditioning, decoding and sampling remain on each rank. The FFN input's independent leading sample dimension is halved, preserving whole spatial sequences or all16 time steps per sample. One gather restores each FFN output before its residual. Native Half GEMM dispatch can change rounding; this is not bitwise-equivalent to the single-GPU path.

The four additive evaluator hooks start the helper after the original process timer and before Torch import; initialize NCCL in the existing reproducibility stage; wrap the original DenoiserGraph after optimization configuration; then reset captured graphs, destroy the process group, wait for the helper and compare rank video hashes before the original profile writer. `timing-proof.json` describes source qualification, whose initial quality-unverified marker is retained; the separate terminal full evidence now establishes actual video quality.

The complete result is **166.190328s /47.080754066dB**,12 rounds/25 videos, on two A800s. It remains experimental: no contemporaneous full control and no new103-video accepted aggregate. See the parent README for the comparison limits.

`wait_nccl_full209.py` is the frozen two-GPU reservation/context-monitor runner, adapted only for this Z1 case and ordinary private tmpfs storage. It checks the selected evaluator's exit code, rounds and profile separately from the intentionally failed all-five-case shell. The qualified private NCCL tmpfs adapter is the same opt-in adapter used by adopted222; no library/kernel change or historical cleanup occurs. `NONV2_FFN_CACHE_DIR` must point to an executable writable cache directory; all data/tmp outputs may use private `/var/tmp`. The race machine's small executable home volume is almost full, so this run reuses its already-qualified FFN shape cache.

Run the CPU-only artifact verifier from the repository root:

```bash
python reproduce/verify_nonv2_ffn61_evidence.py
```
