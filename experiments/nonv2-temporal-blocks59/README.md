# Wider nonV2 Half temporal blocks using the same33 gathers

This extends the existing nonV2 FFN partition, using original59 Half mathematical sources and full official Z1 inputs. The16 spatial FFNs remain partitioned. For17 temporal Transformers, including the512-channel initial block, the existing FFN gather moves after the complete inner BasicTransformerBlock: QKV, both temporal attentions, LayerNorm, residual additions and FFN run on each rank's half of independent spatial positions. Each position retains all16 time steps. Full five-dimensional GroupNorm and Conv1d boundary projections stay outside the partition. The number and payload shapes of output gathers remain33 per phase.

Both real-model stages pass six dynamic states and six alternating timing pairs. Parameter mutation/zero/restore now covers **404 master tensors**, including the temporal attention and norm parameters; the frozen harness retains its older `changed_FFN_masters` state labels. All three output branches pass a broad local numerical screen, RNG is unchanged, and the restored baseline is bitwise exact. Native GEMM M changes may affect rounding, so local tests do not establish video quality. Decision median savings are8.963581/10.377724ms across the two ranks on GPU0/1; world10.944511/9.565697ms on GPU2/3. Do not add rank timings or interpret these as complete-process savings.

The frozen62-source full trial completes **12 rounds /25 videos** on GPU0/1: **160.689794s, official PSNR46.789624872dB**. Both ranks'25 video hashes match. Each phase records600 denoiser calls,33 captured partitions, including17 wider temporal blocks. Model setup, helper launch, NCCL setup, graph captures, output writes and teardown remain inside the original evaluator timer. There are two helper modules plus a lifecycle runtime beyond the59 original math sources; the evaluator has only four additive hooks. All59 original math hashes remain unchanged.

The preceding same-GPU FFN-only trial was166.190328s, making this full trial5.500534s lower. These are sequential complete trials, not six full AB pairs. Relative to historical adopted single-GPU Z1166.780992s, the difference is6.091198s. The source60 Half-producer fusion is not included. This candidate is not yet adopted and has not yet received a fresh103-video aggregate. The root official pair remains unchanged while broader spatial-block work is evaluated.

The single-case shell reports four missing cases and exits1 by design. The selected evaluator exits0 and its complete profile and PSNR pass; its raw single-case log/JSON is not a five-case submission. CPU ownership checks cover36 FFN and32 temporal/mask cases without initializing CUDA. The candidate uses the existing executable Half-FFN cache; temporal attention and residual-norm specializations have dynamic row grids and reuse original warm binaries. Actual executable home free space remained151552B before and after model qualification. No historical cleanup or shared-cache changes occur. NCCL uses the already-qualified opt-in private tmpfs adapter.

Source/guard snapshots are in `decision-model-r1`, `world-model-r1` and `full62-r1`; scalar profiles/logs/video hashes are under `results/aggressive-20261028/nonv2-temporal-blocks59`. Verify offline:

```bash
python reproduce/verify_nonv2_temporal_block62_evidence.py
```
