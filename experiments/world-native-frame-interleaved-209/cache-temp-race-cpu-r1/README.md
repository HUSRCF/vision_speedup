# Triton2.3.1 temporary CUBIN glob race: CPU reproduction

The actual installed FileCacheManager.put writes stem.cubin.tmp.pid_* before
atomic replacement. Actual CompiledKernel.__init__ globs stem.*, classifies only
.cubin as binary, and attempts to decode the temporary binary as text. This
fixture pauses the real writer at replacement and executes the real reader.
A shared directory reproduces UnicodeDecodeError at byte18; per-rank directories
do not expose the other writer's temporary file and read the expected bytes.

Only launcher import and the driver descriptor are mocked in a CPU-only child.
The bytes are a synthetic non-executable fixture, no original cache is edited,
and Torch CUDA stays uninitialized. No runtime compiler patch is proposed.
Run CUDA_VISIBLE_DEVICES='' ~/.venv/bin/python check_triton_cache_temp_race209_cpu.py.
[Results](../../../results/aggressive-20261028/world-native-frame-interleaved-209/cache-temp-race-cpu-r1).
