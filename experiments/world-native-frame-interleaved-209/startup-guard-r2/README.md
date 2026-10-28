# Startup ownership failure classification, CPU verified

These two orchestration files replace the same files in a fresh copy of model-r1.
The rank still requires exactly one non-parent context. Zero/multiple contexts
abort without a numerical or timing result. The exception now records that
reason; the bounded retry reads ranks.log as well as the parent log, excludes
the entire unresolved/foreign-context attempt, and retries only those cases.
A model numerical failure still terminates immediately. No GPU-idle thresholds,
locks, CUBIN gates, candidate kernels or numerics have changed.

The included CPU fixture executes the actual retry script against four subprocess
startup/error scenarios. Zero ownership, multiple ownership and foreign context
retry once into a successful fixture; a numerical mismatch does not retry. It
also evaluates the unchanged worker assertion for cardinalities0/1/2. No Torch
import or GPU allocation. Run python3 check_world_startup_retry209_cpu.py
--source-dir . from this directory. The fixture success is not a GPU-model pass.

[CPU results](../../../results/aggressive-20261028/world-native-frame-interleaved-209/startup-guard-r2).
