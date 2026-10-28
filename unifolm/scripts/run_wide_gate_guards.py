"""Run both numerical and runtime guards sequentially on the reserved GPU."""
import subprocess
import sys

for script in ('benchmark_wide_gate_geglu.py', 'check_world_wide_gate_geglu.py'):
    subprocess.run([sys.executable, 'scripts/' + script], check=True)
