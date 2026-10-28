"""Check the original mutable master wrappers with the new experiment active."""
import os
import subprocess
import sys
os.environ['UNIFOLM_WORLD_ZERO_WEIGHT_LOW_GEGLU']='all'
for script in ['check_world_bk64_c320_geglu.py','check_world_wide_gate_geglu.py','check_prepacked_geglu_runtime.py']:
    subprocess.run([sys.executable,'scripts/'+script],check=True)
print('ZERO_WEIGHT_LOW_GEGLU_WRAPPERS_CHECK passed=true original_factory_guards=3 '
      'first_timed_pack_and_master_refresh_hooks=true full_video_quality=unverified',flush=True)
