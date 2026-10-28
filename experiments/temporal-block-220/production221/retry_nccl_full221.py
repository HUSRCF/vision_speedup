import os,shutil,subprocess,sys,time
from pathlib import Path
root=Path.cwd()
for index in range(1,17):
 attempt=root/('attempt-'+str(index));attempt.mkdir(exist_ok=False)
 (attempt/'flags.json').symlink_to(root/'flags.json')
 print('NCCL_FULL_ATTEMPT='+str(attempt),flush=True)
 with (attempt/'attempt.log').open('x') as out:p=subprocess.run([sys.executable,str(root/'wait_nccl_full209.py')],cwd=attempt,stdout=out,stderr=subprocess.STDOUT)
 text=(attempt/'attempt.log').read_text();print(text,flush=True)
 if p.returncode==0:
  for name in ('nccl-full-report.json','run-directory.txt','gpu-telemetry.jsonl','benchmark.log'):shutil.copyfile(attempt/name,root/name)
  break
 if 'foreign GPU context' in text:
  print('CONTAMINATED_ATTEMPT_EXCLUDED='+str(index),flush=True);time.sleep(2);continue
 raise RuntimeError('Timed evaluator failed; retained '+str(attempt))
else:raise RuntimeError('Too many contaminated attempts')
