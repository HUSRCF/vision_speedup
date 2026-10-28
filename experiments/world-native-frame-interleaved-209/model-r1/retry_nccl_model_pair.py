import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path.cwd();deadline=time.monotonic()+14400
for index in range(1,17):
 assert time.monotonic()<deadline
 attempt=root/('attempt-'+str(index));attempt.mkdir(exist_ok=False)
 for name in ('scripts','flags.json','check_nccl_geglu_model_rank.py'):(attempt/name).symlink_to(root/name,target_is_directory=name=='scripts')
 log=attempt/'attempt.log'
 print('NCCL_MODEL_ATTEMPT='+str(attempt),flush=True)
 with log.open('x') as output:p=subprocess.run([sys.executable,str(root/'wait_nccl_geglu_model_pair.py')],cwd=attempt,stdout=output,stderr=subprocess.STDOUT)
 text=log.read_text();print(text,flush=True)
 if p.returncode==0:
  for name in ('rank0-ready.json','rank1-ready.json','rank0-report.json','rank1-report.json','nccl-geglu-model-report.json','ranks.log'):shutil.copyfile(attempt/name,root/name)
  break
 if 'AssertionError: foreign GPU context' in text:
  print('NCCL_MODEL_CONTAMINATED_ATTEMPT_EXCLUDED='+str(index),flush=True);time.sleep(2);continue
 if (attempt/'ranks.log').exists():print((attempt/'ranks.log').read_text()[-8000:],flush=True)
 raise RuntimeError('Model diagnostic failed: '+str(attempt))
else:raise RuntimeError('Too many contaminated attempts')
