"""Full official-workload V2 trial for the qualified launch-only policy."""
import argparse,hashlib,json,re,tarfile
from datetime import datetime,timezone
from pathlib import Path
from experiment_remote import STATE,DIAGNOSTICS,remote,fetch_files,save_state
from temporal_native_w1_209_sources import FLAG,SOURCE,MANIFEST,source,provenance_source,cpu_checker
from probe_world_persistent_zero_low_full import captures

KEY='temporal_native_w1_current_full_r1'
BASE='world_head_filter_cache_current_full_gpu0_r1'
MODEL='temporal_native_w1_current209_model_r1'

def proof_code(root,base,flags,run=None):
    return f'''from pathlib import Path
import ast,hashlib,json
r=Path({root!r});b=Path({base['root']!r})
old=json.loads((b/{base['manifest_name']!r}).read_text());m=json.loads((r/{MANIFEST!r}).read_text())
assert len(m)==len(old)==209 and set(m)==set(old)=={{str(p.relative_to(r)) for p in r.glob('src/**/*.py')}}
assert hashlib.sha256((b/{base['manifest_name']!r}).read_bytes()).hexdigest()=={base['source_manifest_sha256']!r}
assert all(hashlib.sha256((b/n).read_bytes()).hexdigest()==h for n,h in old.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m.items())
assert all((r/n).read_bytes()==(b/n).read_bytes() for n in m if n!={SOURCE!r})
assert (r/{SOURCE!r}).read_text()=={source((Path(base['local_archive'])/SOURCE).read_text())!r}
def jits(p):
 text=p.read_text()
 if not any(v in text for v in ('@tr.jit','@triton.jit')):return []
 return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and any(ast.unparse(d) in ('tr.jit','triton.jit') for d in n.decorator_list)]
assert sum(len(jits(r/n)) for n in m)==110 and all(jits(r/n)==jits(b/n) for n in m)
assert (r/'scripts/team07_run.sh').read_text()=={provenance_source((Path(base['local_archive'])/'scripts/team07_run.sh').read_text())!r}
proof=dict(original208_other_sources_bytefixed=True,all110JIT_fixed=True,only_host_launch_policy_changed=True,no_new_kernel_math=True,exact_qualified_grid_storage_scale_and_warps=True)
if {run!r} is not None:
 run=Path({run!r});actual=dict(l.split('=',1) for l in (run/'provenance.txt').read_text().splitlines() if l.startswith('UNIFOLM_') and '=' in l)
 assert actual=={flags!r}
 proof.update(clean=not(run/'timing-contaminated.txt').exists(),explicit_effective_flags_provenance=True,videos={{str(p.relative_to(run/'outputs')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'outputs').rglob('*.mp4')}})
print(json.dumps(proof))'''

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','harvest'));a=p.parse_args()
    s=json.loads(STATE.read_text());base=s[BASE];model=s[MODEL]
    if a.action=='launch':
        assert KEY not in s and not s['active_jobs']
        assert base['official_adopted'] and base['root']==s['latest_v2_improvement']['root']==model['baseline_root']
        assert model['clean'] and not model['running'] and model['ready_for_full_trial']
        assert all(c['attention_launches']['captured']==34 and all(e['signedzero_bits'] for e in c['errors']) for stage in model['report']['stages'].values() for c in stage['cases'])
        audit=Path(s['canonical_source_audit']['directory']);assert s['canonical_source_audit']['source_count']==209
        # The original full archive contains only changed files; use the complete pinned audit.
        base_for_proof=dict(base,local_archive=str(audit))
        code={SOURCE:source((audit/SOURCE).read_text()),'scripts/team07_run.sh':provenance_source((audit/'scripts/team07_run.sh').read_text()),'scripts/check_temporal_native_w1_production_cpu.py':cpu_checker()}
        flags={**base['effective_runtime_flags'],FLAG:'1'}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        root='/home/team07/competition/unifolm-temporal-native-w1-full-'+stamp
        log='/home/team07/benchmark-runs/unifolm-launch/temporal-native-w1-full-'+stamp+'.log'
        proof=proof_code(root,base_for_proof,flags)
        result=json.loads(remote(f'''from pathlib import Path
import ast,hashlib,json,os,shlex,shutil,subprocess
b=Path({base['root']!r});r=Path({root!r});assert not r.exists() and shutil.disk_usage(b).free>512*2**20
raw=(b/{base['manifest_name']!r}).read_bytes();assert hashlib.sha256(raw).hexdigest()=={base['source_manifest_sha256']!r}
old=json.loads(raw);assert len(old)==209 and all(hashlib.sha256((b/n).read_bytes()).hexdigest()==h for n,h in old.items())
shutil.copytree(b,r,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__'));shutil.copy2(b/'.git',r/'.git')
for n,text in {code!r}.items():
 if n.endswith('.py'):ast.parse(text)
 assert n in [{SOURCE!r},'scripts/team07_run.sh'] or not(r/n).exists()
 (r/n).write_text(text)
m={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in old}}
(r/{MANIFEST!r}).write_text(json.dumps(m,indent=2)+chr(10))
exec({proof!r},{{}})
subprocess.run(['bash','-n',str(r/'scripts/team07_run.sh')],check=True)
env={{k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')}}
env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'src')+':'+str(r/'scripts'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
out=subprocess.run([str(Path.home()/'.venv/bin/python'),'scripts/check_temporal_native_w1_production_cpu.py'],cwd=r,env=env,capture_output=True,text=True)
assert out.returncode==0,out.stdout+out.stderr
cpu=json.loads((r/'temporal-native-w1-production-cpu.json').read_text());assert cpu['passed'] and not cpu['CUDA_context_initialized']
leaf='scripts/run_temporal_native_w1_full.sh'
lines=['#!/usr/bin/env bash','set -euo pipefail']+['export '+k+'='+shlex.quote(v) for k,v in {flags!r}.items()]
lines+=['export WAIT_RUN_MODE=case-v2','export WAIT_GPU_CANDIDATES="0 1 2 3"','export WAIT_TIMEOUT_SECONDS=86400','export RESERVATION_HOLD_SECONDS=120','export IDLE_GRACE_SECONDS=0','export CLEAN_POLL_SECONDS=2','export LATE_CONTAMINATION_SECONDS=0','exec bash '+shlex.quote(str(r/'scripts/wait_clean_run.sh'))]
(r/leaf).write_text(chr(10).join(lines)+chr(10));subprocess.run(['bash','-n',str(r/leaf)],check=True)
assets={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['scripts/team07_run.sh','scripts/check_temporal_native_w1_production_cpu.py','temporal-native-w1-production-cpu.json',leaf,'scripts/wait_clean_run.sh','scripts/evaluation/world_model_interaction.py','src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so']}}
assert assets['src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so']=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
env={{k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_')}};env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
with open({log!r},'x') as output:process=subprocess.Popen(['bash',str(r/leaf)],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=process.pid,asset_hashes=assets,cpu_policy=cpu,source_manifest_sha256=hashlib.sha256((r/{MANIFEST!r}).read_bytes()).hexdigest())))''',timeout=180).splitlines()[-1])
        job=dict(name=KEY,root=root,log=log,waiter_pid=result.pop('pid'),**result,running=True,gpu_candidates=['0','1','2','3'],baseline_root=base['root'],base_source_run=base['run_directory'],manifest_name=MANIFEST,verified_source_count=209,Triton_JIT_count=110,numerical_candidate=False,full_video_PSNR_unverified=True,official_adopted=False,effective_runtime_flags=flags,baseline_audit_directory=str(audit),state='guarded_temporal_native_w1_full_running',scope='Original IEEE FP32 temporal attention only num_warps4 to1 at five qualified shapes. All110JIT bodies fixed and208other Python sources bytefixed. Same209production flags plus one V2-only launch flag. Micro QK/P/output signedzero exact and both real complete graphs six dynamic/master cases exact/all12timingpairs positive. Full11x50/23videoSHA/quality/clean/faster/official103 required before local pair adoption. SingleGPU perattempt, whole-attempt foreign-context guard, no website upload.')
        s[KEY]=job;s['active_jobs'].append(job);s['next_action']='Harvest original temporal-native-w1 full waiter; preserve current local official pair until full workload/quality/all23SHA/clean/faster+103 validation.';save_state(s)
        print(json.dumps(dict(name=KEY,pid=job['waiter_pid'],root=root)));return
    job=s[KEY]
    if not job['running']:print(job['state']);return
    o=json.loads(remote(f'''from pathlib import Path
import json
p=Path('/proc/{job['waiter_pid']}/stat');print(json.dumps(dict(live=p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z',log=Path({job['log']!r}).read_text())))'''))
    if o['live']:print('LIVE original_pid='+str(job['waiter_pid'])+'\n'+o['log'][-2000:]);return
    archive=DIAGNOSTICS/(KEY+'-archive');fetch_files(str(Path(job['log']).parent),[Path(job['log']).name],archive)
    fetch_files(job['root'],[MANIFEST,SOURCE,*job['asset_hashes']],archive)
    assert hashlib.sha256((archive/MANIFEST).read_bytes()).hexdigest()==job['source_manifest_sha256']
    assert all(hashlib.sha256((archive/n).read_bytes()).hexdigest()==h for n,h in job['asset_hashes'].items())
    runs=re.findall(r'^RESULT_JSON=(.+)/result.json$',o['log'],re.M)
    if not runs:job.update(running=False,clean=False,state='terminal_temporal_native_w1_full_without_result',local_archive=str(archive),do_not_repeat_unchanged=True)
    else:
        run=runs[-1]
        fetch_files(run,['runtime.log','result.json','source.tar.gz','provenance.txt','gpu-info.txt','gpu-telemetry.csv','profiles/unitree_z1_dual_arm_stackbox_v2.json','profiles/psnr.json','profiles/execution.json'],archive)
        manifest=json.loads((archive/MANIFEST).read_text());assert len(manifest)==209
        with tarfile.open(archive/'source.tar.gz') as snapshot:
            for n,h in [*manifest.items(),*job['asset_hashes'].items()]:assert hashlib.sha256(snapshot.extractfile('./'+n).read()).hexdigest()==h,n
        proof=json.loads(remote(proof_code(job['root'],dict(base,local_archive=job['baseline_audit_directory']),job['effective_runtime_flags'],run)))
        runtime=(archive/'runtime.log').read_text();proof['capture_proof']=captures(runtime,s['world_persistent_zero_low_current_full_gpu0_r1']['selection'])
        for marker,count in [('WORLD_HEAD_FILTER_CACHE',45),('NATIVE_CONV1D_WEIGHT_CACHE',43),('NATIVE_REMAINING_FILTER_CACHE',8),('NATIVE_SGEMM_FP16',23)]:
            rows=[json.loads(l.split(' ',1)[1]) for l in runtime.splitlines() if l.startswith(marker+' ')]
            assert len(rows)==1 and rows[0]['selected']==count;proof[marker]=rows[0]
        result=json.loads((archive/'result.json').read_text());case=next(c for c in result['cases'] if c['case'].endswith('stackbox_v2'))
        profile=json.loads((archive/'profiles/unitree_z1_dual_arm_stackbox_v2.json').read_text())
        complete=bool(case['execution_ok'] and case['profile_complete'] and case['rounds']==11 and case['videos']==len(proof['videos'])==23 and all(profile['stages'][k]['count']==11 for k in ('decision_generation','simulation_generation')))
        assert abs(case['process_wall_s']-profile['total_process_wall_s'])<1e-6
        quality=bool(case['psnr_passed'] and case['psnr_db']>25);exact=proof['videos']==base['video_hashes'];faster=case['process_wall_s']<s['latest_v2_improvement']['current_s']
        passed=proof['clean'] and complete and quality and exact and faster
        gpu=re.findall(r'^CLEAN_RUN_GPU=(\d)',o['log'],re.M);assert gpu and gpu[-1] in job['gpu_candidates']
        job.update(running=False,clean=proof['clean'],workload_complete=complete,quality_passed=quality,all23_video_sha_exact=exact,full_video_PSNR_unverified=False,v2_s=case['process_wall_s'],psnr_db=case['psnr_db'],video_hashes=proof['videos'],run_directory=run,local_archive=str(archive),selected_gpu=gpu[-1],source_proof=proof,phases=profile['stages'],state='temporal_native_w1_full_faster_ready_official103' if passed else 'temporal_native_w1_full_slower_or_failed',do_not_repeat_unchanged=not passed)
        path=DIAGNOSTICS/(KEY+'-phase-evidence.json');path.write_text(json.dumps(job,indent=2)+'\n');job['phase_evidence']=str(path)
    s['active_jobs']=[s[j['name']] for j in s['active_jobs'] if s[j['name']]['running']];save_state(s)
    print(json.dumps({k:job.get(k) for k in ('state','v2_s','psnr_db','clean','workload_complete','all23_video_sha_exact')}))

if __name__=='__main__':main()
