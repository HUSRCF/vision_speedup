"""Complete per-case trials of adopted Half inner block routing."""
import argparse,ast,hashlib,json,tarfile
from pathlib import Path
from experiment_remote import STATE,DIAGNOSTICS,WORKSPACE,remote,fetch_files,save_state
from build_nonv2_native_ffn64_sources import production_sources,POLICY,ENTRY,RUNTIME,MODE,ADAPTER
CASES={"g1":("unitree_g1_pack_camera",11)}

KEY=''
ALIAS=''
ROUNDS=0
BASE='nonv2_aggressive_g1_vae_all_display_full_r1'
CASE=''


def prepare(s):
    assert KEY not in s and not s['active_jobs']
    b=s[BASE];check=s['nonv2_ffn59_world_model_pair_r1']
    for stage in ('decision','world'):
        q=s[f'nonv2_native_ffn_grid59_g1_{stage}_model_r2'];assert q['clean']and q['ready_for_full_trial']
    files,proof=production_sources(WORKSPACE/'vision_speedup/experiments/nonv2-vae-all-display/unifolm')
    prior=WORKSPACE/'vision_speedup/experiments/initial-boundary-221/private-tmpfs-full/full-candidate-r1'
    scripts={n:(prior/n).read_text()for n in ('wait_nccl_full209.py','retry_nccl_full221.py','check_dual_gpu_graph.py')}
    guard=scripts['wait_nccl_full209.py'].replace('unitree_z1_dual_arm_stackbox_v2',CASE).replace("else 11","else "+str(ROUNDS)).replace('private222','nonv2-native-ffn64').replace('original209_sources_unchanged=True','original59_math_sources_unchanged=True')
    anchor="        assert proof['video_count']==2*expected_rounds+1 and proof['rank_videos_SHA256_equal']\n";assert guard.count(anchor)==1
    guard=guard.replace("tarfile.open(run/'source.tar.gz','w:gz')","tarfile.open(run/'source.tar.gz','w:gz',dereference=True)")
    guard=guard.replace(anchor,anchor+'''        execution=json.loads((run/'profiles/execution.json').read_text())
        assert execution['unitree_z1_stackbox']['exit_code']==0, 'Selected evaluator failed'
        result=json.loads((run/'result.json').read_text())
        row=next(r for r in result['cases']if r['case']=='unitree_z1_stackbox')
        assert row['rounds']==expected_rounds and row['videos']==2*expected_rounds+1
        assert row['execution_ok']and row['profile_complete']
        # The unchanged all-five-case shell exits1 for the four absent cases.
        # Selected quality is recorded independently, including failures.
''')
    scripts['wait_nccl_full209.py']=guard.replace('unitree_z1_stackbox',CASE)
    cpu=Path(__file__).with_name('check_nonv2_native_ffn_grid59_cpu.py').read_text().replace('from nonv2_native_ffn_grid59_mode import','from unifolm_wma.utils.nonv2_native_ffn_grid59_mode import')
    cpu=cpu.replace("Path(__file__).parent.parent/'qualified-shapes.json'","Path(__import__('unifolm_wma.utils.nonv2_native_ffn_grid59_mode',fromlist=['__file__']).__file__).parent/'qualified-half-native-ffn64.json'")
    cpu+='\n'+Path(__file__).with_name('check_nonv2_native_ffn_grid59_geglu_cpu.py').read_text().replace('from nonv2_native_ffn_grid59_geglu_mode import','from unifolm_wma.utils.nonv2_native_ffn_grid59_geglu_mode import')
    scripts['check_nonv2_ffn59_cpu.py']=cpu
    qualified=json.dumps(dict(synthetic_only=True,ranks=[r['rows'] for r in s['half_native_row_grid59_synthetic_r1']['report']['ranks']]),indent=2)+'\n'
    native=s['world_co225_candidate_full_r1']['root']+'/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'

    for t in scripts.values():ast.parse(t)
    flags=json.loads((WORKSPACE/f'vision_speedup/experiments/nonv2-vae-all-display/{CASE}-flags.json').read_text());flags['UNIFOLM_NCCL_PARALLEL']=POLICY
    storage=s['nccl_private_tmpfs_cpu_r1']
    out=json.loads(remote(rf'''from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess,tempfile
b=Path({b['root']!r});m=json.loads((b/{b['manifest_name']!r}).read_text());assert len(m)==59
assert hashlib.sha256((b/{b['manifest_name']!r}).read_bytes()).hexdigest()=={b['source_manifest_sha256']!r}
assert all(hashlib.sha256((b/n).read_bytes()).hexdigest()==h for n,h in dict(m,**{b['asset_hashes']!r}).items())
assert shutil.disk_usage('/var/tmp').free>128*2**20
w=Path(tempfile.mkdtemp(prefix='team07-nonv2-native-ffn64-'+{ALIAS!r}+'-',dir='/var/tmp'));w.chmod(0o700);r=w/'source';r.mkdir()
def cp(src,dst):
 if src.endswith('.so'):Path(dst).symlink_to(Path(src).resolve())
 else:shutil.copyfile(src,dst)
 return dst
for folder in ('src','configs','scripts/evaluation'):shutil.copytree(b/folder,r/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),copy_function=cp)
for n in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):shutil.copyfile(b/n,r/n)
assert (r/{ENTRY!r}).read_text()=={(WORKSPACE/'vision_speedup/experiments/nonv2-vae-all-display/unifolm'/ENTRY).read_text()!r}
for n,t in {files!r}.items():ast.parse(t);(r/n).write_text(t)
(r/'src/unifolm_wma/utils/qualified-half-native-ffn64.json').write_text({qualified!r})
native=Path({native!r});assert hashlib.sha256(native.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
asset=r/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so';asset.parent.mkdir(exist_ok=True);asset.symlink_to(native.resolve())
manifest={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in r.glob('src/**/*.py')}}
assert len(manifest)==64 and all(manifest[n]==h for n,h in m.items())
(r/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n');(r/'nccl-entry-timing-proof.json').write_text(json.dumps({proof!r},indent=2)+'\n')
d=w/'diagnostic';d.mkdir();(d/'runs').mkdir();(d/'tmp').mkdir();(w/'shared').mkdir(mode=0o700)
for n,t in {scripts!r}.items():(d/n).write_text(t)
(d/'flags.json').write_text(json.dumps({flags!r},indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(r/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(d/'tmp'))
p=subprocess.run([str(Path.home()/'.venv/bin/python'),str(d/'check_nonv2_ffn59_cpu.py')],cwd=d,env=env,capture_output=True,text=True);(d/'cpu-preflight.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
assets={{n:hashlib.sha256((r/n).read_bytes()).hexdigest()for n in ({ENTRY!r},'scripts/run_compute_benchmark.sh','scripts/team07_env.sh','nccl-entry-timing-proof.json','src/unifolm_wma/utils/qualified-half-native-ffn64.json','src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so')}}
print(json.dumps(dict(workspace=str(w),root=str(r),diagnostic_root=str(d),shared_root=str(w/'shared'),manifest=manifest,asset_hashes=assets,script_hashes={{n:hashlib.sha256((d/n).read_bytes()).hexdigest()for n in {list(scripts)!r}}})))''',timeout=180))
    j=dict(**out,name=KEY,state='nonv2_native_ffn64_CPU_qualified',running=False,official_adopted=False,verified_source_count=64,
           effective_runtime_flags=flags,new_shape_cache=check['new_shape_cache'],preload=storage['preload'],preload_sha256=storage['preload_sha256'],timing_proof=proof,case=CASE,alias=ALIAS,official_rounds=ROUNDS,full=True,gpu_candidates=[],full_video_PSNR_unverified=True)
    s[KEY]=j;save_state(s);fetch_files(j['diagnostic_root'],['flags.json','cpu-preflight.log',*scripts],DIAGNOSTICS/(KEY+'-preflight'));print('PREPARED64 '+j['root'])


def launch(s,pair):
    j=s[KEY];assert j['state']=='nonv2_native_ffn64_CPU_qualified'
    assert not any(set(x['gpu_candidates'])&set(pair)for x in s['active_jobs'])
    out=json.loads(remote(rf'''from pathlib import Path
import hashlib,json,os,subprocess
r=Path({j['root']!r});d=Path({j['diagnostic_root']!r});assert not(d/'launch.log').exists()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {dict(j['manifest'],**j['asset_hashes'])!r}.items())
assert all(hashlib.sha256((d/n).read_bytes()).hexdigest()==h for n,h in {j['script_hashes']!r}.items())
assert hashlib.sha256(Path({j['preload']!r}).read_bytes()).hexdigest()=={j['preload_sha256']!r}
assert Path({j['new_shape_cache']!r}).is_dir()
env={{k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('RANK','LOCAL_RANK','WORLD_SIZE','TRITON_CACHE_DIR','TRITON_CACHE_MANAGER')}}
env.update(PATH=str(Path.home()/'.venv/bin')+':'+env['PATH'],LD_PRELOAD={j['preload']!r},TEAM07_NCCL_SHM_ROOT={j['shared_root']!r},CUDA_VISIBLE_DEVICES={','.join(pair)!r},MONITORED_GPUS={','.join(pair)!r},CANONICAL_ROOT=str(r),NONV2_FFN_CACHE_DIR={j['new_shape_cache']!r},FULL_RUN_PARENT=str(d/'runs'),PYTHONPATH=str(d),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(d/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs',TRIAL_N_ITER='original')
with(d/'launch.log').open('x')as out:p=subprocess.Popen([str(Path.home()/'.venv/bin/python'),str(d/'retry_nccl_full221.py')],cwd=d,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(waiter_pid=p.pid)))'''))
    j.update(**out,running=True,state='guarded_full_nonv2_native_ffn64',gpu_candidates=pair);s['active_jobs'].append(j);save_state(s);print(json.dumps(dict(key=KEY,pid=j['waiter_pid'])))


def harvest(s):
    j=s[KEY]
    if not j['running']:print(j['state']);return
    obs=json.loads(remote(rf'''from pathlib import Path
import json
p=Path('/proc/{j['waiter_pid']}/stat');d=Path({j['diagnostic_root']!r});a=d/'attempt-1'
print(json.dumps(dict(live=p.exists()and p.read_text().rsplit(')',1)[1].split()[0]!='Z',log=(d/'launch.log').read_text()[-2500:],attempt=(a/'attempt.log').read_text()[-3500:]if(a/'attempt.log').exists()else'',benchmark=(a/'benchmark.log').read_text()[-1500:]if(a/'benchmark.log').exists()else'',complete=(d/'nccl-full-report.json').exists())))'''))
    if obs['live']:print('LIVE '+obs['log']+obs['attempt']+'\n'+obs['benchmark']);return
    a=DIAGNOSTICS/(KEY+'-archive');names=['launch.log','flags.json','cpu-preflight.log',*j['script_hashes']]
    if obs['complete']:names+=['nccl-full-report.json','run-directory.txt','benchmark.log','gpu-telemetry.jsonl']
    fetch_files(j['diagnostic_root'],names,a)
    j.update(running=False,clean=False,state='terminal_nonv2_native_ffn64_failure',local_archive=str(a))
    if obs['complete']:
        report=json.loads((a/'nccl-full-report.json').read_text());run=report['run_directory']
        names=['runtime.log','result.json','source.tar.gz','provenance.txt','parallel-effective-flags.json','parallel/parallel-complete.json','parallel/worker-profile.json','parallel/rank0-controller.json','parallel/rank1-controller.json',f'profiles/{CASE}.json','profiles/execution.json','profiles/psnr.json']
        fetch_files(run,names,a)
        with tarfile.open(a/'source.tar.gz')as snap:
            for n,h in dict(j['manifest'],**j['asset_hashes']).items():assert hashlib.sha256(snap.extractfile('./'+n).read()).hexdigest()==h
        row=next(r for r in json.loads((a/'result.json').read_text())['cases']if r['case']==CASE)
        for rank in (0,1):
            controller=json.loads((a/f'parallel/rank{rank}-controller.json').read_text());assert controller['stage_captures']==dict(decision=31,world=31)and controller['controller']['captured']==62
            assert controller['controller']['GEGLU']['captured']==62 and len(controller['controller']['native_binding'])==2
        j.update(clean=report['clean'],report=report,run_directory=run,case_result=row,process_wall_s=row['process_wall_s'],psnr_db=row['psnr_db'],workload_complete=row['rounds']==ROUNDS and row['videos']==2*ROUNDS+1 and row['execution_ok']and row['profile_complete'],quality_passed=row['psnr_passed'],rank_videos_exact=report['parallel']['rank_videos_SHA256_equal'],state='complete_nonv2_native_ffn64')
        j['full_video_PSNR_unverified']=False
        j['ready_for_official103']=j['clean']and j['quality_passed']and j['workload_complete']and j['process_wall_s']<s['official_case_sources'][CASE]['process_wall_s']
    s['active_jobs']=[s[x['name']]for x in s['active_jobs']if s[x['name']]['running']];save_state(s)
    print(json.dumps({k:j.get(k)for k in ('state','clean','process_wall_s','psnr_db','quality_passed','ready_for_official103')}));print(obs['log']+obs['attempt'])


def main():
    global KEY,ALIAS,CASE,ROUNDS
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','harvest']);p.add_argument('--gpus',default='0,1');p.add_argument('--case',choices=list(CASES),required=True);a=p.parse_args();ALIAS=a.case;CASE,ROUNDS=CASES[ALIAS];KEY=f'nonv2_native_ffn64_{ALIAS}_full_r1';s=json.loads(STATE.read_text());pair=a.gpus.split(',');assert len(pair)==len(set(pair))==2 and set(pair)<=set('0123')
    if a.action=='prepare':prepare(s)
    elif a.action=='launch':launch(s,pair)
    else:harvest(s)


if __name__=='__main__':main()
