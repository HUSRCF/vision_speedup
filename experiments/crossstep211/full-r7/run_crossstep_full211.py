"""Run original complete V2 timing/validators with a feature-reuse evaluator."""
import json,os,subprocess,tarfile
from pathlib import Path

def main():
    attempt=Path.cwd();source=Path(os.environ['CANONICAL_ROOT']);run=attempt/'run';run.mkdir();(run/'profiles').mkdir()
    flags=json.loads(Path(os.environ['CROSSSTEP_FLAGS_FILE']).read_text());env={k:v for k,v in os.environ.items()if not k.startswith('UNIFOLM_')};env.update(flags)
    env.update(RESULT_DIR=str(run/'outputs'),PROFILE_DIR=str(run/'profiles'),RESULT_JSON_PATH=str(run/'result.json'),LOG_PATH=str(run/'runtime.log'),BENCHMARK_N_ITER='original',UNIFOLM_CASE='unitree_z1_dual_arm_stackbox_v2',TEAM07_GUARD_ATTEMPT_DIR=str(attempt))
    (run/'flags.json').write_text(json.dumps(flags,indent=2)+'\n')
    manifest=json.loads((source/'crossstep211-production-sources.json').read_text())
    paths=[source/n for n in manifest]+[p for folder in ('configs','scripts/evaluation')for p in (source/folder).rglob('*')if p.is_file()and'__pycache__'not in p.parts]
    paths += [source/n for n in ('scripts/run_crossstep_compute_benchmark.sh','scripts/run_compute_benchmark.sh','scripts/team07_env.sh','crossstep211-production-sources.json')]
    with tarfile.open(run/'source.tar.gz','w:gz',dereference=True)as out:
        for p in sorted(set(paths)):out.add(p,arcname='./'+str(p.relative_to(source)),recursive=False)
    code=subprocess.run(['bash','scripts/run_crossstep_compute_benchmark.sh'],cwd=source,env=env).returncode
    result=json.loads((run/'result.json').read_text());row=next(r for r in result['cases']if r['case']=='unitree_z1_dual_arm_stackbox_v2')
    reuse=json.loads((run/'profiles/unitree_z1_dual_arm_stackbox_v2.crossstep.json').read_text())
    assert row['rounds']==11 and row['videos']==23 and row['execution_ok']and row['profile_complete']
    first,last=reuse['window'];interval=reuse['interval'];expected_reuse=11*sum(first<=p<last and(p-first)%interval!=0 for p in range(50))
    assert reuse['counts']['decision_calls']==reuse['counts']['world_calls']==550 and len(reuse['completed_generations'])==22 and reuse['counts']['world_reuse']==expected_reuse>0
    assert reuse['graph_count']==3
    if reuse.get('scope')=='spatial':
        assert len(reuse['selected_blocks'])==16 and reuse['all_convolutions_and_heads_live']and reuse['reuse_graph_head_filter_cache']
    report=dict(diagnostic_complete=True,all_dynamic_gates_passed=True,workload_complete=True,case_result=row,crossstep=reuse,benchmark_exit_code=code,run_directory=str(run),official_adopted=False,single_GPU=True,full_video_PSNR_required=True,quality_passed=row['psnr_passed'])
    (attempt/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('CROSSSTEP_FULL211_COMPLETE '+json.dumps(dict(process_wall_s=row['process_wall_s'],psnr_db=row['psnr_db'],quality_passed=row['psnr_passed'],rounds=11,videos=23)),flush=True)

if __name__=='__main__':main()
