"""Qualify an exact launch-only optimization on both complete production graphs."""
import argparse,json,re
from experiment_remote import STATE,fetch_files
from guarded_current_experiment import start,harvest,finish
from temporal_native_w1_209_model_sources import sources

KEY='temporal_native_w1_current209_model_r1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));args=parser.parse_args()
    s=json.loads(STATE.read_text())
    if args.action=='launch':
        micro=s['temporal_compensated_current209_micro_r3'];assert micro['clean']
        rows=[r for r in micro['report']['eligible'] if r['route']=='native' and r['warps']==1]
        assert len(rows)==5 and all(r['all_positive'] and r['all_modes_three_outputs_signedzero_exact'] for r in rows)
        configs=[dict(shape=r['shape'],heads=r['heads'],warps=1) for r in rows]
        start(s,KEY,sources(configs),'scripts/run_temporal_native_w1_209_pairs.py','TEMPORAL_NATIVE_W1_209_BOTH_STAGES_CHECK',threads=8,
            cpu_import='temporal_native_w1_209_mode',scope='Only original IEEE FP32 temporal JIT num_warps4 to1 for five microqualified actual shapes. Original209sources/110JIT bodies unchanged. Both complete production-stage graphs retain native decision189/world23+45 owners, exact flags and master version guards. Six changed-input/570Linear plus all FP32 GroupNorm/LayerNorm affine edits/zero/restore cases perstage, three final outputs strict int32 including signedzero, baseline restoration and RNG exact. Six alternating timing pairs perstage after balanced warmup. Requires all12 pairs faster before formal full video qualification. No official score or quality claim.')
        return
    value=harvest(s,KEY,[])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        dirs=re.findall(r'^DIAGNOSTIC_DIR=(.+)$',o['log'],re.M);assert len(set(dirs))==1
        fetch_files(dirs[0],['comparison.json','flags.json','source-run.txt','guard.log','decision/comparison.json','world/comparison.json','decision.log','world.log'],archive/'diagnostic')
        report=json.loads((archive/'diagnostic/comparison.json').read_text())
        assert set(report['stages'])=={'decision','world'}
        assert json.loads((archive/'diagnostic/flags.json').read_text())==j['effective_runtime_flags']
        for stage,row in report['stages'].items():
            assert len(row['cases'])==6 and row['Linear_master_tensors']==570 and row['norm_master_tensors']>148
            assert row['attention_launches']['captured']==34 and row['RNG_unchanged'] and row['baseline_restoration_bitwise']
            assert all(len(c['errors'])==3 and all(e['signedzero_bits'] for e in c['errors']) for c in row['cases'])
        j.update(report=report,ready_for_full_trial=report['ready_for_full_trial'])
    finish(s,j,state='clean_temporal_native_w1_model' if clean else 'terminal_temporal_native_w1_model_failed')
    print(json.dumps(dict(state=j['state'],ready_for_full_trial=j.get('ready_for_full_trial'),median_saved_ms={stage:r['median_saved_ms'] for stage,r in j.get('report',{}).get('stages',{}).items()}),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
