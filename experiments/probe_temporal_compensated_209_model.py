"""Qualify actual model error and throughput before any numerical full video trial."""
import argparse,json,re,hashlib
from experiment_remote import STATE,fetch_files
from guarded_current_experiment import start,harvest,finish
from temporal_compensated_209_model_sources import sources

KEY='temporal_compensated_current209_model_r1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));args=parser.parse_args();s=json.loads(STATE.read_text())
    if args.action=='launch':
        micro=s['temporal_compensated_current209_micro_r3'];assert micro['clean']
        rows=[r for r in micro['report']['eligible'] if r['route']=='compensated' and r['warps']==2]
        assert len(rows)==5 and all(r['all_positive'] and r['numerical_gate'] for r in rows)
        configs=[dict(shape=r['shape'],heads=r['heads'],warps=2) for r in rows]
        files=sources(configs);name='scripts/temporal_compensated_209_candidate.py'
        assert hashlib.sha256(files[name].encode()).hexdigest()==micro['script_hashes'][name]
        start(s,KEY,files,'scripts/run_temporal_compensated_209_pairs.py','TEMPORAL_COMPENSATED_209_BOTH_STAGES_CHECK',threads=8,candidates='3',cpu_import='temporal_compensated_209_mode',
            scope='Five microqualified temporal shapes, scaled-half three-product QK/PV with exact original FP32 softmax tree and CTA-local native IEEE domain fallback. Separate numerical route from exact nativeW1 launch optimization. Both complete actual-stage production graphs,570Linear+allFP32norm masters changed/zero/restored,6cases/stage, final3outputs grossrelative<.002/maxabs<.02; baseline restoration strict int32, RNG unchanged. Original209src/110JIT bodies and nativeowners retained. Six balanced alternatingtimingpairs/stage against actualcurrent canonical graph. No full-video quality/official score claim; full11x50 PSNR>25 andclean/faster required before adoption.')
        return
    value=harvest(s,KEY,[])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        dirs=re.findall(r'^DIAGNOSTIC_DIR=(.+)$',o['log'],re.M);assert len(set(dirs))==1
        fetch_files(dirs[0],['comparison.json','flags.json','source-run.txt','guard.log','decision/comparison.json','world/comparison.json','decision.log','world.log'],archive/'diagnostic')
        report=json.loads((archive/'diagnostic/comparison.json').read_text());assert set(report['stages'])=={'decision','world'}
        assert json.loads((archive/'diagnostic/flags.json').read_text())==j['effective_runtime_flags']
        for stage,row in report['stages'].items():
            assert len(row['cases'])==6 and row['Linear_master_tensors']==570 and row['norm_master_tensors']>148
            assert row['attention_launches']['captured']==34 and row['RNG_unchanged'] and row['baseline_restoration_bitwise']
            assert all(len(c['errors'])==3 and all(e['relative_l2']<.002 and e['max_abs']<.02 for e in c['errors']) for c in row['cases'])
        j.update(report=report,ready_for_full_trial=report['ready_for_full_trial'])
    finish(s,j,state='clean_temporal_compensated_model' if clean else 'terminal_temporal_compensated_model_failed')
    print(json.dumps(dict(state=j['state'],ready_for_full_trial=j.get('ready_for_full_trial'),median_saved_ms={stage:r['median_saved_ms'] for stage,r in j.get('report',{}).get('stages',{}).items()}),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
