"""New temporal Tensor Core route, retaining both compensation and FP32 softmax."""
import argparse,ast,json
from pathlib import Path
from experiment_remote import STATE
from guarded_current_experiment import start,harvest,finish

KEY='temporal_compensated_current209_micro_r3'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));args=parser.parse_args();s=json.loads(STATE.read_text())
    if args.action=='launch':
        assert s['canonical209_both_kernel_profiles_r2']['clean']
        names=('scripts/temporal_compensated_209_candidate.py','scripts/check_temporal_compensated_209.py','scripts/check_vendor_geglu_209.py','scripts/vendor_geglu_209_candidate.py')
        files={name:Path(__file__).with_name(Path(name).name).read_text() for name in names}
        for text in files.values():ast.parse(text)
        start(s,KEY,files,'scripts/check_temporal_compensated_209.py','TEMPORAL_COMPENSATED_209_CHECK',cpu_import='temporal_compensated_209_candidate',scope='Revision3 attribution control after r2 compensated W2 all5 real shapes positive: compare ORIGINAL IEEE FP32 JIT warps1/2 versus protected-compensated W2, each against original W4. No new native numerical math; original JIT body/run retained, launch-only policy. Record QK/softmax/output int32 including signedzero in all six dynamic modes, FP64 gates, unchanged canary/fresh/RNG; original coupled graph and native fallback control preserved. Five actual geometries plusB3H3/B1H1 x3routes=21fixtures. Six alternatingpairs7samples10replays100balancedwarmups. Need all5 original-W2 rows exact/positive before actual model launch-only qualification. No inference install/full-video/official claim.')
        return
    value=harvest(s,KEY,['temporal-compensated-209-report.json'])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        report=json.loads((archive/'temporal-compensated-209-report.json').read_text());assert report['passed'] and len(report['records'])+len(report['compiler_failures'])==21
        j.update(report=report,ready_for_actual_model_qualification=bool(report['eligible']))
    finish(s,j,state='clean_temporal_compensated_micro' if clean else 'terminal_temporal_compensated_micro_failed')
    print(json.dumps(dict(state=j['state'],eligible=[{k:r[k] for k in ('shape','route','warps','all_modes_three_outputs_signedzero_exact','median_saved_ms') if k in r} for r in j.get('report',{}).get('eligible',[])]),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
