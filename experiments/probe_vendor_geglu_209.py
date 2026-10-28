"""Guard and archive the standalone main-GEGLU vendor hypothesis."""
import argparse,ast,json
from pathlib import Path
from experiment_remote import STATE
from guarded_current_experiment import start,harvest,finish

KEY='vendor_geglu_current209_micro_r1'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));args=parser.parse_args()
    s=json.loads(STATE.read_text())
    if args.action=='launch':
        assert s['canonical209_both_kernel_profiles_r2']['clean']
        files={name:Path(__file__).with_name(Path(name).name).read_text() for name in ('scripts/vendor_geglu_209_candidate.py','scripts/check_vendor_geglu_209.py')}
        for text in files.values():ast.parse(text)
        start(s,KEY,files,'scripts/check_vendor_geglu_209.py','VENDOR_GEGLU_209_CHECK',cpu_import='vendor_geglu_209_candidate',scope='New main-GEGLU structural route after209 profile20+13.6ms: two vendor strided-batched FP16-to-FP32 products, existing device Count-predicated third correction, ordered FP32 chunk reduction/bias/GELU/Half split. Retain all high/low terms of original primary-gate or both-compensated contract; change reduction order only. Four actual geometries, bias0/1, chunk64/K4/full plus two oddK65/C37/M19 bias0/1 =28 fixtures. Actual operands not claimed; synthetic random Half-exact masters then live general/Half/nearconstant/zero/restored inputs weights bias Count. Sample FP64 value<2e-6/gate<1e-5 plus hidden-vs-original<2e-5 gate all timing; full consumer all kernels timed DEBUGfalse, original input/weight packing equally excluded. NaN scratch/output,64canaries,fresh output,RNG/master restore, original JIT/run unchanged. Six alternating pairs7samples10replays100balancedwarmups. No model install or full-video quality/official score claim; all209/110 untouched.')
        return
    value=harvest(s,KEY,['vendor-geglu-209-report.json'])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        report=json.loads((archive/'vendor-geglu-209-report.json').read_text());assert report['passed'] and len(report['records'])==28
        j.update(report=report,ready_for_actual_operand_qualification=bool(report['eligible']))
    finish(s,j,state='clean_vendor_geglu_micro' if clean else 'terminal_vendor_geglu_micro_failed')
    print(json.dumps(dict(state=j['state'],eligible=[{k:r[k] for k in ('kind','geometry','has_bias','chunk','median_saved_ms')} for r in j.get('report',{}).get('eligible',[])]),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
