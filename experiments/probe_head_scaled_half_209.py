"""Run block-sum/Kahan actual-operand scaled FP16 Conv1d prototypes."""
import argparse,ast,json,math
from pathlib import Path
from experiment_remote import STATE
from guarded_current_experiment import start,harvest,finish

KEY='head_scaled_half_current209_micro_r3'

def main():
    global KEY
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));parser.add_argument('--revision',choices=('r1','r2','r3'),default='r3');args=parser.parse_args();s=json.loads(STATE.read_text())
    KEY='head_scaled_half_current209_micro_'+args.revision
    if args.action=='launch':
        assert args.revision=='r3'
        audit=s['world_head_half_operands_current209_audit_r1'];assert audit['clean']
        rows=[r for r in audit['report']['records'] if r['kind']=='Conv1d'];modules=[]
        for channels in (2048,1024,512,256):
            row=max((r for r in rows if r['weight_shape'][0]==channels),key=lambda r:math.prod(r['weight_shape']))
            modules.append(row['module'])
        extra=next(r for r in rows if r['weight_shape']==[2048,2048,5]);modules.append(extra['module'])
        config=dict(input_source=audit['diagnostic_directory']+'/native-head-operands.pt',input_sha256=audit['operand_snapshot_sha256'],checkpoint='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',modules=modules,
            tiles=[[*tile,acc_mode] for tile in ([16,32,64,4,2],[32,64,64,4,2],[64,16,64,4,2],[32,64,32,4,2]) for acc_mode in (1,2)])
        files={name:Path(__file__).with_name(Path(name).name).read_text() for name in ('scripts/head_scaled_half_209_candidate.py','scripts/check_head_scaled_half_209.py')}
        for text in files.values():ast.parse(text)
        files['head-scaled-half-209-config.json']=json.dumps(config,indent=2)+'\n'
        start(s,KEY,files,'scripts/check_head_scaled_half_209.py','HEAD_SCALED_HALF_209_CHECK',cpu_import='head_scaled_half_209_candidate',scope='Revision3: independent FP32 block summation and Kahan compensation after each 32/64-element FP16 MMA dot, explicit nonfoldable rounding barriers. Long-accumulation r2 closed: relative FP64 error 1.24e-5 versus native 7.14e-7; no unchanged retry. RNA operand hypothesis and X/W scale256/unscale65536 unchanged. Five actual shapes, four tiles x two accumulation modes plus odd bias/no-bias fixtures =44 configurations. Unsupported native cache inventory records exact kernel names, argument sizes and failed plan schema; these shapes excluded from speed claims. Odd SIMT native uses raw FP64 oracle, candidate uses explicit RNA FP64; actual native/candidate use RNA reference, 6e-6/1e-5 accuracy gates before timing. Baseline original cuDNN graph with original native filter cache and only validated filter pointer patched; cache bits/canary/version proof. All209 sources/110 JIT/full flags fixed. Dynamic input/general weight/Half/zero/restored, fresh output, NaN/canary/RNG. Both caches refresh outside steady graph on master version. Six alternating pairs seven samples ten replays after100 balanced warmups. Primitive numerical prototype only; no runtime installation, full-video PSNR or official score claim.')
        return
    value=harvest(s,KEY,['head-scaled-half-209-report.json'])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        report=json.loads((archive/'head-scaled-half-209-report.json').read_text());assert report['passed'] and report['all_measured_baselines_have_original_native_weight_cache']
        assert len(report['records'])+len(report['compiler_failures'])+8*len(report['unsupported_native_caches'])==44
        assert all(r['native_weight_cache_used'] for r in report['records'] if r['pairs'])
        j.update(report=report,ready_for_runtime_qualification=bool(report['eligible']))
    finish(s,j,state='clean_head_scaled_half_micro' if clean else 'terminal_head_scaled_half_micro_failed')
    print(json.dumps(dict(state=j['state'],unsupported=len(j.get('report',{}).get('unsupported_native_caches',[])),eligible=[{k:r[k] for k in ('module','config','median_saved_ms')} for r in j.get('report',{}).get('eligible',[])]),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
