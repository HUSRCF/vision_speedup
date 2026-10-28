"""Run prepacked split-K actual-operand scaled FP16 Conv1d prototype."""
import argparse,ast,json,math
from pathlib import Path
from experiment_remote import STATE
from guarded_current_experiment import start,harvest,finish

KEY='head_prepacked_half_current209_micro_r1'

def main():
    global KEY
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','harvest'));parser.add_argument('--revision',choices=('r1',),default='r1');args=parser.parse_args();s=json.loads(STATE.read_text())
    KEY='head_prepacked_half_current209_micro_'+args.revision
    if args.action=='launch':
        assert args.revision=='r1'
        audit=s['world_head_half_operands_current209_audit_r1'];assert audit['clean']
        rows=[r for r in audit['report']['records'] if r['kind']=='Conv1d'];modules=[]
        for channels in (2048,1024,512,256):
            row=max((r for r in rows if r['weight_shape'][0]==channels),key=lambda r:math.prod(r['weight_shape']))
            modules.append(row['module'])
        extra=next(r for r in rows if r['weight_shape']==[2048,2048,5]);modules.append(extra['module'])
        config=dict(input_source=audit['diagnostic_directory']+'/native-head-operands.pt',input_sha256=audit['operand_snapshot_sha256'],checkpoint='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',modules=modules,
            tiles=[[bm,64,64,4,3,split,mode] for bm in (32,64) for split in (4,16) for mode in (0,1)])
        files={name:Path(__file__).with_name(Path(name).name).read_text() for name in ('scripts/head_scaled_half_209_candidate.py','scripts/head_prepacked_half_209_candidate.py','scripts/check_head_prepacked_half_209.py')}
        for text in files.values():ast.parse(text)
        files['head-prepacked-half-209-config.json']=json.dumps(config,indent=2)+'\n'
        start(s,KEY,files,'scripts/check_head_prepacked_half_209.py','HEAD_PREPACKED_HALF_209_CHECK',cpu_import='head_prepacked_half_209_candidate',scope='New numerical route after implicit gather r3 passed FP64 gates but all8 tiles3-8x slower: timed RNA scaled FP16 im2col, contiguous split-K FP16 GEMM with FP32 accum, timed FP32 partial reduction and bias. Config BM32/64,BN64,BK64,split4/16,longacc or independent blocks;8 real configurations plus4 odd. Scratch workspace canaries/NaN overwritten every call. Native cache uses original plan code with private diagnostic-only SCHEMAS global snapshot extended by only4 exact shape/kernel variants observed in clean r3; unchanged descriptor, path, payload-size and pointer assertions, no production globals edited. Each native filter cache checked original prebias/canary/version and original/cached outputs bitwise. Standalone cuDNN variants differ from model; primitive performance alone cannot qualify production. All209/110 preserved. Same FP64 RNA gates for actual, raw FP64 native oracle for odd, dynamic masters and inputs, RNG and full output checks. Timing includes all3 candidate kernels, master-weight packing excluded from both steady graphs. No runtime installation or full-video quality claim.')
        return
    value=harvest(s,KEY,['head-prepacked-half-209-report.json'])
    if value is None:return
    j,o,archive,clean=value
    if clean:
        report=json.loads((archive/'head-prepacked-half-209-report.json').read_text());assert report['passed'] and report['all_measured_baselines_have_original_native_weight_cache']
        assert len(report['records'])+len(report['compiler_failures'])+8*len(report['unsupported_native_caches'])==44
        assert all(r['native_weight_cache_used'] for r in report['records'] if r['pairs'])
        j.update(report=report,ready_for_runtime_qualification=bool(report['eligible']))
    finish(s,j,state='clean_head_prepacked_half_micro' if clean else 'terminal_head_prepacked_half_micro_failed')
    print(json.dumps(dict(state=j['state'],unsupported=len(j.get('report',{}).get('unsupported_native_caches',[])),eligible=[{k:r[k] for k in ('module','config','median_saved_ms')} for r in j.get('report',{}).get('eligible',[])]),indent=2) if clean else o['attempt'][-6500:])

if __name__=='__main__':main()
