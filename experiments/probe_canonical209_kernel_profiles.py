"""Measure remaining actual kernel costs on a GPU separate from formal work."""
import argparse,ast
import json
from pathlib import Path
import re
from experiment_remote import STATE,fetch_files
from guarded_current_experiment import start,harvest,finish
from canonical204_kernel_profile_sources import sources as old_sources

def sources():
    files = {name.replace("canonical204","canonical209"):text.replace("canonical204","canonical209").replace("CANONICAL204","CANONICAL209") for name,text in old_sources().items()}
    name='scripts/profile_canonical209_graph_replay.py'
    profile=files[name]
    anchor='        original = model.apply_model\n';assert profile.count(anchor)==1
    profile=profile.replace(anchor,'        from audit_cross_context_209 import Audit\n        cross_audit=Audit(model,stage='world' if args.require_cudnn_tf32=='off' else 'decision')\n'+anchor)
    anchor='            allocation_records=[]\n';assert profile.count(anchor)==1
    profile=profile.replace(anchor,'            cross_audit.finish(args.diagnostic_dir)\n'+anchor)
    files[name]=profile
    files['scripts/audit_cross_context_209.py']=Path(__file__).with_name('audit_cross_context_209.py').read_text()
    for name,text in files.items():ast.parse(text)
    return files

KEY='canonical209_both_kernel_profiles_r3'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','harvest'));a=p.parse_args()
    s=json.loads(STATE.read_text())
    if a.action=='launch':
        assert s['canonical_source_audit']['source_count']==209
        kwargs={}
        assert not s['active_jobs']
        start(s,KEY,sources(),'scripts/run_canonical209_profiles.py','CANONICAL209_KERNEL_PROFILES_CHECK',threads=8,
              scope='Revision3 adds outer-stage gating: world snapshots must exclude preceding50decision calls; r2 CUPTI/bitwise replay attribution is valid but both operand files contain first decision operands. Revision2 corrects read-only operand snapshot lifetime: first eager module invocation only, module-qualified query identity, hard256MiB per-stage tensor budget, explicit skip report; r1 failed storage and all r1 operands excluded. Strategic reprofile current209/110 after new205.67s competitor V2 evidence. Both actual complete stage warmed graphs, three FP32 outputs bitwise/RNG/master versions fixed; new read-only cross-attention snapshots outside capture, representative branch shapes/strides/action-mask and shared-query groups for grouped cross-context attention fusion. Original function restored before CUPTI reference/profile, no original graph/cache/math changes. Native decision189/world23+45 owners retained. Attribution only; no official score or full-video quality claim. All GPU candidates exclusive/whole-attempt guard; preserve failures and original waiter.',**kwargs)
        return
    result=harvest(s,KEY,[])
    if result is None:return
    j,o,archive,clean=result
    directories=re.findall(r'^DIAGNOSTIC_DIR=(.+)$',o['log'] if clean else o['attempt'],re.M)
    summaries={};proofs={}
    for directory in directories:
        stage=Path(directory).name.split('-')[2];assert stage in ('decision','world')
        if not clean:continue
        destination=archive/stage
        fetch_files(directory,['operators.json','operators.txt','kernel-summary.json','flags.json','source-run.txt','condition-encoder-counts.json','replay-proof.json','cross-context-audit.json'],destination)
        summary=json.loads((destination/'kernel-summary.json').read_text());proof=json.loads((destination/'replay-proof.json').read_text())
        assert summary['stage']==stage and summary['instrumented'] and summary['no_official_score']
        assert all(proof.values()) and json.loads((destination/'flags.json').read_text())==j['effective_runtime_flags']
        source=s[j['baseline_key']]['run_directory'];assert (destination/'source-run.txt').read_text().strip()==source
        audit=json.loads((destination/'cross-context-audit.json').read_text());assert audit['original_function_restored'] and audit['no_graph_or_weight_or_rng_mutations']
        summaries[stage]=summary;proofs[stage]=proof
    if clean:
        assert set(summaries)=={'decision','world'}
        j.update(kernel_summaries=summaries,replay_proofs=proofs,diagnostic_directories=directories)
    finish(s,j,state='clean_canonical209_kernel_profiles' if clean else 'terminal_canonical209_kernel_profiles_failed')
    print(json.dumps(dict(state=j['state'],stages={stage:dict(total_kernel_us=row['total_kernel_us'],top_kernels=[{k:r[k] for k in ('name','count','self_cuda_us')} for r in row['top_kernels'][:8]]) for stage,row in summaries.items()}),indent=2) if clean else o['attempt'][-4500:])


if __name__=='__main__':main()
