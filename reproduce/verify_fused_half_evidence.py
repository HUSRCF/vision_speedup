"""Off-GPU provenance checks for the new Half fusion experiment."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def verify():
    root=Path(__file__).resolve().parents[1];here=root/'experiments/nonv2-fused-half-geglu';result=root/'results/aggressive-20261028/nonv2-fused-half-geglu'
    base=root/'experiments/nonv2-vae-all-display/unifolm'
    spec=importlib.util.spec_from_file_location('fused_half_builder',here/'build_nonv2_fused_half_production.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    changes=module.production_sources((base/module.ATTENTION).read_text())
    original=json.loads((base/'nonv2-aggressive-production-sources.json').read_text());manifest=json.loads((here/'production-sources.json').read_text());assert len(manifest)==60 and len(original)==59
    for n,h in manifest.items():
        path=here/'production-overlay'/n if n in changes else base/n
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h
        if n in changes:assert path.read_text()==changes[n]
    before=ast.parse((base/module.ATTENTION).read_text());after=ast.parse(changes[module.ATTENTION])
    def forward(tree):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='GEGLU')
        return cls,next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward')
    _,old=forward(before);cls,new=forward(after);cls.body[cls.body.index(new)]=old
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    flags=json.loads((here/'flags.json').read_text());parent=json.loads((base.parent/'unitree_z1_stackbox-flags.json').read_text())
    assert flags==dict(parent,UNIFOLM_NONV2_FUSED_HALF_GEGLU='1')
    for label in('selected-micro-r5','decision-model-r1','world-model-r1'):
        code=here/label
        for n,h in json.loads((code/'source-hashes.json').read_text()).items():assert hashlib.sha256((code/n).read_bytes()).hexdigest()==h
        ev=json.loads((result/label/'summary.json').read_text());assert ev['passed']and ev['clean']and not ev['official_adopted']
        if label=='selected-micro-r5':
            assert ev['selected320_only']and len(ev['records'])==1
            row=ev['records'][0];assert row['all6_pairs_positive']and len(row['pairs'])==6
            assert len(row['cases'])==6 and all(c['signedzero_bits']for c in row['cases'])
            assert row['RNG_unchanged']and row['original_values_and_pointers_restored']
        else:
            assert ev['original59_nonv2_baseline']and ev['RNG_unchanged']and ev['baseline_restoration_bitwise']
            assert ev['all_three_outputs_six_states_bitwise']and len(ev['cases'])==6
            assert ev['controller']['captured']==10 and ev['mutated_projection_tensors']==20
            assert len(ev['pairs'])==6 and ev['all6_pairs_positive']
    # Keep the later512-channel operator-only evidence separate from adoption.
    probe=here/'initial512-micro-r6';ev=result/'initial512-micro-r6'
    for n,h in json.loads((probe/'source-hashes.json').read_text()).items():assert hashlib.sha256((probe/n).read_bytes()).hexdigest()==h
    assert (probe/'nonv2_fused_half_geglu.py').read_bytes()==(here/'nonv2_fused_half_geglu.py').read_bytes()
    report=json.loads((ev/'micro-report.json').read_text());summary=json.loads((ev/'summary.json').read_text())
    assert report['passed']and report['sweep_scope']=='initial512'and len(report['records'])==5
    assert json.loads((ev/'clean.json').read_text())['clean']
    for row in report['records']:
        assert (row['M'],row['C'])==(40960,512)and len(row['cases'])==6 and all(c['signedzero_bits']for c in row['cases'])
        assert row['RNG_unchanged']and row['original_values_and_pointers_restored']and len(row['pairs'])==6
    winners=[row for row in report['records']if row['all6_pairs_positive']]
    assert len(winners)==1 and winners[0]['config']==summary['winning_config']
    assert not summary['official_adopted']and summary['no_actual_model_or_full_video_validation']
    assert summary['conditional_savings_upper_budget_s']<.1
    complete=json.loads((result/'complete-process-AB.json').read_text())
    if 'same_GPU_complete_process_saved_s'in complete:
        assert complete['all25_videos_SHA_equal_original59_baseline']and complete['one_complete_AB_not_six_process_pairs']
        videos=json.loads((result/'full-paired-video-SHA.json').read_text());assert len(videos)==25
        for variant in('baseline','candidate'):
            folder=result/f'full-{variant}-r1';summary=json.loads((folder/'summary.json').read_text())
            assert summary['clean']and summary['workload_complete']and summary['quality_passed']and summary['official_timer_unchanged']
            case=next(c for c in json.loads((folder/'result.json').read_text())['cases']if c['case']=='unitree_z1_stackbox')
            assert (case['rounds'],case['videos'])==(12,25)and case['psnr_passed']and case['execution_ok']and case['profile_complete']
            assert case['process_wall_s']==summary['process_wall_s']and case['psnr_db']==40.07491350059081
            timings=re.findall(r'^TIMING case=(\S+) process_wall_s=([0-9.]+)$',(folder/'runtime.log').read_text(),re.M)
            assert timings==[('unitree_z1_stackbox',str(case['process_wall_s']))]
            code=here/f'full-{variant}-r1'
            for n,h in json.loads((code/'asset-hashes.json').read_text()).items():assert hashlib.sha256((code/n).read_bytes()).hexdigest()==h
        assert abs(complete['same_GPU_complete_process_saved_s']-.67096)<1e-6
    adopted=json.loads((here/'accepted-outcome.json').read_text())
    assert adopted['official_adopted']and adopted['all103_officially_validated']and adopted['same25_videos_as_previous_official']
    assert adopted['Z1_s']==166.780992 and adopted['V2_unchanged_s']==270.825199
    assert adopted['five_case_sum_s']==818.692774
    fused_snapshot=root/'results/history/official221-z1-fused-before222'
    for n,h in adopted['official_pair_sha256'].items():assert hashlib.sha256((fused_snapshot/n).read_bytes()).hexdigest()==h
    gpu0=result/'full-candidate-r2'
    raw_case=next(c for c in json.loads((gpu0/'result.json').read_text())['cases']if c['case']=='unitree_z1_stackbox')
    assert raw_case['execution_ok']and raw_case['profile_complete']and raw_case['psnr_passed']and(raw_case['rounds'],raw_case['videos'])==(12,25)
    assert raw_case['process_wall_s']==adopted['Z1_s']and raw_case['psnr_db']==adopted['PSNR_dB']
    assert json.loads((gpu0/'profiles/unitree_z1_stackbox.json').read_text())['total_process_wall_s']==raw_case['process_wall_s']
    gpu0_code=here/'full-candidate-r2'
    for n,h in json.loads((gpu0_code/'asset-hashes.json').read_text()).items():assert hashlib.sha256((gpu0_code/n).read_bytes()).hexdigest()==h
    final=result/'official103-r2';outcome=json.loads((final/'result.json').read_text());assert outcome['status']=='accepted'and outcome['psnr_validation']['video_count']==103
    assert outcome==json.loads((fused_snapshot/'result.json').read_text())
    current_Z1=next(c for c in json.loads((root/'results/result.json').read_text())['cases']if c['case']=='unitree_z1_stackbox')
    assert current_Z1['process_wall_s']==adopted['Z1_s']and current_Z1['psnr_db']==adopted['PSNR_dB']
    links=json.loads((final/'immutable-video-links.json').read_text());assert links['count']==103 and links['ordinary_files']and links['existing_official103_SHA_equal_hardlinks_only']
    proof=json.loads((final/'same-run-source-proof.json').read_text());assert proof['exact_timed_source60_verified']and proof['no_GPU_rerun_or_timer_rewrite']and len(proof['hashes'])==25
    for n,h in json.loads((final/'script-hashes.json').read_text()).items():assert hashlib.sha256((final/n).read_bytes()).hexdigest()==h
    print('Adopted Z1 Half producer:60 sources, clean model/full AB and fresh official103 evidence verified.')


if __name__=='__main__':verify()
