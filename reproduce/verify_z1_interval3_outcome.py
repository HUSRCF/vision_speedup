"""Verify that all interval3 variants failed full quality and were not adopted."""
import hashlib
import json
from pathlib import Path


def verify():
    repo = Path(__file__).resolve().parents[1]
    load = lambda p: json.loads(p.read_text())
    outcome = load(repo/'experiments/z1-interval3-tail66/outcome.json')
    assert not outcome['official_adopted'] and outcome['all_candidates_quality_failed']
    assert outcome['formal_pair_unchanged'] and outcome['formal_case_sum_s'] == 717.513174
    assert outcome['no_new_official103_claim'] and not outcome['website_uploaded']
    assert outcome['all_original_rounds_and50_DDIM_updates'] and outcome['coefficient_GPU_selfcheck_not_full_quality']
    expected = {
        'interval3': (17, 118.235515, 20.748202622292318),
        'first_position_full': (16, 119.882819, 24.731624557868443),
        'first_two_cycles_full': (15, 122.720353, 19.69892057379987),
    }
    assert set(outcome['trials']) == set(expected)
    for label, (predictions, seconds, psnr) in expected.items():
        row = outcome['trials'][label]
        evidence = repo/'results/aggressive-20261029'/row['frozen_family']/'full-r1'
        summary = load(evidence/'summary.json')
        assert row['process_wall_s'] == summary['process_wall_s'] == seconds
        assert row['PSNR_dB'] == summary['psnr_db'] == psnr < 25
        assert not row['quality_passed'] and not summary['quality_passed'] and not summary['official_adopted']
        assert row['prediction_steps_per_stage_per_round'] == predictions
        assert row['rounds'] == 12 and row['videos'] == 25 and row['GPUs'] == ['0', '2']
    archived = repo/'results/aggressive-20261029/z1-reuse-window66/official103/validated'
    for name, h in outcome['formal_pair_SHA256'].items():
        assert hashlib.sha256((archived/name).read_bytes()).hexdigest() == h
    print('Z1_INTERVAL3_OUTCOME_VERIFIED all_three_full_quality_failed=true not_adopted=true prior717_513174_pair_preserved=true')


if __name__ == '__main__': verify()
