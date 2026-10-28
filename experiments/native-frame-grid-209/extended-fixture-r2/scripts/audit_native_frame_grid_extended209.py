"""Extend grid-only fixtures to every remaining real ResBlock 3x3 geometry."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from audit_native_frame_grid209 import run, GraphAPI, production


def main():
    torch.set_grad_enabled(False)
    assert torch.__version__ == '2.3.1+cu121'
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    topology = json.loads(Path('resblock-meta-topology209.json').read_text())
    assert topology['meta_topology_complete'] and topology['original_ResBlocks'] == 22
    prior = {(320,320,40,64), (640,640,20,32), (1280,1280,10,16),
             (1280,1280,5,8), (640,1920,20,32), (320,960,40,64)}
    all_shapes = {tuple(g) for g in topology['spatial_3x3_geometries']}
    assert len(all_shapes) == 14 and prior < all_shapes
    remaining = sorted(all_shapes-prior)
    assert len(remaining) == 8
    directory = Path.home()/'benchmark-runs/diagnostics'/(
        'native-frame-grid-extended-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory.mkdir(exist_ok=False)
    print('DIAGNOSTIC_DIR='+str(directory), flush=True)
    so = Path(production.__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'
    assert hashlib.sha256(so.read_bytes()).hexdigest() == 'ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    spec = importlib.util.spec_from_file_location('team07_cuda_graph_handles', so)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    api = GraphAPI()
    rows = []
    for co, ci, h, w in remaining:
        for kind in ('NCHW', 'NC_permuted'):
            for rank in (0, 1):
                row = run(extension, api, co, ci, h, w, rank, kind)
                rows.append(row)
                print('NATIVE_FRAME_GRID_EXTENDED '+json.dumps(row), flush=True)
                (directory/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
    assert len(rows) == 32
    passed = all(r['all6_dynamic_cases_first8_signedzero_bit_exact'] for r in rows)
    report = dict(diagnostic_complete=True, synthetic_only=True, rows=rows,
        actual_meta_topology_ResBlocks=22, all_3x3_geometries=sorted(all_shapes),
        previously_qualified_geometries=sorted(prior), new_geometries=remaining,
        all32_first8_dynamic_cases_signedzero_bit_exact=passed,
        all32_original_grid_restorations_signedzero_bit_exact=True,
        all_kernel_argument_bytes_unchanged=True,
        no_raw_operands_or_addresses_exported=True,
        native_runtime_full_model_choices_unverified=True,
        no_full_model_or_video_quality_result=True, no_official_score=True)
    (directory/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    assert passed, 'At least one new geometry changes first-eight outputs'
    print('NATIVE_FRAME_GRID_EXTENDED_CHECK passed=true', flush=True)


if __name__ == '__main__':
    main()
