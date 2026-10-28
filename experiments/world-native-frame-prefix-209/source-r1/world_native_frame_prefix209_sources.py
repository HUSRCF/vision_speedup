"""Generate experimental half-frame Winograd GEMMs without changing arithmetic.

Only CTA row/column mapping gains ACTIVE_COLS; COLS remains the original full
matrix leading dimension in every load, store and mask. Round-trip AST checks
recover each original kernel. This is a source/mapping proof, not numerical
GPU or model evidence. The original 209-file tree is never edited.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path


def sources(base):
    base=Path(base)
    manifest=json.loads((base/'world-head-filter-cache-production-sources.json').read_text())
    assert len(manifest)==209
    specs=[
        ('native_winograd_gemm_fp16_packed_kernel.py','transformed_gemm_fp16_packed','packed_boundary'),
        ('native_winograd_gemm_fp16_kahan_kernel.py','transformed_gemm_fp16_kahan','kahan_boundary'),
    ]
    header='''import triton as tr
import triton.language as tl
from unifolm_wma.utils.native_winograd_gemm_fp16_kernel import boundary as packed_boundary
from unifolm_wma.utils.native_winograd_gemm_fp16_kahan_kernel import boundary as kahan_boundary
'''
    generated=[]
    proof=[]
    for filename,name,boundary in specs:
        path='src/unifolm_wma/utils/'+filename
        raw=(base/path).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        assert digest==manifest[path]
        tree=ast.parse(raw)
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        candidate=copy.deepcopy(original)
        candidate.name=name+'_frame_prefix'
        assert not candidate.args.kwonlyargs and not candidate.args.defaults
        candidate.args.args.append(ast.arg(arg='ACTIVE_COLS',annotation=ast.parse('tl.constexpr',mode='eval').body))
        original_tiles=copy.deepcopy(candidate.body[:2])
        assert [ast.unparse(n.targets[0]) for n in original_tiles]==['row_tile','col_tile']
        candidate.body[:2]=ast.parse('''row_tile=tl.program_id(0)//tl.cdiv(ACTIVE_COLS,BN)
col_tile=tl.program_id(0)%tl.cdiv(ACTIVE_COLS,BN)''').body
        assertion=ast.parse('tl.static_assert(ACTIVE_COLS*2==COLS)').body[0]
        candidate.body.insert(0,assertion)
        for node in ast.walk(candidate):
            if isinstance(node,ast.Name) and node.id=='boundary':node.id=boundary
        ast.fix_missing_locations(candidate)
        # Every memory offset/mask, K block, cast, product, explicit rounding
        # boundary and compensated sum comes from the original AST verbatim.
        restored=copy.deepcopy(candidate)
        restored.name=name
        assert restored.args.args.pop().arg=='ACTIVE_COLS'
        restored.body.pop(0)
        restored.body[:2]=original_tiles
        for node in ast.walk(restored):
            if isinstance(node,ast.Name) and node.id==boundary:node.id='boundary'
        assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
        assert [a.arg for a in candidate.args.args if a.annotation is None]==[
            a.arg for a in original.args.args if a.annotation is None]
        generated.append(ast.unparse(candidate))
        proof.append(dict(source=path,original_SHA256=digest,
            original_function=name,candidate_function=candidate.name,
            roundtrip_original_AST_exact=True,
            all_memory_offsets_masks_K_tiles_casts_products_and_FP32_sums_original=True,
            runtime_pointer_formal_arguments_unchanged=True))
    text=header+'\n\n'+'\n\n'.join(generated)+'\n'
    ast.parse(text)
    return text,proof


def mapping_proof(base):
    path=Path(base)/'src/unifolm_wma/utils/native_graph_sgemm.py'
    manifest=json.loads((Path(base)/'world-head-filter-cache-production-sources.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['src/unifolm_wma/utils/native_graph_sgemm.py']
    tree=ast.parse(path.read_text())
    configs={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id=='configs':
                configs.update({tuple(k):v for k,v in ast.literal_eval(node.value).items()})
            if node.targets[0].id=='additional_configs':
                configs.update({tuple(k):v['config'] for k,v in ast.literal_eval(node.value).items()})
    assert len(configs)==14
    rows=[]
    for (co,cols,k),(bm,bn,bk,warps,stages) in sorted(configs.items()):
        assert cols%2==0
        active=cols//2
        nr,nc=math.ceil(co/bm),math.ceil(active/bn)
        observed={(pid//nc,pid%nc) for pid in range(nr*nc)}
        expected={(r,c) for r in range(nr) for c in range(nc)}
        assert observed==expected and len(observed)==nr*nc
        # Partial final tiles can write extra *unused* columns; do not claim
        # that all stores stop exactly at ACTIVE_COLS. Original COLS masks and
        # strides deliberately remain unchanged.
        rounded=min(cols,nc*bn)
        rows.append(dict(co=co,cols=cols,k=k,bm=bm,bn=bn,bk=bk,
            active_cols=active,rounded_written_prefix_cols=rounded,
            original_grid=[math.ceil(co/bm)*math.ceil(cols/bn),36],
            prefix_grid=[nr*nc,36],all_output_channel_tiles_covered=True,
            all_active_column_tiles_covered=True,duplicate_CTA_tiles=False,
            full_COLS_leading_dimensions_and_masks_preserved=True))
    return rows


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    text,proof=sources(args.base)
    rows=mapping_proof(args.base)
    args.output.mkdir(exist_ok=False)
    (args.output/'world_native_frame_prefix209_kernel.py').write_text(text)
    report=dict(CPU_source_and_mapping_proof_complete=True,source_proof=proof,geometries=rows,
        candidate_SHA256=hashlib.sha256(text.encode()).hexdigest(),
        original209_sources_edited=False,new_JIT_functions=2,
        world_native_input_transform_frame_order_unverified=True,
        runtime_cache_and_CUBIN_ABI_require_fresh_witness=True,
        GPU_numerics_full_model_and_quality_unverified=True,
        no_raw_operands_or_snapshots_used=True,not_a_speed_measurement=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(original_AST_roundtrips=2,geometries=len(rows),new_JIT_functions=2,
        GPU_numerics_unverified=True,output=str(args.output)),indent=2))


if __name__=='__main__':main()
