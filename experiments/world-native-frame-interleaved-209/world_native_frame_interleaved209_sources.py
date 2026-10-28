"""Hypothesis to test: native Winograd columns interleave the 16 frames.

This is an experimental mapping, not a proven native layout. Each effective
column selects frames 0..7 in a group of16. Only GPU convolution checks can
establish whether the native transform really uses this ordering.
"""
import ast,copy,hashlib
from world_native_frame_prefix209_sources import sources as prefix_sources


def sources(base,vector8=False):
    text,original_proof=prefix_sources(base);tree=ast.parse(text);proof=[]
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef):continue
        assert fn.name.endswith('_frame_prefix');original=copy.deepcopy(fn)
        fn.name=fn.name.replace('_frame_prefix','_frame_interleaved')
        old=next(n for n in fn.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='cols')
        position=fn.body.index(old)
        new=ast.parse('logical_cols=col_tile*BN+tl.arange(0,BN)\ncols=(logical_cols//8)*16+logical_cols%8').body
        if vector8:new+=ast.parse('cols=tl.max_contiguous(tl.multiple_of(cols,8),8)').body
        fn.body[position:position+1]=new
        fn.body.insert(1,ast.parse('tl.static_assert(COLS%16==0)').body[0])
        restored=copy.deepcopy(fn);restored.name=original.name;restored.body.pop(1)
        restored.body[position:position+len(new)]=[copy.deepcopy(old)]
        assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
        proof.append(dict(candidate=fn.name,restores_contiguous_prefix_AST_exactly=True,
            physical_column_formula='(logical_col//8)*16+logical_col%8',
            all_original_arithmetic_and_pointer_ABI_preserved=True,vector8_contiguity_hint=vector8,
            native_frame_order_requires_separate_GPU_evidence=True))
    ast.fix_missing_locations(tree);generated=ast.unparse(tree)+'\n';ast.parse(generated)
    assert len(proof)==2
    for bn in (64,128):
        for tile in range(40):
            values=[(v//8)*16+v%8 for v in range(tile*bn,(tile+1)*bn)]
            assert all(values[i]%8==0 and values[i:i+8]==list(range(values[i],values[i]+8)) for i in range(0,bn,8))
    return generated,dict(original_prefix_proof=original_proof,interleaved_proof=proof,
        SHA256=hashlib.sha256(generated.encode()).hexdigest(),vector8_CPU_alignment_contiguity_checks=80,
        vector8=vector8,no_GPU_or_native_frame_layout_claim=True)
