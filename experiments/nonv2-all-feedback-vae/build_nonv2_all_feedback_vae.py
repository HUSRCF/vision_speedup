"""Opt-in Z1 experiment; keep all master weights and the existing cache owner."""
import ast
from pathlib import Path

HELPER = 'src/unifolm_wma/utils/nonv2_nonfeedback_vae.py'


def candidate_source(baseline):
    text = baseline.replace(
        '"""Batch only display frames; retain the original decoder for feedback frames."""',
        '"""Experimental Z1 all-frame Half decoding; full closed-loop quality required."""')
    old = "assert scope in ('world','all-display') and stage in ('decision','simulation')"
    new = "assert scope in ('world','all-display','all-feedback') and stage in ('decision','simulation')\n    if scope=='all-feedback':assert os.environ.get('UNIFOLM_CASE')=='unitree_z1_stackbox'"
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = "    if stage=='decision':\n        result=owner.fast"
    new = "    if stage=='decision' or scope=='all-feedback':\n        result=owner.fast"
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = "owner.calls+=1;records.append(dict(stage=stage,display_only=True,partitions=[(False,0,16)],flags=before))"
    new = "owner.calls+=1;records.append(dict(stage=stage,display_only=stage=='decision',execution_steps=execution_steps,feedback_dtype='fp16' if stage=='simulation' else None,partitions=[(False,0,16)],flags=before))"
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = 'native_feedback_precision_and_flags_preserved=True,all_setup_inside_process_wall=True'
    new = "native_feedback_precision_and_flags_preserved=not any(r.get('feedback_dtype')=='fp16' for r in records),all_setup_inside_process_wall=True"
    assert text.count(old) == 1
    text = text.replace(old, new)
    def owner(value):
        return next(ast.dump(n, include_attributes=False) for n in ast.parse(value).body
                    if isinstance(n, ast.ClassDef) and n.name == 'Decoder')
    assert owner(text) == owner(baseline)
    ast.parse(text)
    return text


if __name__ == '__main__':
    import sys
    print(candidate_source(Path(sys.argv[1]).read_text()), end='')
