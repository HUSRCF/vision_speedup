"""Materialize the experimental59-source tree; no GPU imports or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    base=here.parent/'nonv2-vae-all-display/unifolm'
    target=args.output.resolve()
    assert not target.exists(), 'Choose a new output directory'
    baseline=json.loads((base/'nonv2-aggressive-production-sources.json').read_text())
    manifest=json.loads((here/'production-sources.json').read_text())
    assert len(manifest)==len(baseline)==59
    for name,digest in baseline.items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==digest
    changed={name for name in baseline if baseline[name]!=manifest[name]}
    assert changed=={'src/unifolm_wma/utils/nonv2_nonfeedback_vae.py'}
    shutil.copytree(base,target,ignore=shutil.ignore_patterns('__pycache__'))
    for name in changed:
        raw=(here/'production-overlay'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest[name]
        (target/name).write_bytes(raw)
    (target/'all-feedback-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'all-feedback-flags.json').write_bytes((here/'flags.json').read_bytes())
    assert {str(p.relative_to(target)) for p in (target/'src').rglob('*.py')}==set(manifest)
    for name,digest in manifest.items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(source_count=59,changed_files=sorted(changed),output=str(target),experimental=True)))


if __name__=='__main__':main()
