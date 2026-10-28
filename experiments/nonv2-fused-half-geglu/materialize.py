"""Materialize the opt-in60-source candidate without GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    here=Path(__file__).resolve().parent;base=here.parent/'nonv2-vae-all-display/unifolm'
    target=args.output.resolve();assert not target.exists()
    original=json.loads((base/'nonv2-aggressive-production-sources.json').read_text());manifest=json.loads((here/'production-sources.json').read_text())
    assert len(original)==59 and len(manifest)==60
    assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in original.items())
    changed={n for n,h in manifest.items()if original.get(n)!=h}
    assert changed=={'src/unifolm_wma/modules/attention.py','src/unifolm_wma/utils/nonv2_fused_half_geglu.py'}
    shutil.copytree(base,target,ignore=shutil.ignore_patterns('__pycache__'))
    for n in changed:
        p=target/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((here/'production-overlay'/n).read_bytes())
    (target/'fused-half-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'fused-half-flags.json').write_bytes((here/'flags.json').read_bytes())
    assert set(manifest)=={str(p.relative_to(target))for p in(target/'src').rglob('*.py')}
    assert all(hashlib.sha256((target/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
    print(json.dumps(dict(source_count=60,adopted_Z1_source60=True,output=str(target),changed_files=sorted(changed))))


if __name__=='__main__':main()
