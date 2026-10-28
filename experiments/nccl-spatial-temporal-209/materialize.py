"""Materialize the exact adopted215-source overlay without GPU/model loading."""
import argparse,hashlib,json,shutil
from pathlib import Path
from nccl_spatial_parallel209_sources import production_sources
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;base=here.parents[1]/'unifolm';root=a.output.resolve();assert not root.exists()
files,proof=production_sources(base,'world-spatial-temporal');assert proof==json.loads((here/'timing-proof.json').read_text())
root.mkdir()
for folder in ('src','configs','scripts/evaluation'):shutil.copytree(base/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):shutil.copyfile(base/name,root/name)
for name,text in files.items():(root/name).write_text(text)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('src/**/*.py')}
assert len(manifest)==215 and manifest==json.loads((here/'production-sources.json').read_text())
(root/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'nccl-entry-timing-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
for name,digest in json.loads((here/'production-assets.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print(root)
