from pathlib import Path
import hashlib,importlib.util,json,sys
p=Path(__file__).with_name('assemble_clean_cases.py');spec=importlib.util.spec_from_file_location('initial222_assembly',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources={case:Path(path)for case,path in {'unitree_z1_stackbox': '/dev/shm/team07-fused-half-candidate-meta-98wku9bd/official103-source-view', 'unitree_z1_dual_arm_stackbox': '/home/team07/benchmark-runs/unifolm/20261028T070032Z-case-dual-2184283', 'unitree_z1_dual_arm_stackbox_v2': '/var/tmp/team07-private222-candidate-oih602vt/diagnostic/runs/20261028T205935Z-case-v2-nccl-3826784', 'unitree_z1_dual_arm_cleanup_pencils': '/home/team07/benchmark-runs/unifolm/20261028T054953Z-case-pencils-2046809', 'unitree_g1_pack_camera': '/home/team07/benchmark-runs/unifolm/20261028T060330Z-case-g1-2083029'}.items()};assert set(sources)==set(m.CASES.values())
before={}
for case,path in sources.items():
 assert m.valid_candidate(path,case),(case,str(path))
 for video in(path/'outputs'/case).rglob('*.mp4'):
  assert video.is_file()and not video.is_symlink()
  before[str(video)]=hashlib.sha256(video.read_bytes()).hexdigest()
assert len(before)==103
m.latest_runs=lambda root:sources
sys.argv=[str(p),'--output','/var/tmp/team07-private222-candidate-oih602vt/diagnostic/official103-validated'];m.main()
for case,path in sources.items():
 for video in(path/'outputs'/case).rglob('*.mp4'):
  target=Path('/var/tmp/team07-private222-candidate-oih602vt/diagnostic/official103-validated')/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert target.is_file()and not target.is_symlink()and hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(video.read_bytes()).hexdigest()==before[str(video)]
(Path('/var/tmp/team07-private222-candidate-oih602vt/diagnostic/official103-validated')/'immutable-video-copies.json').write_text(json.dumps(dict(count=103,ordinary_files=True,original_outputs_unchanged=True,SHA_verified_same_FS_links_or_cross_FS_copies=True,hashes=before),indent=2)+'\n')
