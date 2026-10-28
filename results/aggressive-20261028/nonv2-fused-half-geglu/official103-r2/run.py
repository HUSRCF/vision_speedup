from pathlib import Path
import hashlib,importlib.util,json,sys
p=Path(__file__).with_name('assemble_clean_cases.py');spec=importlib.util.spec_from_file_location('fused_half_assembly',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources={case:Path(path)for case,path in {'unitree_z1_stackbox': '/dev/shm/team07-fused-half-candidate-meta-98wku9bd/official103-source-view', 'unitree_z1_dual_arm_stackbox': '/home/team07/benchmark-runs/unifolm/20261028T070032Z-case-dual-2184283', 'unitree_z1_dual_arm_stackbox_v2': '/dev/shm/team07-temporal221-split-full-ob4j7q2o/runs/20261028T185256Z-case-v2-nccl-3705772-ordinary-view', 'unitree_z1_dual_arm_cleanup_pencils': '/home/team07/benchmark-runs/unifolm/20261028T054953Z-case-pencils-2046809', 'unitree_g1_pack_camera': '/home/team07/benchmark-runs/unifolm/20261028T060330Z-case-g1-2083029'}.items()};previous=Path('/dev/shm/team07-temporal221-full-8bip_8ug/official103-validated')
assert set(sources)==set(m.CASES.values())
assert json.loads((previous/'result.json').read_text())['status']=='accepted'
before={}
for case,path in sources.items():
 assert m.valid_candidate(path,case),(case,str(path))
 for video in(path/'outputs'/case).rglob('*.mp4'):
  cached=previous/'outputs'/case/video.relative_to(path/'outputs'/case)
  h=hashlib.sha256(video.read_bytes()).hexdigest();assert cached.is_file()and not cached.is_symlink()and hashlib.sha256(cached.read_bytes()).hexdigest()==h
  assert cached.stat().st_dev==Path('/dev/shm/team07-fused-half-candidate-meta-98wku9bd').stat().st_dev
  m.REUSE_MAP[str(video)]=str(cached);before[str(video)]=h
assert len(before)==103
m.latest_runs=lambda root:sources
sys.argv=[str(p),'--output','/dev/shm/team07-fused-half-candidate-meta-98wku9bd/official103-validated'];m.main()
for case,path in sources.items():
 for video in(path/'outputs'/case).rglob('*.mp4'):
  target=Path('/dev/shm/team07-fused-half-candidate-meta-98wku9bd/official103-validated')/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert target.is_file()and not target.is_symlink()and hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(video.read_bytes()).hexdigest()==before[str(video)]
(Path('/dev/shm/team07-fused-half-candidate-meta-98wku9bd/official103-validated')/'immutable-video-links.json').write_text(json.dumps(dict(count=103,original_outputs_unchanged=True,ordinary_files=True,existing_official103_SHA_equal_hardlinks_only=True,hashes=before),indent=2)+'\n')
