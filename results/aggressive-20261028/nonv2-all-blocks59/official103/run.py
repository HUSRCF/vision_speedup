from pathlib import Path
import hashlib,importlib.util,json,sys
p=Path(__file__).with_name('assemble_clean_cases.py');spec=importlib.util.spec_from_file_location('blocks_assembly',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources={case:Path(path)for case,path in {'unitree_z1_stackbox': '/var/tmp/team07-nonv2-all-blocks63-tjdb597w/diagnostic/runs/20261028T221015Z-case-v2-nccl-3896254', 'unitree_z1_dual_arm_stackbox': '/home/team07/benchmark-runs/unifolm/20261028T070032Z-case-dual-2184283', 'unitree_z1_dual_arm_stackbox_v2': '/var/tmp/team07-private222-candidate-oih602vt/diagnostic/runs/20261028T205935Z-case-v2-nccl-3826784', 'unitree_z1_dual_arm_cleanup_pencils': '/home/team07/benchmark-runs/unifolm/20261028T054953Z-case-pencils-2046809', 'unitree_g1_pack_camera': '/home/team07/benchmark-runs/unifolm/20261028T060330Z-case-g1-2083029'}.items()};previous=Path('/var/tmp/team07-private222-candidate-oih602vt/diagnostic/official103-validated');assert json.loads((previous/'result.json').read_text())['status']=='accepted'
before={};selected=0
for case,path in sources.items():
 assert m.valid_candidate(path,case),(case,str(path))
 for video in(path/'outputs'/case).rglob('*.mp4'):
  cached=video if case=='unitree_z1_stackbox'else previous/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert cached.is_file()and not cached.is_symlink()and cached.stat().st_dev==Path('/var/tmp/team07-nonv2-all-blocks63-tjdb597w/diagnostic').stat().st_dev
  digest=hashlib.sha256(video.read_bytes()).hexdigest();assert digest==hashlib.sha256(cached.read_bytes()).hexdigest()
  m.REUSE_MAP[str(video)]=str(cached);before[str(video)]=digest;selected+=int(case=='unitree_z1_stackbox')
assert len(before)==103 and selected==25
m.latest_runs=lambda root:sources
sys.argv=[str(p),'--output','/var/tmp/team07-nonv2-all-blocks63-tjdb597w/diagnostic/official103-validated'];m.main()
for case,path in sources.items():
 for video in(path/'outputs'/case).rglob('*.mp4'):
  target=Path('/var/tmp/team07-nonv2-all-blocks63-tjdb597w/diagnostic/official103-validated')/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert target.is_file()and not target.is_symlink()and hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(video.read_bytes()).hexdigest()==before[str(video)]
(Path('/var/tmp/team07-nonv2-all-blocks63-tjdb597w/diagnostic/official103-validated')/'immutable-video-links.json').write_text(json.dumps(dict(count=103,ordinary_files=True,original_outputs_unchanged=True,selected25_from_new_timed_run=True,other78_SHA_equal_prior_accepted=True,no_GPU_rerun_or_timer_rewrite=True,hashes=before),indent=2)+'\n')
