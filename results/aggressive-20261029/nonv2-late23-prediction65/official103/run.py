from pathlib import Path
import hashlib,importlib.util,json,sys
p=Path(__file__).with_name('assemble_clean_cases.py');spec=importlib.util.spec_from_file_location('other63_assembly',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sources={case:Path(path)for case,path in {'unitree_z1_stackbox': '/var/tmp/team07-nonv2-predict65-ps9786m_/diagnostic/runs/20261029T095627Z-case-v2-nccl-912747', 'unitree_z1_dual_arm_stackbox': '/var/tmp/team07-nonv2-other-prediction65-dual-ou2ydhxe/diagnostic/runs/20261029T100849Z-case-v2-nccl-930901', 'unitree_z1_dual_arm_stackbox_v2': '/var/tmp/team07-world-co225-candidate-cdhnv0dt/diagnostic/runs/20261028T233913Z-case-v2-nccl-3980330', 'unitree_z1_dual_arm_cleanup_pencils': '/var/tmp/team07-nonv2-other-prediction65-pencils-4dxcxljn/diagnostic/runs/20261029T113612Z-case-v2-nccl-1031924', 'unitree_g1_pack_camera': '/var/tmp/team07-nonv2-native-shared68-g1-gbv_kjtx/diagnostic/runs/20261029T061359Z-case-v2-nccl-522189'}.items()};previous=Path('/var/tmp/team07-nonv2-other-prediction65-dual-ou2ydhxe/diagnostic/official103-bundle-validated');assert json.loads((previous/'result.json').read_text())['status']=='accepted'
before={};selected=0
for case,path in sources.items():
 assert m.valid_candidate(path,case),(case,str(path))
 for video in(path/'outputs'/case).rglob('*.mp4'):
  cached=video if case in ['unitree_z1_dual_arm_cleanup_pencils']else previous/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert cached.is_file()and not cached.is_symlink()and cached.stat().st_dev==Path('/var/tmp/team07-nonv2-other-prediction65-pencils-4dxcxljn/diagnostic').stat().st_dev
  digest=hashlib.sha256(video.read_bytes()).hexdigest();assert digest==hashlib.sha256(cached.read_bytes()).hexdigest()
  m.REUSE_MAP[str(video)]=str(cached);before[str(video)]=digest;selected+=int(case in ['unitree_z1_dual_arm_cleanup_pencils'])
assert len(before)==103 and selected==17
m.latest_runs=lambda root:sources
sys.argv=[str(p),'--output','/var/tmp/team07-nonv2-other-prediction65-pencils-4dxcxljn/diagnostic/official103-bundle-validated'];m.main()
for case,path in sources.items():
 for video in(path/'outputs'/case).rglob('*.mp4'):
  target=Path('/var/tmp/team07-nonv2-other-prediction65-pencils-4dxcxljn/diagnostic/official103-bundle-validated')/'outputs'/case/video.relative_to(path/'outputs'/case)
  assert target.is_file()and not target.is_symlink()and hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(video.read_bytes()).hexdigest()==before[str(video)]
(Path('/var/tmp/team07-nonv2-other-prediction65-pencils-4dxcxljn/diagnostic/official103-bundle-validated')/'immutable-video-links.json').write_text(json.dumps(dict(count=103,ordinary_files=True,original_outputs_unchanged=True,selected_actual_timed_videos=17,prior_accepted_SHA_equal_videos=86,selected_cases=['unitree_z1_dual_arm_cleanup_pencils'],no_GPU_rerun_or_timer_rewrite=True,hashes=before),indent=2)+'\n')
