"""Rebuild G1 two-refresh prediction with exact-grid68 mathematics and full binding contracts."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(13,23),default=23);p.add_argument('--prediction-cache-parent',type=Path);p.add_argument('--case',choices=['g1'],default='g1');p.add_argument('--native-handles',type=Path)
    p.add_argument('--storage-config',type=Path);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    devices=a.gpus.split(',');assert len(devices)==len(set(devices))==2 and set(devices)<=set('0123')
    repo=Path(__file__).resolve().parents[1];base=repo/'experiments/nonv2-vae-all-display/unifolm'
    frozen=repo/'experiments/g1-prediction70'/f'deep{a.depth}-r1';overlay=frozen/'production-overlay';manifest=json.loads((overlay/'nccl-parallel-production-sources.json').read_text());assets=json.loads((frozen/'asset-hashes.json').read_text())
    directory=a.workdir.resolve();assert not directory.exists();directory.mkdir(parents=True,mode=0o700)
    source=directory/'source';source.mkdir();diagnostic=directory/'diagnostic';diagnostic.mkdir();(diagnostic/'runs').mkdir();(diagnostic/'tmp').mkdir()
    for name in('src','configs','scripts/evaluation'):shutil.copytree(base/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):shutil.copyfile(base/name,source/name)
    shutil.copytree(overlay,source,dirs_exist_ok=True)
    native=(a.native_handles or repo/'unifolm/src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so').resolve()
    assert native.is_file() and hashlib.sha256(native.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
    target=source/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
    target.parent.mkdir(exist_ok=True);target.symlink_to(native)

    (source/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(manifest)==70
    assert set(manifest)=={str(f.relative_to(source))for f in source.glob('src/**/*.py')}
    for name,digest in dict(manifest,**assets).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    guards=json.loads((frozen/'guard-source-hashes.json').read_text())
    for name,digest in guards.items():
        assert hashlib.sha256((frozen/'guard'/name).read_bytes()).hexdigest()==digest;shutil.copyfile(frozen/'guard'/name,diagnostic/name)
    shutil.copyfile(frozen/'guard/flags.json',diagnostic/'flags.json')
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,source_count=len(manifest),source=str(source),case=a.case)));return
    assert a.storage_config,'The qualified private-tmpfs guard requires --storage-config'
    config=json.loads(a.storage_config.read_text());qualified=json.loads((repo/'experiments/private-nccl-tmpfs/qualified-adapter.json').read_text())
    assert config['cpu_cross_process_passed']and config['source_sha256']==qualified['source_sha256']
    library=Path(config['preload']);shared=Path(config['shared_root']);assert hashlib.sha256(library.read_bytes()).hexdigest()==config['preload_sha256']
    assert shared.is_dir()and not shared.is_symlink()and shared.stat().st_uid==os.getuid()and shared.stat().st_mode&0o777==0o700
    assert source.stat().st_dev==shared.stat().st_dev,'Use the same private tmpfs for source, metadata and NCCL backing'
    chosen=a.cache_dir or os.environ.get('NONV2_FFN_CACHE_DIR')
    if chosen:
        cache=Path(chosen).resolve();assert cache.is_dir()
    else:
        parent=Path.home()/'.tmp';parent.mkdir(exist_ok=True)
        assert shutil.disk_usage(parent).free>64*2**20,'Need executable space for new FFN JIT shapes, or pass an existing --cache-dir'
        cache=Path(tempfile.mkdtemp(prefix='team07-nonv2-FFN-shapes-',dir=parent))
    prediction_parent=(a.prediction_cache_parent or cache).resolve()
    assert prediction_parent.is_dir()and not os.statvfs(prediction_parent).f_flag&os.ST_NOEXEC
    prediction_cache=Path(tempfile.mkdtemp(prefix='team07-g1-predict70-',dir=prediction_parent));prediction_cache.chmod(0o700)
    for rank in(0,1):
        local=prediction_cache/('rank'+str(rank));local.mkdir(mode=0o700)
        for name in('gpu','exec'):(local/name).mkdir(mode=0o700)
    env={k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('RANK','LOCAL_RANK','WORLD_SIZE','TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(TEAM07_PREDICT_CACHE=str(prediction_cache))
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],LD_PRELOAD=str(library.resolve()),TEAM07_NCCL_SHM_ROOT=str(shared.resolve()),CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,CANONICAL_ROOT=str(source),NONV2_FFN_CACHE_DIR=str(cache),FULL_RUN_PARENT=str(diagnostic/'runs'),PYTHONPATH=str(diagnostic),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(diagnostic/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs',TRIAL_N_ITER='1'if a.smoke else'original')
    subprocess.run([sys.executable,str(diagnostic/'check_nonv2_ffn59_cpu.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src')),check=True)
    subprocess.run([sys.executable,str(diagnostic/'check_g1_prediction70_cpu.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src')),check=True)
    subprocess.run([sys.executable,str(diagnostic/'retry_nccl_full221.py')],cwd=diagnostic,env=env,check=True)
    print((diagnostic/'nccl-full-report.json').read_text())


if __name__=='__main__':main()
