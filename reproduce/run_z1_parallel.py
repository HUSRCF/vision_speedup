"""Materialize and run the frozen wider-block Z1 variant under its original timer."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--variant',choices=['ffn61','temporal62','all63'],default='all63')
    p.add_argument('--storage-config',type=Path);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    devices=a.gpus.split(',');assert len(devices)==len(set(devices))==2 and set(devices)<=set('0123')
    repo=Path(__file__).resolve().parents[1];base=repo/'experiments/nonv2-vae-all-display/unifolm'
    folders={'ffn61':'experiments/nonv2-ffn-59/full61-r1','temporal62':'experiments/nonv2-temporal-blocks59/full62-r1','all63':'experiments/nonv2-all-blocks59/full63-r1'}
    frozen=repo/folders[a.variant];manifest=json.loads((frozen/'production-sources.json').read_text());assets=json.loads((frozen/'timing-asset-hashes.json').read_text())
    directory=a.workdir.resolve();assert not directory.exists();directory.mkdir(parents=True,mode=0o700)
    source=directory/'source';source.mkdir();diagnostic=directory/'diagnostic';diagnostic.mkdir();(diagnostic/'runs').mkdir();(diagnostic/'tmp').mkdir()
    for name in('src','configs','scripts/evaluation'):shutil.copytree(base/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):shutil.copyfile(base/name,source/name)
    shutil.copytree(frozen/'production-overlay',source,dirs_exist_ok=True)
    shutil.copyfile(frozen/'timing-proof.json',source/'nccl-entry-timing-proof.json')
    (source/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(manifest)=={'ffn61':61,'temporal62':62,'all63':63}[a.variant]
    assert set(manifest)=={str(f.relative_to(source))for f in source.glob('src/**/*.py')}
    for name,digest in dict(manifest,**assets).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    guards=json.loads((frozen/'guard-source-hashes.json').read_text())
    for name,digest in guards.items():
        assert hashlib.sha256((frozen/name).read_bytes()).hexdigest()==digest;shutil.copyfile(frozen/name,diagnostic/name)
    shutil.copyfile(frozen/'flags.json',diagnostic/'flags.json')
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,source_count=len(manifest),source=str(source),variant=a.variant)));return
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
        cache=Path(tempfile.mkdtemp(prefix='team07-z1-FFN-shapes-',dir=parent))
    env={k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('RANK','LOCAL_RANK','WORLD_SIZE','TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],LD_PRELOAD=str(library.resolve()),TEAM07_NCCL_SHM_ROOT=str(shared.resolve()),CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,CANONICAL_ROOT=str(source),NONV2_FFN_CACHE_DIR=str(cache),FULL_RUN_PARENT=str(diagnostic/'runs'),PYTHONPATH=str(diagnostic),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(diagnostic/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs',TRIAL_N_ITER='1'if a.smoke else'original')
    subprocess.run([sys.executable,str(diagnostic/'check_nonv2_ffn59_cpu.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src')),check=True)
    subprocess.run([sys.executable,str(diagnostic/'retry_nccl_full221.py')],cwd=diagnostic,env=env,check=True)
    print((diagnostic/'nccl-full-report.json').read_text())


if __name__=='__main__':main()
