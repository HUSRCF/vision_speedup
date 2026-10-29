"""Rebuild qualified226 math and run a guarded four-GPU original V2 evaluator."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--workdir',type=Path,required=True);p.add_argument('--storage-config',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    gpus=a.gpus.split(',');assert len(gpus)==len(set(gpus))==4 and set(gpus)==set('0123')
    repo=Path(__file__).resolve().parents[1];frozen=repo/'experiments/four-way226/full-candidate-r2'
    directory=a.workdir.resolve();assert not directory.exists()
    subprocess.run([sys.executable,str(repo/'reproduce/run_v2_world_co225.py'),'--gpus',','.join(gpus[:2]),'--workdir',str(directory),'--prepare-only'],check=True)
    source=directory/'source';diagnostic=directory/'diagnostic'
    shutil.copytree(frozen/'production-overlay',source,dirs_exist_ok=True)
    shutil.copyfile(frozen/'timing-proof.json',source/'nccl-entry-timing-proof.json')
    manifest=json.loads((frozen/'production-sources.json').read_text());assets=json.loads((frozen/'timing-asset-hashes.json').read_text())
    assert len(manifest)==226 and set(manifest)=={str(f.relative_to(source)) for f in source.glob('src/**/*.py')}
    for n,h in dict(manifest,**assets).items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
    (source/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for n,h in json.loads((frozen/'guard-source-hashes.json').read_text()).items():
        assert hashlib.sha256((frozen/n).read_bytes()).hexdigest()==h;shutil.copyfile(frozen/n,diagnostic/n)
    shutil.copyfile(frozen/'flags.json',diagnostic/'flags.json')
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,source_count=226,source=str(source))));return
    assert a.storage_config,'The qualified guard requires --storage-config'
    config=json.loads(a.storage_config.read_text());qualified=json.loads((repo/'experiments/private-nccl-tmpfs/qualified-adapter.json').read_text())
    assert config['cpu_cross_process_passed'] and config['source_sha256']==qualified['source_sha256']
    library=Path(config['preload']);shared=Path(config['shared_root'])
    assert library.is_file() and hashlib.sha256(library.read_bytes()).hexdigest()==config['preload_sha256']
    assert shared.is_dir() and not shared.is_symlink() and shared.stat().st_uid==os.getuid() and shared.stat().st_mode&0o777==0o700
    assert source.stat().st_dev==shared.stat().st_dev
    parent=Path(os.environ.get('TEAM07_EXECUTABLE_CACHE_PARENT',str(Path.home()/'.tmp')));parent.mkdir(exist_ok=True)
    cache=os.environ.get('FOUR_WAY226_CACHE_ROOT')
    if cache:cache=Path(cache).resolve();assert cache.is_dir()
    else:
        assert shutil.disk_usage(parent).free>512*2**20,'Need executable cache space or an explicit existing cache root'
        cache=Path(tempfile.mkdtemp(prefix='team07-four-way226-kernels-',dir=parent))
        for n in ('initial','temporal','decision','general'):(cache/n).mkdir()
    assert all((cache/n).is_dir() for n in ('initial','temporal','decision','general'))
    env={k:v for k,v in os.environ.items() if not k.startswith(('UNIFOLM_','NCCL_')) and k not in ('RANK','LOCAL_RANK','WORLD_SIZE','TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],LD_PRELOAD=str(library.resolve()),TEAM07_NCCL_SHM_ROOT=str(shared.resolve()),CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,CANONICAL_ROOT=str(source),INIT_FFN_CACHE_DIR=str(cache/'initial'),TEMPORAL_BLOCK_CACHE_DIR=str(cache/'temporal'),DECISION_TEMPORAL_CACHE_DIR=str(cache/'decision'),FOUR_WAY_KERNEL_CACHE_DIR=str(cache/'general'),FULL_RUN_PARENT=str(diagnostic/'runs'),PYTHONPATH=str(diagnostic),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(diagnostic/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs',TRIAL_N_ITER='original')
    cpu=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src'))
    for name in ('check_four_way226_lifecycle_cpu.py','check_four_way226_cpu.py'):
        command=[sys.executable,str(diagnostic/name)]
        if 'lifecycle' in name:command.append(str(source/'src/unifolm_wma/utils/nccl_parallel_runtime209.py'))
        subprocess.run(command,cwd=diagnostic,env=cpu,check=True)
    subprocess.run([sys.executable,str(diagnostic/'retry_nccl_full221.py')],cwd=diagnostic,env=env,check=True)
    print((diagnostic/'nccl-full-report.json').read_text())

if __name__=='__main__':main()
