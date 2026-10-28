"""Materialize the frozen223 V2 experiment and run its original timed evaluator."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--storage-config',type=Path);p.add_argument('--decision-cache-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--variant',choices=['decision-temporal223'],default='decision-temporal223');a=p.parse_args()
    pair=a.gpus.split(',');assert len(pair)==len(set(pair))==2 and set(pair)<=set('0123')
    repo=Path(__file__).resolve().parents[1];frozen=repo/'experiments/decision-temporal-222/full-candidate-r1'
    directory=a.workdir.resolve();assert not directory.exists();directory.mkdir(parents=True,mode=0o700)
    source=directory/'source';diagnostic=directory/'diagnostic';diagnostic.mkdir();(diagnostic/'runs').mkdir();(diagnostic/'tmp').mkdir()
    subprocess.run([sys.executable,str(repo/'experiments/initial-boundary-221/production222/materialize.py'),'--output',str(source)],check=True)
    shutil.copytree(frozen/'production-overlay',source,dirs_exist_ok=True)
    shutil.copyfile(frozen/'timing-proof.json',source/'nccl-entry-timing-proof.json')
    manifest=json.loads((frozen/'production-sources.json').read_text());assets=json.loads((frozen/'timing-asset-hashes.json').read_text())
    assert len(manifest)==223 and set(manifest)=={str(f.relative_to(source))for f in source.glob('src/**/*.py')}
    for n,h in dict(manifest,**assets).items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
    (source/'nccl-parallel-production-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for n,h in json.loads((frozen/'guard-source-hashes.json').read_text()).items():
        assert hashlib.sha256((frozen/n).read_bytes()).hexdigest()==h;shutil.copyfile(frozen/n,diagnostic/n)
    shutil.copyfile(frozen/'flags.json',diagnostic/'flags.json')
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,source_count=223,source=str(source))));return
    assert a.storage_config,'The qualified guard requires --storage-config'
    config=json.loads(a.storage_config.read_text());qualified=json.loads((repo/'experiments/private-nccl-tmpfs/qualified-adapter.json').read_text())
    assert config['cpu_cross_process_passed']and config['source_sha256']==qualified['source_sha256']
    library=Path(config['preload']);shared=Path(config['shared_root'])
    assert library.is_file()and hashlib.sha256(library.read_bytes()).hexdigest()==config['preload_sha256']
    assert shared.is_dir()and not shared.is_symlink()and shared.stat().st_uid==os.getuid()and shared.stat().st_mode&0o777==0o700
    assert source.stat().st_dev==shared.stat().st_dev
    parent=Path(os.environ.get('TEAM07_EXECUTABLE_CACHE_PARENT',str(Path.home()/'.tmp')));parent.mkdir(exist_ok=True)
    def cache(option,prefix):
        if option:
            path=Path(option).resolve();assert path.is_dir();return str(path)
        assert shutil.disk_usage(parent).free>64*2**20,'Need executable cache space or an explicit existing cache'
        return tempfile.mkdtemp(prefix=prefix,dir=parent)
    initial=cache(os.environ.get('INIT_FFN_CACHE_DIR'),'team07-initial220-')
    temporal=cache(os.environ.get('TEMPORAL_BLOCK_CACHE_DIR'),'team07-temporal221-')
    decision=cache(a.decision_cache_dir or os.environ.get('DECISION_TEMPORAL_CACHE_DIR'),'team07-decision223-')
    env={k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('RANK','LOCAL_RANK','WORLD_SIZE','TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],LD_PRELOAD=str(library.resolve()),TEAM07_NCCL_SHM_ROOT=str(shared.resolve()),CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,CANONICAL_ROOT=str(source),INIT_FFN_CACHE_DIR=initial,TEMPORAL_BLOCK_CACHE_DIR=temporal,DECISION_TEMPORAL_CACHE_DIR=decision,FULL_RUN_PARENT=str(diagnostic/'runs'),PYTHONPATH=str(diagnostic),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(diagnostic/'tmp'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs',TRIAL_N_ITER='1'if a.smoke else'original')
    flags=json.loads((diagnostic/'flags.json').read_text())
    subprocess.run([sys.executable,str(diagnostic/'check_nccl_parallel209_cpu.py'),str(source/'src/unifolm_wma/utils/nccl_parallel_runtime209.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src'),CPU_TEST_POLICY=flags['UNIFOLM_NCCL_PARALLEL'],CPU_TEST_GPUS=a.gpus),check=True)
    subprocess.run([sys.executable,str(diagnostic/'retry_nccl_full221.py')],cwd=diagnostic,env=env,check=True)
    print((diagnostic/'nccl-full-report.json').read_text())

if __name__=='__main__':main()
