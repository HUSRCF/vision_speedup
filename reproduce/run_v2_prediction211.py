"""Rebuild one frozen two-refresh V2 temporal-prediction experiment and run its original guarded workload."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--trial',required=True);p.add_argument('--gpu',choices=tuple('0123'),required=True)
    p.add_argument('--workdir',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--native-handles',type=Path);p.add_argument('--exec-cache-dir',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];assert Path(a.trial).name==a.trial and a.trial not in('.','..')
    frozen=repo/'experiments/v2-prediction211'/a.trial;metadata=json.loads((frozen/'trial.json').read_text());base=repo/metadata['base']
    manifest=json.loads((frozen/'production-overlay/crossstep211-production-sources.json').read_text());assets=json.loads((frozen/'asset-hashes.json').read_text())
    root=a.workdir.resolve();assert not root.exists();root.mkdir(parents=True,mode=0o700)
    source=root/'source';source.mkdir();diagnostic=root/'diagnostic';diagnostic.mkdir();(diagnostic/'tmp').mkdir()
    def copy(src,dst):
        if src.endswith('.so'):Path(dst).symlink_to(Path(src).resolve())
        else:shutil.copyfile(src,dst)
        return dst
    for name in('src','configs','scripts/evaluation'):shutil.copytree(base/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),copy_function=copy)
    for name in('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):shutil.copyfile(base/name,source/name)
    shutil.copytree(frozen/'production-overlay',source,dirs_exist_ok=True)
    if a.native_handles:
        assert hashlib.sha256(a.native_handles.read_bytes()).hexdigest()=='ab2f19b88cb9af02b42fb5fdcb39da49e062346fd30794be45fb81e12cb1b927'
        target=source/'src/unifolm_wma/utils/native-graph-handles/team07_cuda_graph_handles.so'
        assert target.is_symlink();target.unlink();target.symlink_to(a.native_handles.resolve())
    assert len(manifest)==metadata['source_count']
    assert set(manifest)=={str(f.relative_to(source))for f in(source/'src').rglob('*.py')}
    for name,digest in dict(manifest,**assets).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    for name,digest in json.loads((frozen/'guard-source-hashes.json').read_text()).items():
        content=(frozen/'guard'/name).read_bytes();assert hashlib.sha256(content).hexdigest()==digest; (diagnostic/name).write_bytes(content)
    shutil.copyfile(frozen/'guard/flags.json',diagnostic/'flags.json')
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,source_count=len(manifest),source=str(source),trial=a.trial)));return
    assert 'interrupted'not in a.trial,'This frozen trial was stopped for a stage-label bug; use its corrected revision.'
    cache=root/'cache';cache.mkdir();(cache/'gpu').mkdir();executable=(a.exec_cache_dir or cache/'exec').resolve()
    assert not executable.exists();executable.mkdir(parents=True,mode=0o700)
    assert not os.statvfs(executable).f_flag&os.ST_NOEXEC,'The new Triton launcher cache needs an executable mount.'
    env={k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','RANK','LOCAL_RANK','WORLD_SIZE','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],CUDA_VISIBLE_DEVICES=a.gpu,MONITORED_GPU=a.gpu,CANONICAL_ROOT=str(source),CROSSSTEP_FLAGS_FILE=str(diagnostic/'flags.json'),PYTHONPATH=str(diagnostic)+':'+str(source/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TMPDIR=str(diagnostic/'tmp'),TRITON_CACHE_DIR=str(cache/'gpu'),EXPERIMENT_EXEC_CACHE=str(executable),TRITON_CACHE_MANAGER='split_experiment_triton_cache:ReadWarmSplitCache',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='123',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs')
    subprocess.run([sys.executable,str(diagnostic/'guard_crossstep211.py'),'--gpu',a.gpu,'--entry','run_crossstep_full211.py'],cwd=diagnostic,env=env,check=True)
    report=json.loads((diagnostic/'guard-report.json').read_text());row=report['diagnostic']['case_result']
    print(json.dumps(dict(clean=report['clean'],case=row['case'],process_wall_s=row['process_wall_s'],psnr_db=row['psnr_db'],quality_passed=row['psnr_passed'],official_adopted=False)))


if __name__=='__main__':main()
