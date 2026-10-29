"""Rebuild and optionally run one unadopted, complete single-GPU V2 trial."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path

def build(repo,output,revision):
    package=repo/f'experiments/crossstep211/full-r{revision}'
    identity=json.loads((package/'identity.json').read_text())
    assert not output.exists(),'Use a fresh output directory'
    output.mkdir(parents=True);source=output/'source';diagnostic=output/'diagnostic';source.mkdir();diagnostic.mkdir();(diagnostic/'tmp').mkdir()
    for folder in ('src','configs','scripts/evaluation'):
        shutil.copytree(repo/'unifolm'/folder,source/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('scripts/run_compute_benchmark.sh','scripts/team07_env.sh'):
        shutil.copyfile(repo/'unifolm'/name,source/name)
    for path in (package/'production-overlay').rglob('*'):
        if path.is_file():
            target=source/path.relative_to(package/'production-overlay');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    for name,expected in identity['assets'].items():
        path=package/'timing-assets'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        shutil.copyfile(path,source/name)
    for name,expected in identity['scripts'].items():
        path=package/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==expected;shutil.copyfile(path,diagnostic/name)
    shutil.copyfile(package/'flags.json',diagnostic/'flags.json')
    actual={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()for p in source.glob('src/**/*.py')}
    assert actual==identity['manifest'] and len(actual)==211
    (source/'crossstep211-production-sources.json').write_text(json.dumps(actual,indent=2)+'\n')
    print('CROSSSTEP211_PREPARED '+json.dumps(dict(revision=revision,source_count=211,policy=identity['policy'],official_adopted=False)),flush=True)
    return source,diagnostic

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--revision',type=int,choices=(3,4,5,6,7,8),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--gpu',choices=tuple('0123'),default='0');p.add_argument('--cache-dir',type=Path);p.add_argument('--python',type=Path,default=Path.home()/'.venv/bin/python');a=p.parse_args()
    if not a.prepare_only and a.cache_dir is None:p.error('--cache-dir is required on an executable filesystem')
    repo=Path(__file__).resolve().parents[1];source,diagnostic=build(repo,a.output.resolve(),a.revision)
    if a.prepare_only:return
    assert a.python.is_file()
    row=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid,memory,util=[x.strip()for x in row]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    assert int(memory)<=16 and int(util)<=1 and not any(x.split(',')[0].strip()==uuid for x in apps.splitlines()),'GPU must currently be idle; no process is evicted'
    cache=a.cache_dir.resolve();cache.mkdir(parents=True,exist_ok=False);(cache/'exec').mkdir();(cache/'gpu').mkdir()
    env={k:v for k,v in os.environ.items()if not k.startswith(('UNIFOLM_','NCCL_'))and k not in('TRITON_CACHE_DIR','TRITON_CACHE_MANAGER','RANK','LOCAL_RANK','WORLD_SIZE')}
    env.update(PATH=str(a.python.parent)+':'+env['PATH'],CANONICAL_ROOT=str(source),CROSSSTEP_FLAGS_FILE=str(diagnostic/'flags.json'),PYTHONPATH=str(diagnostic)+':'+str(source/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=a.gpu,MONITORED_GPU=a.gpu,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TMPDIR=str(diagnostic/'tmp'),TRITON_CACHE_DIR=str(cache/'gpu'),EXPERIMENT_EXEC_CACHE=str(cache/'exec'),TRITON_CACHE_MANAGER='split_experiment_triton_cache:ReadWarmSplitCache',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='123',HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs')
    # Guard rechecks strict idle admission, takes existing team07 flock locks,
    # records telemetry, and rejects a foreign context through the whole run.
    subprocess.run([str(a.python),str(diagnostic/'guard_crossstep211.py'),'--gpu',a.gpu,'--entry','run_crossstep_full211.py'],cwd=diagnostic,env=env,check=True)

if __name__=='__main__':main()
