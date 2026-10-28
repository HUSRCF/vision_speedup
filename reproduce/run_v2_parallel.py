"""Run the adopted two-GPU variant under the original evaluator's timer."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True,help='Two physical GPU indices, for example 2,3')
    p.add_argument('--workdir',type=Path,required=True,help='New empty directory for source and diagnostic evidence')
    p.add_argument('--variant',choices=('world-spatial-temporal','world-ffn-temporal'),default='world-spatial-temporal')
    p.add_argument('--smoke',action='store_true',help='One round only; not a competition result');a=p.parse_args()
    devices=a.gpus.split(',');assert len(devices)==2 and len(set(devices))==2 and all(v in ('0','1','2','3') for v in devices)
    directory=a.workdir.resolve();assert not directory.exists();directory.mkdir(parents=True)
    repository=Path(__file__).resolve().parents[1]
    overlay=repository/('experiments/nccl-spatial-temporal-209' if a.variant=='world-spatial-temporal' else 'experiments/nccl-ffn-209/timed-entry')
    source=directory/'source';diagnostic=directory/'diagnostic';diagnostic.mkdir()
    subprocess.run([sys.executable,str(overlay/'materialize.py'),'--output',str(source)],check=True)
    for name in ('flags.json','wait_nccl_full209.py','check_dual_gpu_graph.py','check_nccl_parallel209_cpu.py'):shutil.copyfile(overlay/name,diagnostic/name)
    env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_') and k not in ('RANK','LOCAL_RANK','WORLD_SIZE')}
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,
        CANONICAL_ROOT=str(source),TRIAL_N_ITER='1' if a.smoke else 'original',PYTHONPATH=str(diagnostic),
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONHASHSEED='123',TMPDIR=str(Path.home()/'.tmp'),
        HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    subprocess.run([sys.executable,str(diagnostic/'check_nccl_parallel209_cpu.py'),str(source/'src/unifolm_wma/utils/nccl_parallel_runtime209.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
    subprocess.run([sys.executable,str(diagnostic/'wait_nccl_full209.py')],cwd=diagnostic,env=env,check=True)
    print(json.dumps(json.loads((diagnostic/'nccl-full-report.json').read_text()),indent=2))

if __name__=='__main__':main()
