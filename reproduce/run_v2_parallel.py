"""Run the adopted two-GPU variant under the original evaluator's timer."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True,help='Two physical GPU indices, for example 2,3')
    p.add_argument('--workdir',type=Path,required=True,help='New empty directory for source and diagnostic evidence')
    p.add_argument('--storage-config',type=Path,help='Optional CPU-qualified private tmpfs adapter config');p.add_argument('--prepare-only',action='store_true',help='Prepare source/guard without GPU execution')
    p.add_argument('--variant',choices=('both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block','both-spatial-temporal-resblock-grid-native-attention-initial-ffn','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid','both-spatial-temporal','world-spatial-temporal','world-ffn-temporal'),default='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block')
    p.add_argument('--smoke',action='store_true',help='One round only; not a competition result');a=p.parse_args()
    devices=a.gpus.split(',');assert len(devices)==2 and len(set(devices))==2 and all(v in ('0','1','2','3') for v in devices)
    directory=a.workdir.resolve();assert not directory.exists();directory.mkdir(parents=True,mode=0o700)
    repository=Path(__file__).resolve().parents[1]
    overlay=repository/{'both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block':'experiments/initial-boundary-221/production222','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block':'experiments/temporal-block-220/production221','both-spatial-temporal-resblock-grid-native-attention-initial-ffn':'experiments/initial-ffn-219/production220','both-spatial-temporal-resblock-grid-native-attention':'experiments/native-attention-218/production219','both-spatial-temporal-resblock-grid':'experiments/nccl-native-frame-grid-209','both-spatial-temporal':'experiments/nccl-both-spatial-temporal-209','world-spatial-temporal':'experiments/nccl-spatial-temporal-209','world-ffn-temporal':'experiments/nccl-ffn-209/timed-entry'}[a.variant]
    source=directory/'source';diagnostic=directory/'diagnostic';diagnostic.mkdir()
    subprocess.run([sys.executable,str(overlay/'materialize.py'),'--output',str(source)],check=True)
    for name in ('flags.json','wait_nccl_full209.py','check_dual_gpu_graph.py','check_nccl_parallel209_cpu.py'):shutil.copyfile(overlay/name,diagnostic/name)
    storage={}
    if a.storage_config:
        import hashlib
        config=json.loads(a.storage_config.read_text())
        qualified=json.loads((repository/'experiments/private-nccl-tmpfs/qualified-adapter.json').read_text())
        assert config['cpu_cross_process_passed']and config['source_sha256']==qualified['source_sha256']
        library=Path(config['preload']);shared=Path(config['shared_root'])
        assert library.is_file()and hashlib.sha256(library.read_bytes()).hexdigest()==config['preload_sha256']
        assert shared.is_dir()and not shared.is_symlink()and shared.stat().st_uid==os.getuid()and shared.stat().st_mode&0o777==0o700
        assert directory.stat().st_dev==shared.stat().st_dev,'Keep source, metadata and shared backing on the private tmpfs device'
        assert a.variant in ('both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block')
        kind='candidate'if a.variant=='both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block'else'baseline'
        tested=repository/'experiments/initial-boundary-221/private-tmpfs-full'/('full-'+kind+'-r1')
        shutil.copyfile(tested/'wait_nccl_full209.py',diagnostic/'wait_nccl_full209.py')
        storage=dict(LD_PRELOAD=str(library.resolve()),TEAM07_NCCL_SHM_ROOT=str(shared.resolve()))
    if a.prepare_only:
        print(json.dumps(dict(prepared=True,no_GPU_execution=True,variant=a.variant,source_count=len(list((source/'src').rglob('*.py'))),source_directory=str(source))))
        return
    env={k:v for k,v in os.environ.items() if not k.startswith('UNIFOLM_') and k not in ('RANK','LOCAL_RANK','WORLD_SIZE','LD_PRELOAD','TEAM07_NCCL_SHM_ROOT')}
    env.update(storage)
    env.update(PATH=str(Path(sys.executable).parent)+':'+env['PATH'],CUDA_VISIBLE_DEVICES=a.gpus,MONITORED_GPUS=a.gpus,
        CANONICAL_ROOT=str(source),TRIAL_N_ITER='1' if a.smoke else 'original',PYTHONPATH=str(diagnostic),
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONHASHSEED='123',TMPDIR=str(directory/'tmp')if storage else str(Path.home()/'.tmp'),
        HF_HUB_CACHE='/share/huggingface/hub',HUGGINGFACE_HUB_CACHE='/share/huggingface/hub',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        CKPT_PATH='/share/weights/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt',INPUT_CACHE_DIR='/share/problems/unifolm-world-model-action/examples/compute_benchmark/preprocessed_inputs')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    if a.variant in ('both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block-initial-block','both-spatial-temporal-resblock-grid-native-attention','both-spatial-temporal-resblock-grid-native-attention-initial-ffn','both-spatial-temporal-resblock-grid-native-attention-initial-ffn-temporal-block'):
        runs=directory/'runs';runs.mkdir();env.update(FULL_RUN_PARENT=str(runs),PYTHONDONTWRITEBYTECODE='1')
        if a.variant.endswith(('-initial-ffn','-temporal-block','-initial-block')):
            import tempfile
            executable_cache_parent=Path.home()/'.tmp';executable_cache_parent.mkdir(exist_ok=True)
            chosen=os.environ.get('INIT_FFN_CACHE_DIR')
            initial_cache=Path(chosen).resolve() if chosen else Path(tempfile.mkdtemp(prefix='team07-initial-ffn220-',dir=executable_cache_parent))
            assert initial_cache.is_dir();env['INIT_FFN_CACHE_DIR']=str(initial_cache)
            check='check_init_ffn220_cpu.py'
            if a.variant.endswith(('-temporal-block','-initial-block')):
                chosen=os.environ.get('TEMPORAL_BLOCK_CACHE_DIR')
                temporal_cache=Path(chosen).resolve() if chosen else Path(tempfile.mkdtemp(prefix='team07-temporal221-',dir=executable_cache_parent))
                assert temporal_cache.is_dir();env['TEMPORAL_BLOCK_CACHE_DIR']=str(temporal_cache)
                check='check_initial222_cpu.py'if a.variant.endswith('-initial-block')else'check_temporal221_cpu.py'
        else:check='check_native_attention219_cpu.py'
        shutil.copyfile(overlay/check,diagnostic/check)
        subprocess.run([sys.executable,str(diagnostic/check)],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src')),check=True)
    subprocess.run([sys.executable,str(diagnostic/'check_nccl_parallel209_cpu.py'),str(source/'src/unifolm_wma/utils/nccl_parallel_runtime209.py')],cwd=diagnostic,env=dict(env,CUDA_VISIBLE_DEVICES='',CPU_TEST_POLICY=a.variant,CPU_TEST_GPUS=a.gpus),check=True)
    subprocess.run([sys.executable,str(diagnostic/'wait_nccl_full209.py')],cwd=diagnostic,env=env,check=True)
    print(json.dumps(json.loads((diagnostic/'nccl-full-report.json').read_text()),indent=2))

if __name__=='__main__':main()
