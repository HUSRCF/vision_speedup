"""CPU lifecycle checks before a dual-GPU timed evaluator is queued."""
import importlib.util,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch

def main():
    source=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('parallel_lifecycle_fixture',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert 'torch' not in sys.modules
    events=[]
    class Child:
        def __init__(self,command,**kwargs):self.command=command;self.kwargs=kwargs;self.returncode=None;events.append(self)
        def poll(self):return self.returncode
        def terminate(self):self.returncode=-15
        def wait(self,timeout=None):assert self.returncode is not None;return self.returncode
        def kill(self):self.returncode=-9
    with tempfile.TemporaryDirectory(prefix='nccl-cpu-',dir=Path.cwd()) as temporary:
        root=Path(temporary)/'parallel'
        arguments=['scripts/evaluation/world_model_interaction.py','--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir','original-output','--profile_path','original-profile','--seed','123','--n_iter','11','--ddim_steps','50','--video_length','16']
        policy=os.environ.get('CPU_TEST_POLICY','world-ffn');pair=os.environ.get('CPU_TEST_GPUS','2,3').split(',')
        assert len(pair)==len(set(pair))==2 and set(pair)<=set('0123')
        environment=dict(UNIFOLM_NCCL_PARALLEL=policy,MONITORED_GPUS=','.join(pair),CUDA_VISIBLE_DEVICES=pair[0],TEAM07_PARALLEL_RANK='0',TEAM07_PARALLEL_DIR=str(root))
        with patch.dict(os.environ,environment,clear=True),patch.object(sys,'argv',arguments),patch.object(module.subprocess,'Popen',Child):
            module.early_start();assert len(events)==1 and 'torch' not in sys.modules
            child=events[0];actual=child.command[1:]
            assert actual==module.helper_arguments(arguments,root)
            assert child.kwargs['env']['CUDA_VISIBLE_DEVICES']==pair[1] and child.kwargs['env']['TEAM07_PARALLEL_RANK']=='1'
            assert not any(k in child.kwargs['env'] for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
            for flag in ('--dataset','--seed','--n_iter','--ddim_steps','--video_length'):assert module.argument(actual,flag)==module.argument(arguments,flag)
            assert module.argument(actual,'--savedir')!=module.argument(arguments,'--savedir')
            module.cleanup();assert child.returncode==-15 and module._child_log is None
        try:module.argument(arguments+['--seed','1'],'--seed')
        except AssertionError:pass
        else:raise AssertionError('duplicate sampler arguments accepted')
    import torch
    from torch.distributed import TCPStore
    from datetime import timedelta
    store=TCPStore('127.0.0.1',0,2,True,timeout=timedelta(seconds=10),wait_for_workers=False)
    assert store.port>0;store.set('fixture','ok');assert store.get('fixture')==b'ok'
    assert not torch.cuda.is_initialized()
    print('NCCL_PARALLEL_CPU_CHECK '+json.dumps(dict(passed=True,helper_only_changes_output_paths_and_device=True,helper_cleanup_verified=True,no_pre_timer_torch_import=True,tcpstore_port_zero_supported=True,no_GPU_initialized=True)))

if __name__=='__main__':main()
