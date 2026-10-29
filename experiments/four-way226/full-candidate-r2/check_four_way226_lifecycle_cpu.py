"""Check rank routing, unchanged workload arguments and owned-child cleanup."""
import importlib.util,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch

def main():
    spec=importlib.util.spec_from_file_location('four_lifecycle_fixture',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert 'torch' not in sys.modules
    events=[]
    class Child:
        def __init__(self,command,**kw):self.command=command;self.kw=kw;self.code=None;events.append(self)
        def poll(self):return self.code
        def terminate(self):self.code=-15
        def wait(self,timeout=None):assert self.code is not None;return self.code
        def kill(self):self.code=-9
    with tempfile.TemporaryDirectory(prefix='four-lifecycle-',dir=Path.cwd()) as d:
        root=Path(d)/'parallel'
        args=['scripts/evaluation/world_model_interaction.py','--dataset','unitree_z1_dual_arm_stackbox_v2','--savedir','original-output','--profile_path','original-profile','--seed','123','--n_iter','11','--ddim_steps','50','--video_length','16']
        env=dict(UNIFOLM_NCCL_PARALLEL='four-way225-native-pairs',MONITORED_GPUS='0,1,2,3',CUDA_VISIBLE_DEVICES='0',TEAM07_PARALLEL_RANK='0',TEAM07_PARALLEL_DIR=str(root))
        with patch.dict(os.environ,env,clear=True),patch.object(sys,'argv',args),patch.object(m.subprocess,'Popen',Child):
            m.early_start();assert len(events)==3 and 'torch' not in sys.modules
            for rank,child in enumerate(events,1):
                actual=child.command[1:];assert actual==m.helper_arguments(args,root,rank)
                assert child.kw['env']['CUDA_VISIBLE_DEVICES']==str(rank) and child.kw['env']['TEAM07_PARALLEL_RANK']==str(rank)
                for flag in ('--dataset','--seed','--n_iter','--ddim_steps','--video_length'):assert m.argument(actual,flag)==m.argument(args,flag)
            assert len({m.argument(c.command[1:],'--savedir') for c in events})==3
            m.cleanup();assert all(c.code==-15 for c in events) and not m._child_logs
        try:m.argument(args+['--seed','1'],'--seed')
        except AssertionError:pass
        else:raise AssertionError('duplicate sampler arguments accepted')
    print('FOUR_WAY226_LIFECYCLE_CPU '+json.dumps(dict(passed=True,children=3,workload_arguments_unchanged=True,unique_output_paths=True,owned_child_cleanup=True,no_pre_timer_torch_import=True)))

if __name__=='__main__':main()
