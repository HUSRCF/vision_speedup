"""Fresh guard/ranks per NCCL environment configuration, sequential only."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root=Path.cwd();configurations=json.loads((root/'configurations.json').read_text());results=[]
    for config in configurations:
        path=root/config['name'];path.mkdir(mode=0o700)
        for name in('wait_nccl_geglu_model_pair.py','check_nccl_geglu_model_rank.py','check_dual_gpu_graph.py','payloads.json'):
            os.link(root/name,path/name)
        (path/'tuning.json').write_text(json.dumps(config,indent=2)+'\n')
        env={k:v for k,v in os.environ.items()if not k.startswith('NCCL_')}
        env.update(config['env']);env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,P2P,GRAPH,ENV',PYTHONPATH=str(path),TMPDIR=str(path))
        print('NCCL_TUNING_CONFIGURATION='+config['name'],flush=True)
        with(path/'launch.log').open('x')as output:
            p=subprocess.run([sys.executable,str(path/'wait_nccl_geglu_model_pair.py')],cwd=path,env=env,stdout=output,stderr=subprocess.STDOUT)
        assert p.returncode==0,(config['name'],(path/'launch.log').read_text()[-2000:])
        report=json.loads((path/'nccl-geglu-model-report.json').read_text());assert report['clean']
        results.append(dict(config=config,phase_ms={phase:max(r['phases'][phase]['timing']['median_ms']for r in report['ranks'])for phase in('decision','world')}))
        (root/'partial-summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print('NCCL_TUNING_RESULT '+json.dumps(results[-1]),flush=True)
    (root/'sweep-summary.json').write_text(json.dumps(dict(clean=True,results=results,all_configurations_completed=True,no_model_or_video_quality_claim=True),indent=2)+'\n')
    print('NCCL_TUNING_SWEEP passed=true',flush=True)


if __name__=='__main__':main()
