"""Observe original prerequisites and start a full run only after clean proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def live(pid):
    try:
        os.kill(pid,0)
        status=Path('/proc')/str(pid)/'status'
        state=next(line for line in status.read_text().splitlines() if line.startswith('State:'))
        return state.split()[1]!='Z'
    except (ProcessLookupError,FileNotFoundError):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--wait-timeout-seconds',type=int,default=7*3600)
    args=parser.parse_args();spec=json.loads(args.spec.read_text())
    assert args.wait_timeout_seconds>0
    deadline=time.monotonic()+args.wait_timeout_seconds
    while True:
        waiting=[p['pid'] for p in spec['prerequisites'] if live(p['pid'])]
        if not waiting:break
        if time.monotonic()>deadline:raise RuntimeError('GEGLU prerequisites wait deadline expired')
        print('GEGLU_PREREQUISITES_WAIT '+json.dumps(dict(live_original_pids=waiting)),flush=True)
        time.sleep(30)
    for prerequisite in spec['prerequisites']:
        text=Path(prerequisite['log']).read_text()
        if (prerequisite['marker'] not in text or 'MICROBENCHMARK_CLEAN' not in text
                or 'MICROBENCHMARK_CONTAMINATED' in text):
            print('GEGLU_FULL_NOT_STARTED '+json.dumps(dict(reason='Prerequisite not clean or failed',
                name=prerequisite['name'],log=prerequisite['log'])),flush=True)
            return 2
    pair=next(p for p in spec['prerequisites'] if p['name']=='actual_pair')
    directories=re.findall(r'^DIAGNOSTIC_DIR=(.*)$',Path(pair['log']).read_text(),re.M)
    assert len(directories)==1
    comparison=json.loads((Path(directories[0])/'comparison.json').read_text())
    assert set(comparison['stages'])=={'world','decision'}
    assert all(stage['output_signed_zero_bitwise'] and stage['exact_output_tensors']==3
               and len(stage['pairs'])==4 for stage in comparison['stages'].values())
    saved=sum(stage['median_saved_ms'] for stage in comparison['stages'].values())
    if comparison['stages']['world']['median_saved_ms']<=0 or saved<=0:
        print('GEGLU_FULL_NOT_STARTED '+json.dumps(dict(reason='No complete graph speed gain',
            comparison=comparison)),flush=True)
        return 3
    source,root=Path(spec['source_root']),Path(spec['full_root'])
    for name in spec['verified_source_files']:
        assert hashlib.sha256((source/name).read_bytes()).digest()==hashlib.sha256((root/name).read_bytes()).digest()
    print('GEGLU_FULL_PREREQUISITES_PASSED '+json.dumps(dict(world_saved_ms=
        comparison['stages']['world']['median_saved_ms'],both_stage_saved_ms=saved,
        source_files_exact=len(spec['verified_source_files']),full_root=str(root))),flush=True)
    # Preserve the original waiter PID and its output stream through the full run.
    os.chdir(root)
    os.execvpe('bash',['bash',str(root/spec['leaf'])],os.environ)


if __name__=='__main__':sys.exit(main())
