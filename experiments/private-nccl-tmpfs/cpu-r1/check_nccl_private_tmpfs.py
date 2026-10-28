"""CPU cross-process mmap and path isolation checks for the storage adapter."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root=Path(os.environ['TEAM07_NCCL_SHM_ROOT'])
    libc=ctypes.CDLL(None,use_errno=True)
    results=[]
    for symbol in ('mkstemp','mkstemp64'):
        func=getattr(libc,symbol);func.argtypes=[ctypes.c_char_p];func.restype=ctypes.c_int
        name=ctypes.create_string_buffer(b'/dev/shm/nccl-XXXXXX')
        fd=func(name);assert fd>=0,ctypes.get_errno()
        virtual=os.fsdecode(name.value);actual=root/Path(virtual).name
        assert not Path(virtual).exists()and actual.is_file()
        assert actual.stat().st_uid==os.getuid()and actual.stat().st_mode&0o777==0o600
        os.ftruncate(fd,8192)
        with mmap.mmap(fd,8192)as view:
            view[:8]=b'parent01'
            code="import os,mmap,sys;f=os.open(sys.argv[1],os.O_RDWR);v=mmap.mmap(f,8192);assert v[:8]==b'parent01';v[4096:4104]=b'child002';v.close();os.close(f)"
            subprocess.run([sys.executable,'-c',code,virtual],check=True)
            assert view[4096:4104]==b'child002'
        os.close(fd);os.unlink(virtual);assert not actual.exists()
        results.append(dict(symbol=symbol,virtual_name_preserved=True,backing_private=True,child_mmap_exact=True,unlink_exact=True))
    # Unrelated /dev/shm and ordinary tmpfs paths must retain their storage.
    fd,name=tempfile.mkstemp(prefix='team07-private-tmpfs-control-',dir='/dev/shm')
    try:
        os.write(fd,b'unchanged');assert Path(name).read_bytes()==b'unchanged'
        assert not(root/Path(name).name).exists()
    finally:os.close(fd);os.unlink(name)
    fd,name=tempfile.mkstemp(prefix='ordinary-control-',dir=root)
    os.close(fd);assert Path(name).is_file();os.unlink(name)
    # Disabled adapter is transparent, including canonical NCCL-shaped names.
    env=dict(os.environ);env.pop('TEAM07_NCCL_SHM_ROOT')
    subprocess.run([sys.executable,'-c',"import ctypes,os;f=ctypes.CDLL(None).mkstemp;p=ctypes.create_string_buffer(b'/dev/shm/nccl-XXXXXX');d=f(p);assert d>=0 and os.path.isfile(p.value);os.close(d);os.unlink(p.value)"],env=env,check=True)
    invalid=dict(os.environ,TEAM07_NCCL_SHM_ROOT='/var/tmp')
    rejected=subprocess.run([sys.executable,'-c','raise AssertionError("must not run")'],env=invalid,capture_output=True,text=True)
    assert rejected.returncode==125 and 'invalid private tmpfs root'in rejected.stderr
    assert not list(root.iterdir()),'The adapter must not leave temporary files'
    report=dict(passed=True,records=results,unrelated_paths_unchanged=True,disabled_transparent=True,nonprivate_root_rejected=True,own_files_cleaned=True,no_GPU_execution=True)
    Path(os.environ['CPU_STORAGE_REPORT']).write_text(json.dumps(report,indent=2)+'\n')
    print('NCCL_PRIVATE_TMPFS_CPU passed=true',flush=True)


if __name__=='__main__':main()
