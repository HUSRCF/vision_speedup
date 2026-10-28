"""New diagnostic cache: bulk artifacts in /tmp, tiny launchers in team home.

The contest's /tmp is noexec. Python launcher shared objects therefore belong
in a fresh private home directory. No original production cache is accessed.
"""
import os,shutil
from pathlib import Path
from triton.runtime.cache import FileCacheManager


class SplitExperimentCache(FileCacheManager):
    def __init__(self,key,override=False,dump=False):
        assert not override and not dump,'Only an isolated diagnostic cache is supported'
        self.executable_root=Path(os.environ['TRITON_EXECUTABLE_CACHE_DIR'])
        assert self.executable_root.is_absolute() and self.executable_root.is_dir()
        super().__init__(key,override=override,dump=dump)

    def _make_path(self,filename):
        assert Path(filename).name==filename
        if filename.endswith('.so'):
            directory=self.executable_root/self.key
            directory.mkdir(mode=0o700,exist_ok=True)
            return str(directory/filename)
        return super()._make_path(filename)

    def put(self,data,filename,binary=True):
        destination=self.executable_root if filename.endswith('.so') else Path(self.cache_dir)
        size=len(data) if isinstance(data,bytes) else len(str(data).encode())
        assert shutil.disk_usage(destination).free>size+4*2**20,'Diagnostic cache free-space margin'
        return super().put(data,filename,binary=binary)
