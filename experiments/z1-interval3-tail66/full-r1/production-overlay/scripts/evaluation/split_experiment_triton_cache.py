"""Fresh diagnostic cache: executable host stubs on home, GPU artifacts on RAM."""
import os
import gzip
from pathlib import Path
from triton.runtime.cache import FileCacheManager


class SplitCache(FileCacheManager):
    def __init__(self,key,override=False,dump=False):
        assert not override and not dump, 'Diagnostic does not override/dump shared kernels'
        self.executable_root=Path(os.environ['EXPERIMENT_EXEC_CACHE'])
        assert self.executable_root.is_dir() and not self.executable_root.is_symlink()
        super().__init__(key,override=False,dump=False)

    def _make_path(self,filename):
        assert Path(filename).name==filename
        if filename.endswith('.so'):
            directory=self.executable_root/self.key
            directory.mkdir(mode=0o700,exist_ok=True)
            return str(directory/filename)
        return super()._make_path(filename)

    def put(self,data,filename,binary=True):
        # Runtime loads only metadata, launcher.so and cubin. Preserve readable
        # intermediate diagnostics as compressed evidence without changing any
        # compiler stage's in-memory output, options or runtime binary.
        if filename.endswith(('.ttir','.ttgir','.llir','.ptx')):
            raw=data if isinstance(data,bytes)else str(data).encode()
            # Triton2.3 scans '<kernel>.*' beside metadata. Give compressed
            # evidence a different prefix so it is not parsed as textual asm.
            return super().put(gzip.compress(raw,mtime=0),'IR-archive-'+filename+'.gz',binary=True)
        return super().put(data,filename,binary=binary)


class ReadWarmSplitCache(SplitCache):
    """Reuse complete original warm keys read-only, isolate new diagnostic keys."""
    def __init__(self,key,override=False,dump=False):
        assert not override and not dump
        self.original_warm=Path.home()/'.triton/cache'/key
        self.readonly_warm=self.original_warm.is_dir()
        if self.readonly_warm:
            self.key=key;self.cache_dir=str(self.original_warm);self.lock_path=None
        else:super().__init__(key,override=False,dump=False)

    def _make_path(self,filename):
        if self.readonly_warm:
            assert Path(filename).name==filename
            return str(self.original_warm/filename)
        return super()._make_path(filename)

    def put(self,data,filename,binary=True):
        assert not self.readonly_warm,'Refuse to mutate an original warm cache key'
        return super().put(data,filename,binary=binary)
