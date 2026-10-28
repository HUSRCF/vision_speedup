"""CPU reproduction of Triton2.3.1's temporary CUBIN glob race.

Execute the installed FileCacheManager.put and CompiledKernel constructor.
Only the launcher import and driver descriptor are mocked; no GPU code is
loaded, launched or allocated. All artifacts are synthetic temporary bytes.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
import triton
from triton.runtime.cache import FileCacheManager
import triton.compiler.compiler as compiler


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    assert triton.__version__=='2.3.1'
    binary=b'\x7fELF'+b'\0'*14+b'\xbe\0'+b'synthetic fixture only'
    rows=[]
    with tempfile.TemporaryDirectory(prefix='triton-cache-race-cpu-') as name:
        with patch.dict(os.environ,TRITON_CACHE_DIR=name):
            shared=FileCacheManager('shared')
            first=FileCacheManager('rank0')
            second=FileCacheManager('rank1')
        for cache in (shared,first,second):
            cache.put(json.dumps(dict(name='fixture',tensormaps_info=[])),'fixture.json')
            cache.put(binary,'fixture.cubin')
        fake_loader=SimpleNamespace(exec_module=lambda module:None)
        def construct(cache):
            with patch.object(compiler,'driver',SimpleNamespace(binary_ext='cubin')):
                with patch.object(importlib.util,'spec_from_file_location',return_value=SimpleNamespace(loader=fake_loader)):
                    with patch.object(importlib.util,'module_from_spec',return_value=SimpleNamespace(launch=lambda:None)):
                        return compiler.CompiledKernel('mock-launcher-not-loaded.so',str(Path(cache.cache_dir)/'fixture.json'))
        real_replace=os.replace
        def overlap(writer,reader):
            observation={}
            def replace(source,destination):
                assert Path(source).name.startswith('fixture.cubin.tmp.pid_')
                assert Path(source).exists()
                files=list(Path(reader.cache_dir).glob('fixture.*'))
                observation['reader_glob_contains_temporary_binary']=Path(source) in files
                try:
                    kernel=construct(reader)
                    observation.update(loaded=True,synthetic_binary_equal=kernel.kernel==binary)
                except UnicodeDecodeError as exc:
                    observation.update(loaded=False,error=type(exc).__name__,codec=exc.encoding,byte_offset=exc.start)
                return real_replace(source,destination)
            with patch.object(os,'replace',replace):writer.put(binary,'fixture.cubin')
            return observation
        shared_result=overlap(shared,shared)
        isolated_result=overlap(first,second)
        assert not shared_result['loaded'] and shared_result['error']=='UnicodeDecodeError'
        assert shared_result['reader_glob_contains_temporary_binary']
        assert isolated_result['loaded'] and isolated_result['synthetic_binary_equal']
        assert not isolated_result['reader_glob_contains_temporary_binary']
        rows=[dict(case='shared_directory',**shared_result),dict(case='per_rank_directory',**isolated_result)]
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,CPU_only=True,Torch_CUDA_not_initialized=True,
        installed_Triton=triton.__version__,rows=rows,
        actual_installed_cache_writer_and_compiled_kernel_constructor=True,
        mocked_only_launcher_import_and_driver_descriptor=True,
        binary_is_synthetic_fixture_not_model_or_GPU_executable=True,
        installed_writer_source_SHA256=hashlib.sha256(inspect.getsource(FileCacheManager.put).encode()).hexdigest(),
        installed_reader_source_SHA256=hashlib.sha256(inspect.getsource(compiler.CompiledKernel.__init__).encode()).hexdigest(),
        production_cache_or_sources_modified=False,GPU_speed_or_numerics_result=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
