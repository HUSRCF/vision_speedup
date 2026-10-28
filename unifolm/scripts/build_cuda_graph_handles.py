"""Build the pinned-version diagnostic handle accessor without a CUDA context."""
import argparse
import json
import os
from pathlib import Path
import hashlib
import subprocess
import sys
import zipfile


def private_headers(output):
    directory = output / 'private-header-wheel'
    directory.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--no-cache-dir',
                    '--only-binary=:all:', '--disable-pip-version-check', '--retries', '0',
                    '--timeout', '15', '--index-url', 'https://pypi.org/simple',
                    '--dest', str(directory), 'nvidia-cuda-nvcc-cu12==12.1.105'], check=True)
    wheels = list(directory.glob('nvidia_cuda_nvcc_cu12-12.1.105-*.whl'))
    assert len(wheels) == 1
    extracted = output / 'private-headers'
    prefix = 'nvidia/cuda_nvcc/include/'
    count = 0
    with zipfile.ZipFile(wheels[0]) as archive:
        for entry in archive.infolist():
            name = entry.filename
            if not name.startswith(prefix) or entry.is_dir():
                continue
            assert '..' not in Path(name).parts and not Path(name).is_absolute()
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(entry))
            count += 1
    include = extracted / prefix
    assert (include / 'crt/host_defines.h').is_file()
    print('CUDA_GRAPH_PRIVATE_HEADERS ' + json.dumps(dict(
        package='nvidia-cuda-nvcc-cu12==12.1.105', wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
        headers=count, installed_into_venv=False, includes=str(include))), flush=True)
    return include


def build(output, private_include=None):
    import torch
    from torch.utils.cpp_extension import load
    assert torch.__version__ == '2.3.1+cu121'
    output.mkdir(exist_ok=True)
    include = Path(torch.__file__).resolve().parent.parent / 'nvidia/cuda_runtime/include'
    os.environ['MAX_JOBS'] = '1'
    return load(name='team07_cuda_graph_handles',
                sources=[str(Path(__file__).with_name('cuda_graph_handles.cpp'))],
                extra_include_paths=([str(private_include)] if private_include else []) + [str(include)], extra_cflags=['-O2'],
                build_directory=str(output), with_cuda=False, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-headers', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    assert not torch.cuda.is_initialized()
    if args.private_headers:
        include = private_headers(args.output)
    else:
        include = None
    module = build(args.output, private_include=include)
    assert callable(module.get_handles) and not torch.cuda.is_initialized()
    print('CUDA_GRAPH_HANDLES_BUILD ' + json.dumps(dict(
        passed=True, torch_version=torch.__version__, CUDA_context_initialized=False,
        graph_or_executable_mutated=False, isolated_build=True)), flush=True)


if __name__ == '__main__':
    main()
