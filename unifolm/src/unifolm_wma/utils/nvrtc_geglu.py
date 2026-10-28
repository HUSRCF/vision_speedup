"""FP32 GEGLU fusion compiled with the installed CUDA runtime compiler.

Use CUDA's matching device math implementation for erf. Modules stay loaded
for the process lifetime because CUDA graphs can retain their kernel handles.
No weights, projection arithmetic, or model parameters are changed here.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import site
import torch


_SOURCE = r'''
#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 1
#endif
extern "C" __global__ void fp32_geglu(
    const float* projected, float* output, long long count, int channels) {
    long long start = (long long)blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        long long index = start + (long long)item * blockDim.x;
        if (index < count) {
            long long source = (index / channels) * (2LL * channels) + index % channels;
            float value = projected[source];
            float gate = projected[source + channels];
            float activated = gate * 0.5f * (1.0f + erff(gate * 0.7071067811865476f));
            output[index] = value * activated;
        }
    }
}

struct __align__(16) GeGLUVec4 { float x, y, z, w; };

__device__ __forceinline__ float geglu_element(float value, float gate) {
    float activated = gate * 0.5f * (1.0f + erff(gate * 0.7071067811865476f));
    return value * activated;
}

extern "C" __global__ void fp32_geglu_vec4(
    const float* __restrict__ projected, float* __restrict__ output,
    long long count, int channels) {
    long long start = (long long)blockIdx.x * blockDim.x * ITEMS_PER_THREAD + threadIdx.x;
    long long vectors = count / 4;
    int channel_vectors = channels / 4;
    const GeGLUVec4* input = reinterpret_cast<const GeGLUVec4*>(projected);
    GeGLUVec4* result = reinterpret_cast<GeGLUVec4*>(output);
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        long long index = start + (long long)item * blockDim.x;
        if (index < vectors) {
            long long source = (index / channel_vectors) * (2LL * channel_vectors)
                               + index % channel_vectors;
            GeGLUVec4 value = input[source];
            GeGLUVec4 gate = input[source + channel_vectors];
            GeGLUVec4 activated;
            activated.x = geglu_element(value.x, gate.x);
            activated.y = geglu_element(value.y, gate.y);
            activated.z = geglu_element(value.z, gate.z);
            activated.w = geglu_element(value.w, gate.w);
            result[index] = activated;
        }
    }
}
'''


def _check(status, operation):
    if status:
        raise RuntimeError('%s failed with CUDA status %d' % (operation, status))


@lru_cache(None)
def _libraries():
    candidates = [Path(root) / 'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
                  for root in site.getsitepackages()]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError('CUDA NVRTC library was not found in this environment')
    rtc = ctypes.CDLL(str(path))
    driver = ctypes.CDLL('libcuda.so.1')
    void = ctypes.c_void_p
    rtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(void), ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int, void, void]
    rtc.nvrtcCompileProgram.argtypes = [void, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_char_p)]
    rtc.nvrtcGetProgramLogSize.argtypes = [void, ctypes.POINTER(ctypes.c_size_t)]
    rtc.nvrtcGetProgramLog.argtypes = [void, void]
    rtc.nvrtcGetPTXSize.argtypes = [void, ctypes.POINTER(ctypes.c_size_t)]
    rtc.nvrtcGetPTX.argtypes = [void, void]
    rtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(void)]
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(void), void]
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(void), void, ctypes.c_char_p]
    driver.cuLaunchKernel.argtypes = [void, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, void,
        ctypes.POINTER(void), void]
    return rtc, driver


@lru_cache(None)
def _compile_ptx(major, minor, fmad, items_per_thread=1):
    rtc, _ = _libraries()
    program = ctypes.c_void_p()
    _check(rtc.nvrtcCreateProgram(ctypes.byref(program), _SOURCE.encode(),
                                 b'fp32_geglu.cu', 0, None, None), 'nvrtcCreateProgram')
    try:
        options = [('--gpu-architecture=compute_%d%d' % (major, minor)).encode(),
                   b'--std=c++14', b'--fmad=true' if fmad else b'--fmad=false',
                   ('-DITEMS_PER_THREAD=%d' % items_per_thread).encode()]
        array = (ctypes.c_char_p * len(options))(*options)
        status = rtc.nvrtcCompileProgram(program, len(options), array)
        if status:
            size = ctypes.c_size_t()
            _check(rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size)), 'nvrtcGetProgramLogSize')
            log = ctypes.create_string_buffer(size.value)
            _check(rtc.nvrtcGetProgramLog(program, log), 'nvrtcGetProgramLog')
            raise RuntimeError('NVRTC compilation failed: ' + log.value.decode())
        size = ctypes.c_size_t()
        _check(rtc.nvrtcGetPTXSize(program, ctypes.byref(size)), 'nvrtcGetPTXSize')
        ptx = ctypes.create_string_buffer(size.value)
        _check(rtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
        return ptx.raw
    finally:
        rtc.nvrtcDestroyProgram(ctypes.byref(program))


@lru_cache(None)
def _kernel(device_index, fmad, items_per_thread=1, vectorized=False):
    major, minor = torch.cuda.get_device_capability(device_index)
    ptx = _compile_ptx(major, minor, fmad, items_per_thread)
    driver = _libraries()[1]
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    buffer = ctypes.create_string_buffer(ptx)
    _check(driver.cuModuleLoadData(ctypes.byref(module), buffer), 'cuModuleLoadData')
    name = b'fp32_geglu_vec4' if vectorized else b'fp32_geglu'
    _check(driver.cuModuleGetFunction(ctypes.byref(function), module, name),
           'cuModuleGetFunction')
    return module, function


def nvrtc_geglu(projected, *, fmad=True, block_size=256, items_per_thread=1,
                 vectorized=False):
    if (not projected.is_cuda or projected.dtype != torch.float32 or
            not projected.is_contiguous() or projected.shape[-1] % 2 or
            torch.is_grad_enabled()):
        raise ValueError('NVRTC GEGLU requires contiguous CUDA FP32 inference')
    if block_size not in (128, 256, 512) or items_per_thread not in (1, 2, 4, 8):
        raise ValueError('Unsupported NVRTC GEGLU launch configuration')
    channels = projected.shape[-1] // 2
    if channels % 4 or projected.data_ptr() % 16:
        vectorized = False
    output = torch.empty((*projected.shape[:-1], channels), device=projected.device,
                         dtype=torch.float32)
    count = output.numel()
    if not count:
        return output
    with torch.cuda.device(projected.device):
        _, function = _kernel(projected.device.index, fmad, items_per_thread, vectorized)
        values = [ctypes.c_void_p(projected.data_ptr()), ctypes.c_void_p(output.data_ptr()),
                  ctypes.c_longlong(count), ctypes.c_int(channels)]
        pointers = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values])
        driver = _libraries()[1]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        tile = block_size * items_per_thread
        units = count // 4 if vectorized else count
        _check(driver.cuLaunchKernel(function, (units + tile - 1) // tile, 1, 1,
            block_size, 1, 1, 0, stream, pointers, None), 'cuLaunchKernel')
    return output


def make_activate(module, *, fmad=True, vectorized=False):
    original = module.activate

    def activate(projected):
        if (module.training or torch.is_grad_enabled() or not projected.is_cuda or
                projected.dtype != torch.float32 or not projected.is_contiguous()):
            return original(projected)
        return nvrtc_geglu(projected, fmad=fmad, vectorized=vectorized,
                           block_size=128 if vectorized else 256)

    return activate


def install_nvrtc_geglus(model, *, fmad=True, vectorized=False):
    from unifolm_wma.modules.attention import GEGLU
    count = 0
    for module in model.modules():
        if isinstance(module, GEGLU):
            if getattr(module, '_nvrtc_geglu_installed', False):
                raise RuntimeError('NVRTC GEGLU already installed')
            module.activate = make_activate(module, fmad=fmad, vectorized=vectorized)
            module._nvrtc_geglu_installed = True
            count += 1
    return count
