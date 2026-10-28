"""Bounded FP32 head GroupNorm/Mish fusion; reads live affine masters."""
import ctypes
from functools import lru_cache

import torch
from unifolm_wma.utils.nvrtc_geglu import _libraries, _check


SOURCE = r'''
struct Moment { float mean, m2, count; };
__device__ __forceinline__ Moment merge(Moment a, Moment b) {
    if (a.count == 0.f) return b;
    if (b.count == 0.f) return a;
    float delta = b.mean-a.mean;
    float count = a.count+b.count;
    float ratio = b.count/count;
    return {a.mean+delta*ratio,
            a.m2+b.m2+delta*delta*a.count*ratio, count};
}
__device__ __forceinline__ Moment warp(Moment a) {
    #pragma unroll
    for (int offset=16; offset; offset/=2) {
        Moment b = {__shfl_down_sync(0xffffffff,a.mean,offset),
                    __shfl_down_sync(0xffffffff,a.m2,offset),
                    __shfl_down_sync(0xffffffff,a.count,offset)};
        a=merge(a,b);
    }
    return a;
}
extern "C" __global__ void gn_mish(
    const float* x, const float* weight, const float* bias, float* out,
    int channels, int length, int size, float epsilon, int affine_mode,
    float* debug_norm, float* debug_stats) {
    int tid=threadIdx.x, lane=tid%32, wid=tid/32;
    int group=blockIdx.x;
    Moment a={0.f,0.f,0.f};
    int count=0;
    for (int j=tid; j<size; j+=blockDim.x) {
        float value=x[group*size+j];
        ++count;
        float delta=value-a.mean;
        float mean=a.mean+delta/float(count);
        float delta2=value-mean;
        a={mean,a.m2+delta*delta2,float(count)};
    }
    a=warp(a);
    __shared__ Moment partial[16];
    __shared__ float center, inverse;
    __syncthreads();
    if (lane==0) partial[wid]=a;
    __syncthreads();
    a=tid<16 ? partial[lane] : Moment{0.f,0.f,0.f};
    if (wid==0) a=warp(a);
    if (tid==0) {
        center=a.mean;
        inverse=rsqrtf(a.m2/a.count+epsilon);
        if (debug_stats) {
            debug_stats[group*2]=center;
            debug_stats[group*2+1]=inverse;
        }
    }
    __syncthreads();
    for (int j=tid; j<size; j+=blockDim.x) {
        int index=group*size+j;
        int channel=(index/length)%channels;
        float scale=__fmul_rn(inverse,weight[channel]);
        float shift=(affine_mode&1) ? __fmaf_rn(-scale,center,bias[channel])
            : __fadd_rn(__fmul_rn(-scale,center),bias[channel]);
        float value=(affine_mode&2) ? __fmaf_rn(scale,x[index],shift)
            : __fadd_rn(__fmul_rn(scale,x[index]),shift);
        if (debug_norm) debug_norm[index]=value;
        out[index]=value*tanhf(log1pf(expf(value)));
    }
}
'''


@lru_cache(None)
def kernel(device_index, fmad):
    rtc, driver = _libraries()
    major, minor = torch.cuda.get_device_capability(device_index)
    program = ctypes.c_void_p()
    _check(rtc.nvrtcCreateProgram(ctypes.byref(program), SOURCE.encode(),
                                b'gn_mish.cu', 0, None, None), 'nvrtcCreateProgram')
    try:
        opts = [f'--gpu-architecture=compute_{major}{minor}'.encode(),
                b'--std=c++14', b'--fmad=true' if fmad else b'--fmad=false']
        options = (ctypes.c_char_p * len(opts))(*opts)
        status = rtc.nvrtcCompileProgram(program, len(opts), options)
        if status:
            size = ctypes.c_size_t()
            _check(rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size)), 'log size')
            log = ctypes.create_string_buffer(size.value)
            _check(rtc.nvrtcGetProgramLog(program, log), 'log')
            raise RuntimeError(log.value.decode())
        size = ctypes.c_size_t()
        _check(rtc.nvrtcGetPTXSize(program, ctypes.byref(size)), 'ptx size')
        ptx = ctypes.create_string_buffer(size.value)
        _check(rtc.nvrtcGetPTX(program, ptx), 'ptx')
    finally:
        rtc.nvrtcDestroyProgram(ctypes.byref(program))
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    _check(driver.cuModuleLoadData(ctypes.byref(module), ptx), 'module load')
    _check(driver.cuModuleGetFunction(ctypes.byref(function), module, b'gn_mish'), 'function')
    return module, function


def fused(value, weight, bias, *, groups=8, eps=1e-5, fmad=True, affine_mode=3, debug=False):
    if (torch.is_grad_enabled() or value.ndim != 3 or not value.is_cuda or
            value.dtype != torch.float32 or not value.is_contiguous() or
            tuple(value.shape) not in ((16,256,32),(16,512,16),(16,1024,8),(16,2048,4)) or
            groups != 8 or torch.is_autocast_enabled()):
        raise ValueError('unsupported head GroupNorm/Mish input')
    if any(p.device != value.device or p.dtype != torch.float32 or
           not p.is_contiguous() or p.shape != (value.shape[1],) for p in (weight,bias)):
        raise ValueError('unsupported GroupNorm affine parameters')
    output = torch.empty_like(value)
    norm = torch.empty_like(value) if debug else None
    stats = torch.empty((value.shape[0]*groups,2),device=value.device) if debug else None
    with torch.cuda.device(value.device):
        _, function = kernel(value.device.index, fmad)
        values = [ctypes.c_void_p(p.data_ptr()) for p in (value,weight,bias,output)]
        values += [ctypes.c_int(value.shape[1]),ctypes.c_int(value.shape[2]),
                   ctypes.c_int(1024),ctypes.c_float(eps),ctypes.c_int(affine_mode),
                   ctypes.c_void_p(norm.data_ptr() if debug else 0),
                   ctypes.c_void_p(stats.data_ptr() if debug else 0)]
        pointers = (ctypes.c_void_p*len(values))(
            *[ctypes.cast(ctypes.pointer(p),ctypes.c_void_p) for p in values])
        _check(_libraries()[1].cuLaunchKernel(function,value.shape[0]*groups,1,1,
            512,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            pointers,None),'GroupNorm/Mish launch')
    return (output,norm,stats) if debug else output
