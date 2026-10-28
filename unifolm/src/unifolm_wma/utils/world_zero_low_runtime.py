"""Decision-only existing zero-low GEMM, with persistent original-pack counts."""
import os
import torch

ORIGINAL_PACK=None
ORIGINAL_JIT=None
ORIGINAL_RUN=None
active=None


def flag_enabled():
    value=os.environ.get('UNIFOLM_WORLD_PREPACKED_ZERO_LOW','0')
    if value not in {'0','1'}:
        raise ValueError('Invalid decision prepacked zero-low mode')
    if value=='0':return False
    for name,expected in [('UNIFOLM_DECISION_COMBINED_ZERO_LOW','1'),('UNIFOLM_CUDA_GRAPH','1'),
                          ('UNIFOLM_NATIVE_GRAPH_WEIGHT_CACHE','1'),
                          ('UNIFOLM_COMPENSATED_LINEAR_INPUT_PACK','wide')]:
        if os.environ.get(name)!=expected:
            raise ValueError('Decision prepacked zero-low requires '+name+'='+expected)
    return True


class Enabled:
    def __bool__(self):return flag_enabled()


class Shader:
    def __getitem__(self,grid):
        from .triton_zero_weight_low_linear import _zero_low_gemm as kernel
        return kernel[grid]


enabled=Enabled()
_zero_low_gemm=Shader()


def exact(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape
    if a.dtype==torch.float32:
        assert torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    else:assert torch.equal(a,b)


def old_launch(grid,args,kwargs):
    return ORIGINAL_RUN(*args,grid=grid,warmup=False,**kwargs)


class CountOwner:
    def __init__(self,high,low):
        self.high,self.low=high,low
        self.count=torch.empty((),device=low.device,dtype=torch.int64)
        self.pointer=self.count.data_ptr();self.refreshes=0;self.valid=False
        self.refresh()

    def refresh(self):
        if torch.cuda.is_current_stream_capturing():
            # The existing full three-product kernel remains the cold/stale
            # capture fallback; do not cache an unrefreshed device predicate.
            self.valid=False
            return
        self.count.copy_(torch.count_nonzero(self.low))
        self.version=self.low._version;self.refreshes+=1;self.valid=True


class Controller:
    def __init__(self,configs):
        self.configs={(tuple(c['geometry']),c['has_bias'],c['output_groups']):c
                      for c in configs}
        self.owners={};self.captures=[];self.fallbacks={}

    def fallback(self,reason,grid,args,kwargs):
        self.fallbacks[reason]=self.fallbacks.get(reason,0)+1
        return old_launch(grid,args,kwargs)

    def launch(self,grid,args,kwargs):
        if (not enabled or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
            return old_launch(grid,args,kwargs)
        values=dict(zip(ORIGINAL_JIT.arg_names,args));values.update(kwargs)
        values.setdefault('OUTPUT_GROUPS',1)
        names=['XH','XL','WH','WL','Bias','Output','M','N','K','HAS_BIAS','BM','BN','BK','GROUP_M']
        if any(n not in values for n in names):return self.fallback('arguments',grid,args,kwargs)
        v=values;geometry=tuple(v[n] for n in ['M','N','K'])
        config=self.configs.get((geometry,v['HAS_BIAS'],v['OUTPUT_GROUPS']))
        if (config is None or [v[n] for n in ['BM','BN','BK','GROUP_M']]!=config['config']
                or kwargs.get('num_warps')!=4 or kwargs.get('num_stages')!=2
                or kwargs.get('enable_fp_fusion') is not False):
            return self.fallback('unqualified_config',grid,args,kwargs)
        low,high=v['WL'],v['WH']
        if (low.dtype!=torch.float16 or high.dtype!=torch.float16
                or low.shape!=high.shape or not low.is_cuda or not low.is_contiguous()
                or not high.is_contiguous()):return self.fallback('parts',grid,args,kwargs)
        owner=self.owners.get(id(low))
        capturing=torch.cuda.is_current_stream_capturing()
        if owner is None:
            if capturing:return self.fallback('cold_capture',grid,args,kwargs)
            owner=CountOwner(high,low);self.owners[id(low)]=owner
        assert owner.low is low and owner.high is high
        if owner.valid and owner.version!=low._version:owner.refresh()
        if not owner.valid:return self.fallback('stale_capture',grid,args,kwargs)
        assert owner.count.data_ptr()==owner.pointer
        if capturing:self.captures.append(dict(geometry=list(geometry),low_ptr=low.data_ptr(),Count_ptr=owner.pointer))
        return _zero_low_gemm[grid](v['XH'],v['XL'],high,low,owner.count,v['Bias'],v['Output'],
            v['M'],v['N'],v['K'],v['HAS_BIAS'],v['BM'],v['GROUP_M'],v['OUTPUT_GROUPS'],**kwargs)

    def report(self):
        assert not torch.cuda.is_current_stream_capturing()
        return dict(owners=len(self.owners),Count_bytes=len(self.owners)*8,captured=len(self.captures),
            captured_consumers=self.captures.copy(),fallback_counts=self.fallbacks.copy(),
            rows=[dict(low_ptr=o.low.data_ptr(),Count_ptr=o.count.data_ptr(),Count_ptr_same=o.count.data_ptr()==o.pointer,
                low_nonzero=int(o.count),valid=o.valid,low_version_matches=o.valid and o.version==o.low._version,
                refreshes=o.refreshes) for o in self.owners.values()])


def packed(*args,**kwargs):
    parts=ORIGINAL_PACK(*args,**kwargs)
    if active is not None and kwargs.get('weight_scale',256.)==256.:
        owner=active.owners.get(id(parts[1]))
        if owner is not None:
            assert owner.low is parts[1] and owner.high is parts[0]
            owner.refresh()
    return parts


def launched(*args,grid=None,warmup=False,**kwargs):
    if warmup:return ORIGINAL_RUN(*args,grid=grid,warmup=warmup,**kwargs)
    return active.launch(grid,args,kwargs)


CONFIGS=[{'geometry': [2560, 1280, 1280], 'config': [64, 64, 64, 32], 'has_bias': True, 'output_groups': 1, 'num_stages': 2, 'nodes': 40}, {'geometry': [2560, 1280, 1280], 'config': [64, 64, 64, 32], 'has_bias': False, 'output_groups': 1, 'num_stages': 2, 'nodes': 5}, {'geometry': [640, 1280, 1280], 'config': [32, 64, 64, 32], 'has_bias': True, 'output_groups': 1, 'num_stages': 2, 'nodes': 8}, {'geometry': [640, 1280, 1280], 'config': [32, 64, 64, 32], 'has_bias': False, 'output_groups': 1, 'num_stages': 2, 'nodes': 1}]


def wrap_pack(original):
    global ORIGINAL_PACK
    flag_enabled()
    assert ORIGINAL_PACK is None
    ORIGINAL_PACK=original
    return packed


def wrap_gemm(original):
    global ORIGINAL_JIT,ORIGINAL_RUN,active
    flag_enabled()
    assert ORIGINAL_JIT is None
    ORIGINAL_JIT=original;ORIGINAL_RUN=original.run
    active=Controller(CONFIGS)
    original.run=launched
    return original


def report_capture():
    if (flag_enabled() and not torch.backends.cudnn.allow_tf32
            and not torch.backends.cuda.matmul.allow_tf32):
        import json
        report=active.report()
        report.update(all_setup_inside_process_wall=True,
            original_JIT_object_retained=True,original_pack_refresh_owners=True)
        print('WORLD_PREPACKED_ZERO_LOW_CAPTURE '+json.dumps(report),flush=True)
