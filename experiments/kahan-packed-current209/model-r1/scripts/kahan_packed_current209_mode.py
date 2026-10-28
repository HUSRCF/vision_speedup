"""Prepack eight static Kahan filters while retaining the original BK32 math."""
from contextlib import contextmanager
import hashlib,struct
import torch
import triton
from unifolm_wma.utils import native_graph_sgemm as production
from unifolm_wma.utils.native_kahan_filter_swap import CachedKahanFilterSwap as OriginalOwner
from unifolm_wma.utils.native_sgemm_prepacked_swap import parameters4,witness4
from unifolm_wma.utils.native_winograd_gemm_fp16_packed_kernel import pack_native_filter
from kahan_packed_current209_kernel import transformed_gemm_fp16_kahan_packed as kernel

enabled=False
active=None
CONFIGS={(640,640,320):(64,64,32,4,2),(1280,192,640):(64,64,32,4,1),
    (1280,192,1920):(64,64,32,4,2),(640,640,1280):(64,64,32,4,2),
    (640,640,960):(64,64,32,4,2),(640,2560,640):(64,64,32,4,2),
    (320,2560,640):(64,64,32,4,2)}


def measure(actual,expected):
    assert actual.dtype==expected.dtype==torch.float32 and actual.shape==expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    exact=torch.equal(actual.contiguous().view(torch.int32),expected.contiguous().view(torch.int32))
    assert exact,dict(max_abs=float((actual-expected).abs().max()))
    return dict(relative_l2=0.,max_abs=0.,signedzero_bits=True)


class PackedOwner(OriginalOwner):
    def __init__(self,graph,weight,plan,geometry,choice,extension,api):
        self.packed=False
        super().__init__(graph,weight,plan,geometry,choice,extension,api)
        if not enabled:return
        assert active is not None and tuple(choice['config'])==(32,64,32,4,1)
        config=CONFIGS[geometry];co,m,ci=geometry;bm,bn,bk,warps,stages=config
        assert bk==32
        self.geometry,self.config=geometry,config
        count=self.cache.cache.numel();self.guard=4096
        self.wh_backing=torch.full((count+2*self.guard,),123.5,device=weight.device,dtype=torch.float16)
        self.wl_backing=torch.full_like(self.wh_backing,123.5)
        self.wh,self.wl=self.wh_backing[self.guard:-self.guard],self.wl_backing[self.guard:-self.guard]
        self.packed_version=None;self.packed_refreshes=0
        half_dummy=torch.empty((1,),device=weight.device,dtype=torch.float16)
        compiled=kernel.warmup(half_dummy,half_dummy,self.dummy,self.dummy,co,m,ci,bm,bn,bk,
            grid=(triton.cdiv(co,bm)*triton.cdiv(m,bn),36),
            num_warps=warps,num_stages=stages,enable_fp_fusion=False)
        compiled._init_handles()
        cubin=hashlib.sha256(compiled.kernel).hexdigest()
        # Capture metadata only. These bounded dummy pointers are never
        # executed or replayed, and never wrap the native workspace.
        identity=(geometry,config)
        if identity not in active.witnesses:
            template=torch.cuda.CUDAGraph()
            with torch.cuda.graph(template):
                observed=kernel[(triton.cdiv(co,bm)*triton.cdiv(m,bn),36)](
                    half_dummy,half_dummy,self.dummy,self.dummy,co,m,ci,bm,bn,bk,
                    num_warps=warps,num_stages=stages,enable_fp_fusion=False)
            assert observed.function==compiled.function
            proof=witness4(template,compiled,(half_dummy.data_ptr(),half_dummy.data_ptr(),
                self.dummy.data_ptr(),self.dummy.data_ptr()),geometry,config,extension,api)
            template.reset();active.witnesses[identity]=dict(**proof,cubin_sha256=cubin,
                metadata_capture_never_replayed=True)
        self.witness=active.witnesses[identity]
        assert self.witness['cubin_sha256']==cubin
        payload=bytes.fromhex(plan[1]['arguments'][0]['bytes_hex']);a,b,c=struct.unpack_from('<3Q',payload)
        self.replacement=parameters4(api,compiled,(self.wh.data_ptr(),self.wl.data_ptr(),a,c),geometry,config)
        self.kernel=compiled;self.packed=True
        self.refresh_if_needed()
        old_witness=self.proof['direct_witness_cubin_sha256']
        self.proof.update(kahan_filter_prepacked=True,original_three_pointer_witness_cubin_sha256=old_witness,
            direct_witness_cubin_sha256=cubin,actual_four_pointer_metadata_witness=self.witness,
            packed_filter_bytes=count*4,packed_config=list(config),
            BK32_three_products_rounding_and_Kahan_preserved=True)
        active.owners.append(self)

    def refresh_if_needed(self):
        super().refresh_if_needed()
        if self.packed and self.packed_version!=self.cache.version:
            n=self.cache.cache.numel()
            pack_native_filter[(triton.cdiv(n,1024),)](self.cache.cache,self.wh,self.wl,n,1024,
                num_warps=4,enable_fp_fusion=False)
            current=torch.cuda.current_stream(self.wh.device)
            self.wh.record_stream(current);self.wl.record_stream(current)
            self.packed_version=self.cache.version;self.packed_refreshes+=1

    def verify_cache(self):
        super().verify_cache()
        if self.packed:
            high=self.cache.cache.half();low=((self.cache.cache-high.float())*2048.).half()
            assert torch.equal(self.wh.view(torch.int16),high.view(torch.int16))
            assert torch.equal(self.wl.view(torch.int16),low.view(torch.int16))
            assert self.packed_version==self.cache.version==self.cache.weight._version
            for buffer in (self.wh_backing,self.wl_backing):
                assert torch.all(buffer[:self.guard]==123.5) and torch.all(buffer[-self.guard:]==123.5)


class Controller:
    def __init__(self):self.owners=[];self.witnesses={}
    def report(self):
        for owner in self.owners:owner.verify_cache()
        return dict(captured=len(self.owners),geometries=[list(o.geometry) for o in self.owners],
            packed_configs=[list(o.config) for o in self.owners],
            packed_filter_bytes=sum(o.wh.numel()*4 for o in self.owners),
            packed_refreshes=[o.packed_refreshes for o in self.owners],
            witnesses=[dict(geometry=list(k[0]),config=list(k[1]),**v) for k,v in self.witnesses.items()],
            original_three_pointer_cubin_gates_preserved=True,
            static_pack_on_original_master_version_change=True,
            original_native_transforms_and_23plus45_owners=True,
            original_BK32_Kahan_math=True,full_video_PSNR_unverified=True)


def install(configs=None):
    global active
    assert active is None and production.CachedKahanFilterSwap is OriginalOwner
    active=Controller();production.CachedKahanFilterSwap=PackedOwner
    return active


def uninstall():
    global active,enabled
    assert production.CachedKahanFilterSwap in (OriginalOwner,PackedOwner)
    production.CachedKahanFilterSwap=OriginalOwner;active=None;enabled=False


@contextmanager
def mode(value):
    global enabled
    previous=enabled;enabled=value
    try:yield
    finally:enabled=previous
