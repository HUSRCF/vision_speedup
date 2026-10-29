"""Package original-kernel Half FFN partitioning with timed two-graph ownership."""
import ast,hashlib
from pathlib import Path
from experiment_remote import WORKSPACE
import build_nonv2_ffn61_sources as previous

ENTRY=previous.ENTRY
RUNTIME='src/unifolm_wma/utils/nccl_nonv2_native_ffn_runtime64.py'
MODE='src/unifolm_wma/utils/nonv2_native_ffn_grid59_mode.py'
POLICY='nonv2-native-ffn-grid'
CASE='unitree_g1_pack_camera'

ADAPTER='''class ProductionController(Controller):
    def bind(self,graph):
        rows,records,owners=self.rows,self.records,self.owners
        batches=len(self.binding_reports)
        assert len(rows)==31*(batches+1) and len(records)==62*(batches+1)
        assert len(owners)==62*batches
        self.rows=rows[-31:];self.records=records[-62:];self.owners=[]
        try:
            super().bind(graph)
        finally:
            created=self.owners
            self.rows=rows;self.records=records;self.owners=owners+created

'''

def production_sources(base):
    old,proof=previous.production_sources(base)
    runtime=old[previous.RUNTIME].replace('nonv2-ffn',POLICY).replace('unitree_z1_stackbox',CASE).replace('from .nonv2_ffn59_mode import','from .nonv2_native_ffn_grid59_geglu_mode import')
    runtime=runtime.replace('    _denoiser_graph=original;calls=0','    _denoiser_graph=original;calls=0;bound_graphs=set()')
    anchor='        with _mode(True):result=original(*args,**kwargs)\n'
    assert runtime.count(anchor)==1
    runtime=runtime.replace(anchor,anchor+'''        if len(_controller.rows)!=before:
            unbound=[entry['graph'] for entry in original.entries.values()
                     if id(entry['graph']) not in bound_graphs]
            assert len(unbound)==1
            _controller.bind(unbound[0]);bound_graphs.add(id(unbound[0]))
            # A fresh replay returns cloned outputs from the now-bound graph.
            # This extra initialization work remains inside the process timer.
            with _mode(True):result=original(*args,**kwargs)
''')
    runtime=runtime.replace('dict(decision=33,world=33)','dict(decision=31,world=31)').replace("report['captured']==66","report['captured']==62")
    anchor="    assert all(row['collectives']==1 for row in report['rows'])\n"
    assert runtime.count(anchor)==1
    runtime=runtime.replace(anchor,anchor+"    assert report['GEGLU']['captured']==62 and report['native_linear_capture_count']==124\n    assert len(report['native_binding'])==2\n")
    anchor="    for entry in _denoiser_graph.entries.values():\n"
    assert runtime.count(anchor)==1
    runtime=runtime.replace(anchor,"    _controller.close_owners()\n"+anchor)
    entry=old[ENTRY].replace('nccl_nonv2_ffn_runtime61','nccl_nonv2_native_ffn_runtime64')
    scripts=Path(__file__).parent
    mode=(scripts/'nonv2_native_ffn_grid59_mode.py').read_text()
    mode=mode.replace('from half_native_owned.native_graph_weight_cache_probe import','from .native_graph_weight_cache_probe import')
    mode=mode.replace("Path(__file__).parent.parent/'qualified-shapes.json'","Path(__file__).parent/'qualified-half-native-ffn64.json'")
    mode=mode.replace("Path(os.environ['NATIVE_GRAPH_HANDLES_PATH'])","Path(__file__).parent/'native-graph-handles/team07_cuda_graph_handles.so'")
    geglu=(scripts/'nonv2_native_ffn_grid59_geglu_mode.py').read_text().replace('import nonv2_native_ffn_grid59_mode as adopted','from . import nonv2_native_ffn_grid59_mode as adopted')
    anchor='def install(model):\n';assert geglu.count(anchor)==1
    geglu=geglu.replace(anchor,ADAPTER+'\n'+anchor).replace('active=Controller(model);','active=ProductionController(model);')
    files={ENTRY:entry,RUNTIME:runtime,MODE:mode,'src/unifolm_wma/utils/nonv2_native_ffn_grid59_geglu_mode.py':geglu}
    for name in ('native_graph_weight_cache_probe.py','cuda_graph_readonly_inventory.py'):
        files['src/unifolm_wma/utils/'+name]=(WORKSPACE/'vision_speedup/unifolm/src/unifolm_wma/utils'/name).read_text()
    for text in files.values():ast.parse(text)
    proof.update(source59_to61=False,source59_to64=True,original_Half_vendor_GEMM_and_GEGLU_CUBIN_preserved=True,
                 native_GEMM_owned_tensor_binding=True,no_new_math_JIT=True,two_graph_ownership_and_restore_before_reset=True,
                 qualified_FFNs_per_stage=31,excluded_small_M640_FFNs=2,all_bind_capture_and_extra_replay_inside_process_timer=True)
    proof['numerical_FFN_helper_SHA256']=hashlib.sha256(mode.encode()).hexdigest()
    return files,proof

if __name__=='__main__':
    f,p=production_sources(WORKSPACE/'vision_speedup/experiments/nonv2-vae-all-display/unifolm')
    print('NONV2_NATIVE_FFN64_BUILDER passed=true files='+str(len(f)))
