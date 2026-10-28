"""V2-only numerical candidate with original native launch as the default path."""
import ast
from pathlib import Path
import temporal_native_w1_209_sources as native

FLAG='UNIFOLM_TEMPORAL_THREE_PRODUCT'
SOURCE=native.SOURCE
KERNEL='src/unifolm_wma/utils/triton_temporal_three_product.py'
MANIFEST='temporal-three-product-production-sources.json'

def kernel_source():return Path(__file__).with_name('temporal_compensated_209_candidate.py').read_text()

def source(old):
    text=native.source(old).replace(native.FLAG,FLAG).replace('_native_temporal_warps','_temporal_three_product_warps')
    text=text.replace('return (1 if tuple(query.shape)','return (2 if tuple(query.shape)')
    begin=text.index('        _attention_fp32[(');end=text.index('    return (output, qk, probabilities)',begin)
    text=text[:begin]+'''        if _temporal_three_product_warps(query, heads, scale) == 2:
            from .triton_temporal_three_product import launch
            launch(query, key, value, output, qk, probabilities, heads, scale, 2, return_intermediates)
        else:
            _attention_fp32[(query.shape[0] * heads,)](
                query, key, value, output, qk, probabilities, heads, scale,
                return_intermediates, num_warps=4, enable_fp_fusion=False)
'''+text[end:]
    ast.parse(text);return text

def provenance_source(old):return native.provenance_source(old).replace(native.FLAG,FLAG)

def cpu_checker():
    text=native.cpu_checker().replace(native.FLAG,FLAG).replace('temporal-native-w1-production-cpu.json','temporal-three-product-production-cpu.json')
    text=text.replace("from unifolm_wma.utils import triton_temporal_attention_fp32 as r", "from unifolm_wma.utils import triton_temporal_attention_fp32 as r\nfrom unifolm_wma.utils import triton_temporal_three_product as candidate")
    text=text.replace("with patch.object(r,'_attention_fp32',Kernel()),patch.object(r,'empty_attention_output',lambda kind,q:q):", "with patch.object(r,'_attention_fp32',Kernel()),patch.object(r,'empty_attention_output',lambda kind,q:q),patch.object(candidate,'launch',private_launch):")
    text=text.replace('calls=[]', "calls=[]\ndef private_launch(*args):\n    q=args[0];calls.append(((q.shape[0]*args[6],),args[:8]+(args[9],),{'num_warps':args[8],'enable_fp_fusion':False,'private_candidate':True}))")
    text=text.replace("kwargs=={'num_warps':1,'enable_fp_fusion':False}","kwargs=={'num_warps':2,'enable_fp_fusion':False,'private_candidate':True}")
    text=text.replace('only_warps_changed=True','private_candidate_inputs_grid_storage_scale_preserved=True,original_native_default=True')
    ast.parse(text);return text
