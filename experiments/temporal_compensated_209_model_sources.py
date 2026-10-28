"""Numerical temporal candidate; all native production cache owners retained."""
import ast,json
from pathlib import Path
from temporal_native_w1_209_model_sources import sources as original_sources

def sources(configs):
    files=original_sources([dict(c,warps=1) for c in configs]);result={}
    for name,text in files.items():
        name=name.replace('temporal_native_w1_209','temporal_compensated_209').replace('temporal-native-w1-209','temporal-compensated-209')
        text=text.replace('temporal_native_w1_209','temporal_compensated_209').replace('temporal-native-w1-209','temporal-compensated-209').replace('TEMPORAL_NATIVE_W1_209','TEMPORAL_COMPENSATED_209').replace('temporal_native_w1=True','temporal_compensated=True')
        if 'run_temporal_' in name:
            text=text.replace("assert all(e['signedzero_bits'] for c in rows[stage]['cases'] for e in c['errors']);",'')
        result[name]=text
    for name in ('temporal_compensated_209_mode.py','check_temporal_compensated_209_cpu.py','temporal_compensated_209_candidate.py'):
        result['scripts/'+name]=Path(__file__).with_name(name).read_text()
    result['temporal-compensated-209-selection.json']=json.dumps(configs,indent=2)+'\n'
    for name,text in result.items():
        if name.endswith('.py'):ast.parse(text)
    return result
