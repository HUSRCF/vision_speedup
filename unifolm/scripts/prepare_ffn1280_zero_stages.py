"""Host-only schedule selection; preserve every pre-existing shader line."""


def transform(text):
    anchor='import torch\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,'import torch; import os\n')
    anchor='    bm=64 if rows>=2560 else 32\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''    mode=os.environ.get('UNIFOLM_WORLD_FFN1280_ZERO_STAGES','2')
    if mode not in {'2','3'}:raise ValueError('Invalid world C1280 FFN depth')
    stages=3 if mode=='3' and (rows,channels,inner) in {(2560,1280,5120),(640,1280,5120)} else 2
''')
    anchor='        num_warps=4,num_stages=2,enable_fp_fusion=False)'
    assert text.count(anchor)==1
    return text.replace(anchor,'        num_warps=4,num_stages=stages,enable_fp_fusion=False)')
