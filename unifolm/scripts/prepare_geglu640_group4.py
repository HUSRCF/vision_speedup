"""Host dispatch only; existing shader and all FP32 arithmetic stay unchanged."""


def transform(text):
    anchor='import torch\n';assert text.count(anchor)==1
    text=text.replace(anchor,'import os\nimport torch\n')
    split=text.index('def packed_hidden_ffn_residual(')
    prefix,suffix=text[:split],text[split:]
    anchor='    outputs=tuple(empty_geglu((rows,channels),device=high.device,dtype=torch.float16) for _ in range(2))'
    assert prefix.count(anchor)==1
    prefix=prefix.replace(anchor,'''    mode=os.environ.get('UNIFOLM_WORLD_GEGLU_C640_GROUP','32')
    if mode not in {'32','4'}:raise ValueError('Invalid world C640 GEGLU group')
    group=4 if (mode=='4' and (rows,channels,inner)==(10240,2560,640)
                and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32) else 32
'''+anchor)
    anchor='rows,channels,inner,bias is not None,64,64,64,32,False,True,False,'
    assert prefix.count(anchor)==1
    return prefix.replace(anchor,'rows,channels,inner,bias is not None,64,64,64,group,False,True,False,')+suffix
