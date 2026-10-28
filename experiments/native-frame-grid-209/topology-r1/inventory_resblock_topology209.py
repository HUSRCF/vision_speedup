"""Infer ResBlock convolution geometries from the actual configured modules.

Construct only the backbone on the meta device. Head factories are replaced
by Identity because no head is inspected or executed. No weights or input
operands are loaded and no CUDA context is created. Runtime strides and native
consumer choices require a separate GPU check.
"""
import json
from unittest.mock import patch
import torch
import torch.utils.checkpoint
from omegaconf import OmegaConf
from unifolm_wma.modules.networks import wma_model as network


def main():
    assert not torch.cuda.is_initialized()
    config = OmegaConf.load('configs/inference/world_model_interaction.yaml')
    params = OmegaConf.to_container(config.model.params.wma_config.params, resolve=True)
    with torch.device('meta'), patch.object(network, 'instantiate_from_config',
                                          side_effect=lambda config: torch.nn.Identity()):
        model = network.WMAModel(**params).eval()
    assert all(p.device.type == 'meta' for p in model.parameters())
    rows = []

    def block(name, module, shape):
        c, h, w = shape
        if type(module) is network.ResBlock:
            assert c == module.channels
            if isinstance(module.h_upd, network.Upsample):
                h, w = 2*h, 2*w
            elif isinstance(module.h_upd, network.Downsample):
                h, w = h//2, w//2
            for suffix, conv in [('in_layers.2', module.in_layers[2]),
                                 ('out_layers.3', module.out_layers[3]),
                                 ('skip_connection', module.skip_connection)]:
                if not isinstance(conv, torch.nn.Conv2d):
                    continue
                assert conv.in_channels == (module.out_channels if suffix == 'out_layers.3' else c)
                rows.append(dict(name=name+'.'+suffix, co=conv.out_channels,
                    ci=conv.in_channels, height=h, width=w, kernel=list(conv.kernel_size),
                    stride=list(conv.stride), padding=list(conv.padding), full_frames=16))
            return module.out_channels, h, w
        if isinstance(module, network.Downsample):
            return module.out_channels, h//2, w//2
        if isinstance(module, network.Upsample):
            return module.out_channels, h*2, w*2
        if isinstance(module, torch.nn.Conv2d):
            assert c == module.in_channels
            assert module.stride == (1, 1)
            return module.out_channels, h, w
        return shape

    shape = (params['in_channels'], 40, 64)
    skips = []
    for i, sequence in enumerate(model.input_blocks):
        for j, module in enumerate(sequence):
            shape = block(f'input_blocks.{i}.{j}', module, shape)
        skips.append(shape)
    for j, module in enumerate(model.middle_block):
        shape = block(f'middle_block.{j}', module, shape)
    for i, sequence in enumerate(model.output_blocks):
        other = skips.pop()
        assert shape[1:] == other[1:]
        shape = (shape[0]+other[0], *shape[1:])
        for j, module in enumerate(sequence):
            shape = block(f'output_blocks.{i}.{j}', module, shape)
    assert not skips and shape == (320, 40, 64)
    resblocks = sorted({r['name'].rsplit('.', 2)[0] for r in rows if '.in_layers.' in r['name']})
    assert len(resblocks) == 22
    geometries = sorted({(r['co'], r['ci'], r['height'], r['width'])
                         for r in rows if r['kernel'] == [3, 3]})
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(meta_topology_complete=True, original_ResBlocks=len(resblocks),
        rows=rows, spatial_3x3_geometries=[list(g) for g in geometries],
        no_checkpoint_or_input_loaded=True, no_GPU_initialized=True,
        native_consumer_and_runtime_strides_unverified=True), indent=2))


if __name__ == '__main__':
    main()
