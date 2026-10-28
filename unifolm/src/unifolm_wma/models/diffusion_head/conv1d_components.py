import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self._inference_gemm = False

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        if self._inference_gemm:
            conv, norm, activation = self.block
            kernel = conv.kernel_size[0]
            padding = conv.padding[0]
            if conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
                raise ValueError("Conv1d GEMM path only supports dense stride-1 convolutions")
            windows = F.pad(x, (padding, padding)).unfold(2, kernel, 1)
            windows = windows.permute(0, 2, 1, 3).reshape(
                x.shape[0], x.shape[2], -1)
            out = F.linear(windows, conv.weight.flatten(1), conv.bias)
            return activation(norm(out.transpose(1, 2)))
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)
