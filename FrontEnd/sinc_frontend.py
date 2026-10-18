"""SincNet front end with the same contract as Leaf and Run9's MelFrontend:
(B, 1, T) -> (B, n_filters, frames).

Follows the SincNet baseline from the LEAF paper (leaf-audio's SincNet):
sinc band-pass conv -> LeakyReLU -> max-pool -> LayerNorm over the filters.
The sinc layer is the official one (SincNet/dnn_models.py); only the low cut-off
and bandwidth of each filter (in Hz) are learned.
"""
import torch
import torch.nn.functional as F
from torch import nn

from SincNet.dnn_models import SincConv_fast


class SincNetFrontend(nn.Module):
    def __init__(self, n_filters=80, sample_rate=16000, window_len=25.0, window_stride=10.0):
        super().__init__()
        self.win = int(sample_rate * window_len // 1000 + 1)  # 401 at 25 ms, same as Leaf
        self.hop = int(sample_rate * window_stride // 1000)
        self.sinc = SincConv_fast(n_filters, self.win, sample_rate=sample_rate, padding=self.win // 2)
        self.norm = nn.LayerNorm(n_filters)

    def forward(self, x):
        x = F.leaky_relu(self.sinc(x), 0.2)
        x = F.max_pool1d(x, self.win, self.hop, padding=self.win // 2)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


if __name__ == "__main__":  # python FrontEnd/sinc_frontend.py (from the project root)
    fe = SincNetFrontend(n_filters=64)
    x = torch.randn(2, 1, 16000 * 3)
    y = fe(x)
    assert y.shape == (2, 64, 300), y.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert fe.sinc.low_hz_.grad.abs().sum() > 0 and fe.sinc.band_hz_.grad.abs().sum() > 0
    print("ok", tuple(y.shape))
