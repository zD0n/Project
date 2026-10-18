import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_cqt_filterbank(sr, fmin, n_bins, bins_per_octave, device):
    """Generates complex CQT kernel weights for a PyTorch Conv1d layer."""
    q = 1.0 / (2**(1.0 / bins_per_octave) - 1.0)
    frequencies = fmin * (2 ** (torch.arange(n_bins, dtype=torch.float32, device=device) / bins_per_octave))
    
    max_len = int(math.ceil(q * sr / fmin))
    if max_len % 2 == 0:
        max_len += 1
        
    kernel_real = torch.zeros((n_bins, max_len), device=device)
    kernel_imag = torch.zeros((n_bins, max_len), device=device)
    
    for i, freq in enumerate(frequencies):
        n_samples = int(math.ceil(q * sr / freq.item()))
        if n_samples % 2 == 0:
            n_samples += 1
        if n_samples > max_len:
            n_samples = max_len
            
        start = (max_len - n_samples) // 2
        t = torch.arange(n_samples, dtype=torch.float32, device=device) - (n_samples - 1) / 2
        
        window = 0.5 * (1 - torch.cos(2 * torch.pi * torch.arange(n_samples, device=device) / (n_samples - 1)))
        phase = 2 * torch.pi * freq * (t / sr)
        
        kernel_real[i, start:start+n_samples] = window * torch.cos(phase) / n_samples
        kernel_imag[i, start:start+n_samples] = window * torch.sin(phase) / n_samples
        
    kernel = torch.stack([kernel_real, kernel_imag], dim=1).view(2 * n_bins, 1, max_len)
    return kernel

class PyTorchCQT(nn.Module):
    def __init__(self, sr=22050, fmin=32.7, n_bins=84, bins_per_octave=12, hop_length=512, device="cuda"):
        super().__init__()
        self.hop_length = hop_length
        self.n_bins = n_bins
        
        kernel = get_cqt_filterbank(sr, fmin, n_bins, bins_per_octave, device)
        self.register_buffer("kernel", kernel)
        self.padding = kernel.shape[-1] // 2
        
    def forward(self, x):
        # x shape: (B, 1, samples)
        cqt_conv = F.conv1d(x, self.kernel, padding=self.padding, stride=self.hop_length)
        
        # (B, 2 * n_bins, frames) -> (B, n_bins, 2, frames)
        cqt_conv = cqt_conv.view(x.shape[0], self.n_bins, 2, -1)
        
        # Calculate magnitude: sqrt(real^2 + imag^2)
        magnitude = torch.sqrt(cqt_conv[:, :, 0, :]**2 + cqt_conv[:, :, 1, :]**2 + 1e-8)
        return magnitude # Shape: (B, n_bins, frames)
