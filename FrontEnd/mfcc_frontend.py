import torch
import math

def get_dct_matrix(n_filters, n_mfcc, device):
    """Generates a Type-II DCT matrix for pure PyTorch tensor operations.
    
    Args:
        n_filters (int): Number of frequency channels/filters in the input spectrogram.
        n_mfcc (int): Desired number of MFCC coefficients to retain.
        device (torch.device or str): Device to allocate the tensor on.
        
    Returns:
        torch.Tensor: DCT-II transformation matrix of shape (n_mfcc, n_filters).
    """
    n = torch.arange(n_filters, dtype=torch.float32, device=device)
    k = torch.arange(n_mfcc, dtype=torch.float32, device=device).unsqueeze(1)
    
    # DCT-II formula with orthogonal scaling factors
    dct_mat = torch.cos(torch.pi * k * (2 * n + 1) / (2 * n_filters))
    dct_mat *= math.sqrt(1.0 / n_filters)
    dct_mat[1:] *= math.sqrt(2.0 / n_filters)
    return dct_mat
