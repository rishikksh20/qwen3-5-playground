from torch import nn
import torch
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        # Qwen3.5 uses zero-init weight with (1 + weight) scaling
        self.weight = nn.Parameter(torch.zeros(n_embed))
        self.variance_epsilon = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        input_dtype = x.dtype
        x_norm = self._norm(x.float())
        x_norm = x_norm * (1.0 + self.weight.float())
        return x_norm.to(input_dtype)


class RMSNormGated(nn.Module):
    """
    RMSNorm followed by an element-wise SiLU gate.
    Matches Qwen3_5RMSNormGated from the HF implementation.
    forward(x, gate) → scale * rms_norm(x) * silu(gate)  (all cast to input dtype)
    """
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        return (x * F.silu(gate.to(torch.float32))).to(input_dtype)