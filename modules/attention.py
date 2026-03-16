import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from modules.rmsnorm import RMSNorm, RMSNormGated
from modules.pos_enc import apply_rope


def l2norm(x, dim=-1, eps=1e-6):
    """Unit L2 normalisation without a learnable scale (matches FLA / HF convention)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class GQAttention(nn.Module):
    """
    Grouped Query Attention with partial RoPE and optional sigmoid output gate.
    Used as the 'full_attention' layer in Qwen3.5 (every 4th layer).
    Qwen3.5-0.8B config: n_heads=8, num_groups=2, head_dim=256, rope_dim=64.
    """

    def __init__(self, idim, n_heads, num_groups, head_dim, rope_dim, dtype,
                 attn_output_gate=True):
        super().__init__()
        self.idim = idim
        self.n_heads = n_heads
        self.num_groups = num_groups
        self.head_dim = head_dim
        # Only the first rope_dim features of each head are rotated (partial RoPE).
        # rope_dim = int(head_dim * partial_rotary_factor) = int(256 * 0.25) = 64
        self.rope_dim = rope_dim
        self.group_size = n_heads // num_groups
        self.n_kv_embed = head_dim * num_groups
        self.odim = n_heads * head_dim
        self.scale = head_dim ** -0.5
        self.has_output_gate = attn_output_gate

        # Q and output gate are tied into a single 2× projection (HF convention).
        # With attn_output_gate=False the gate half is unused; use standard 1× proj.
        if attn_output_gate:
            self.W_query = nn.Linear(idim, self.odim * 2, dtype=dtype, bias=False)
        else:
            self.W_query = nn.Linear(idim, self.odim, dtype=dtype, bias=False)
        self.k_proj = nn.Linear(idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.v_proj = nn.Linear(idim, self.n_kv_embed, dtype=dtype, bias=False)
        self.o_proj = nn.Linear(self.odim, idim, dtype=dtype, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, cos, sin, mask=None):
        b, L, _ = x.shape

        # Combined Q + gate projection (2× linear), then split
        q_raw = self.W_query(x)                                  # (B, L, odim[*2])
        if self.has_output_gate:
            q_raw = q_raw.view(b, L, self.n_heads, self.head_dim * 2)
            q, gate = torch.chunk(q_raw, 2, dim=-1)              # each (B, L, H, head_dim)
            gate = gate.reshape(b, L, self.odim)                 # (B, L, odim)
            q = q.transpose(1, 2)                                # (B, H, L, head_dim)
        else:
            q = rearrange(q_raw, 'b l (n d) -> b n l d', n=self.n_heads)
            gate = None

        k = self.k_proj(x)   # (B, L, n_kv_embed)
        v = self.v_proj(x)   # (B, L, n_kv_embed)
        k = rearrange(k, 'b l (g d) -> b g l d', g=self.num_groups)
        v = rearrange(v, 'b l (g d) -> b g l d', g=self.num_groups)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE: rotate only the leading rope_dim features per head
        q = torch.cat([apply_rope(q[..., :self.rope_dim], cos, sin),
                       q[..., self.rope_dim:]], dim=-1)
        k = torch.cat([apply_rope(k[..., :self.rope_dim], cos, sin),
                       k[..., self.rope_dim:]], dim=-1)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots.masked_fill(mask, -torch.inf)
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        if self.has_output_gate:
            out = out * torch.sigmoid(gate)

        return self.o_proj(out)


class GatedDeltaNetAttention(nn.Module):
    """
    Gated DeltaNet linear attention, aligned with HF Qwen3_5GatedDeltaNet.

    Recurrent update rule (per step t):
        M[t] = exp(g[t]) * M[t-1]                           # decay old memory
             + k[t]^T * ((v[t] - M[t-1]@k[t]) * beta[t])  # delta correction
        o[t] = scale * q[t] @ M[t]

    Key differences vs. plain DeltaNet:
      - Input-dependent memory decay via Mamba-style discretisation:
            g[t] = softplus(in_proj_a(x[t]) + dt_bias) * (-exp(A_log))
        exp(g[t]) ∈ (0, 1) acts as a per-head forgetting factor each step.
      - Single causal depthwise conv over concatenated [Q, K, V] + SiLU,
        replacing separate per-branch convs.
      - L2-normalised Q and K (no learnable scale) before the recurrence
        (matches use_qk_l2norm_in_kernel=True in the HF kernel call).
      - RMSNormGated output: rms_norm(o) * silu(z), where z = in_proj_z(x).

    Note: replace the Python loop with torch_chunk_gated_delta_rule or the
    FLA triton kernel for training on long sequences.

    Qwen3.5-0.8B config: n_heads=16, key_head_dim=128, value_head_dim=128, conv_kernel_dim=4.
    """

    def __init__(self, idim, n_heads, key_head_dim, value_head_dim, conv_kernel_dim, dtype):
        super().__init__()
        self.n_heads        = n_heads
        self.key_head_dim   = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim        = n_heads * key_head_dim    # 16 * 128 = 2048
        self.value_dim      = n_heads * value_head_dim  # 16 * 128 = 2048

        # ── Projections ──────────────────────────────────────────────────────
        # Q + K + V combined; all three pass through the single causal conv
        self.in_proj_qkv = nn.Linear(idim, self.key_dim * 2 + self.value_dim,
                                     dtype=dtype, bias=False)
        # Output gate z; used in RMSNormGated as the silu(z) multiplier
        self.in_proj_z   = nn.Linear(idim, self.value_dim, dtype=dtype, bias=False)
        # Per-head sigmoid beta: learning rate that scales the delta correction
        self.in_proj_b   = nn.Linear(idim, n_heads, dtype=dtype, bias=False)
        # Per-head input for Mamba-style step-size (dt) computation
        self.in_proj_a   = nn.Linear(idim, n_heads, dtype=dtype, bias=False)

        # ── Single causal depthwise conv over [Q, K, V] ──────────────────────
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=conv_kernel_dim,
                                padding=conv_kernel_dim - 1, groups=conv_dim,
                                dtype=dtype, bias=False)

        # ── Learned decay parameters (Mamba / GLA style) ─────────────────────
        # A_log = log(|A|);  A = -exp(A_log) < 0
        # dt    = softplus(in_proj_a(x) + dt_bias)  →  positive step size
        # g     = dt * A  →  negative log-decay  ⟹  exp(g) ∈ (0, 1)
        A = torch.empty(n_heads).uniform_(0, 16)
        self.A_log   = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(torch.ones(n_heads, dtype=dtype))

        # ── Output ────────────────────────────────────────────────────────────
        # RMSNorm gated by silu(z), normalises per value-head dimension
        self.norm     = RMSNormGated(value_head_dim, eps=1e-6)
        self.out_proj = nn.Linear(self.value_dim, idim, dtype=dtype, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        # 1. Combined QKV projection → causal depthwise conv → SiLU
        qkv = self.in_proj_qkv(x)                          # (B, L, 2*key_dim + value_dim)
        qkv = self.conv1d(qkv.transpose(1, 2))[..., :L]   # (B, conv_dim, L) – discard right pad
        qkv = F.silu(qkv).transpose(1, 2)                 # (B, L, 2*key_dim + value_dim)

        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # 2. Output gate, beta (learning rate), and input-dependent decay g
        z    = self.in_proj_z(x)                           # (B, L, value_dim)
        beta = torch.sigmoid(self.in_proj_b(x))            # (B, L, H)
        A    = -self.A_log.float().exp()                   # (H,) negative log-decay magnitude
        dt   = F.softplus(self.in_proj_a(x).float() +
                          self.dt_bias.float())             # (B, L, H) positive step sizes
        g    = dt * A                                      # (B, L, H) negative log-decay

        # 3. Reshape to (B, L, H, D) then L2-normalise Q and K (no learnable scale)
        q = l2norm(q.view(B, L, self.n_heads, self.key_head_dim))
        k = l2norm(k.view(B, L, self.n_heads, self.key_head_dim))
        v = v.view(B, L, self.n_heads, self.value_head_dim)

        # 4. Transpose to (B, H, L, D) and cast to float32 for the recurrence
        q    = q.transpose(1, 2).float()                   # (B, H, L, dk)
        k    = k.transpose(1, 2).float()                   # (B, H, L, dk)
        v    = v.transpose(1, 2).float()                   # (B, H, L, dv)
        beta = beta.transpose(1, 2)                        # (B, H, L)
        g    = g.transpose(1, 2)                           # (B, H, L)

        scale = self.key_head_dim ** -0.5

        # 5. Recurrent gated delta-rule (float32 throughout for numerical stability)
        out = torch.zeros(B, self.n_heads, L, self.value_head_dim,
                          device=x.device, dtype=torch.float32)
        M   = torch.zeros(B, self.n_heads, self.key_head_dim, self.value_head_dim,
                          device=x.device, dtype=torch.float32)

        for t in range(L):
            q_t   = q[:, :, t] * scale                          # (B, H, dk)
            k_t   = k[:, :, t]                                  # (B, H, dk)
            v_t   = v[:, :, t]                                  # (B, H, dv)
            b_t   = beta[:, :, t, None].float()                 # (B, H, 1)
            # exp(g) ∈ (0, 1): fraction of memory retained this step
            decay = g[:, :, t].float().exp()[..., None, None]   # (B, H, 1, 1)

            M     = M * decay                                    # forget old memory
            Mk    = einsum('bhd, bhdv -> bhv', k_t, M)          # memory prediction
            M     = M + einsum('bhd, bhv -> bhdv',
                               k_t, (v_t - Mk) * b_t)           # delta update
            out[:, :, t] = einsum('bhd, bhdv -> bhv', q_t, M)  # read out

        # 6. RMSNormGated (norm then silu-gate) then project to model dim
        out = out.to(x.dtype).transpose(1, 2)              # (B, L, H, dv)
        z   = z.view(B, L, self.n_heads, self.value_head_dim)
        out = self.norm(out, z)                            # (B, L, H, dv)
        out = out.reshape(B, L, self.value_dim)
        return self.out_proj(out)