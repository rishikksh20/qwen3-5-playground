import torch
from torch import nn
from modules.attention import GQAttention, GatedDeltaNetAttention
from torch.nn import functional as F

from modules.llm_utils import model_memory_size
from modules.pos_enc import rope_rotate
from modules.rmsnorm import RMSNorm


class GatedFeedForward(nn.Module):
    def __init__(self, idim, hidden_dim, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.up_proj   = nn.Linear(idim, hidden_dim, dtype=dtype, bias=False)
        self.down_proj = nn.Linear(hidden_dim, idim,  dtype=dtype, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedAttentionBlock(nn.Module):
    """
    Full (quadratic-softmax) GQA block with sigmoid output gate.
    Instantiated every 4th layer (full_attention_interval=4).
    Qwen3.5-0.8B: n_heads=8, num_groups=2, head_dim=256, rope_dim=64.
    """
    def __init__(self, dim, n_heads, num_groups, head_dim, rope_dim, mlp_dim, dtype):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GQAttention(dim, n_heads=n_heads, num_groups=num_groups,
                                 head_dim=head_dim, rope_dim=rope_dim,
                                 dtype=dtype, attn_output_gate=True)
        self.norm2 = RMSNorm(dim)
        self.ff    = GatedFeedForward(dim, mlp_dim, dtype)

    def forward(self, x, cos, sin, mask=None):
        x = self.attn(self.norm1(x), cos, sin, mask) + x
        x = self.ff(self.norm2(x)) + x
        return x


class GatedDeltaNetBlock(nn.Module):
    """
    Linear-attention DeltaNet block with SiLU output gate.
    Occupies the 3 layers preceding each GatedAttentionBlock.
    Qwen3.5-0.8B: n_heads=16, key_head_dim=128, value_head_dim=128, conv_kernel_dim=4.
    """
    def __init__(self, dim, n_heads, key_head_dim, value_head_dim,
                 conv_kernel_dim, mlp_dim, dtype):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = GatedDeltaNetAttention(dim, n_heads=n_heads,
                                            key_head_dim=key_head_dim,
                                            value_head_dim=value_head_dim,
                                            conv_kernel_dim=conv_kernel_dim,
                                            dtype=dtype)
        self.norm2 = RMSNorm(dim)
        self.ff    = GatedFeedForward(dim, mlp_dim, dtype)

    def forward(self, x, cos=None, sin=None, mask=None):
        # DeltaNet does not use positional encoding; cos/sin kept for a
        # uniform block interface so the model loop can call all blocks alike.
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Qwen35Model(nn.Module):
    """
    Qwen3.5 language model backbone.

    Architecture: 6 × (3 × GatedDeltaNet → 1 × GatedAttention) = 24 layers total.
    layer_types list from config drives which block class is used per layer.

    Key differences from Qwen3:
      - Hybrid linear / full attention (3:1 ratio)
      - Gated DeltaNet (delta-rule linear attention) for 18 layers
      - Gated full GQA (softmax) for 6 layers
      - Partial RoPE on full-attention heads (rope_dim = head_dim * 0.25 = 64)
      - Sigmoid output gate on full-attention; SiLU gate on DeltaNet
      - Larger vocab: 248 320
    """

    def __init__(self, dim, depth, n_heads, num_groups, head_dim, rope_dim,
                 linear_n_heads, linear_key_head_dim, linear_value_head_dim,
                 linear_conv_kernel_dim, mlp_dim, vocab_size, context_length,
                 layer_types, dtype=torch.bfloat16):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim, dtype=dtype)

        self.blocks = nn.ModuleList()
        for ltype in layer_types:
            if ltype == 'full_attention':
                self.blocks.append(
                    GatedAttentionBlock(dim, n_heads, num_groups, head_dim,
                                        rope_dim, mlp_dim, dtype))
            else:  # 'linear_attention'
                self.blocks.append(
                    GatedDeltaNetBlock(dim, linear_n_heads, linear_key_head_dim,
                                       linear_value_head_dim, linear_conv_kernel_dim,
                                       mlp_dim, dtype))

        self.final_norm = RMSNorm(dim, eps=1e-6)
        # Weight-tied to tok_emb (tie_word_embeddings=true in config)
        self.final_proj = nn.Linear(dim, vocab_size, bias=False, dtype=dtype)

        # RoPE tables sized for rope_dim only (partial_rotary_factor=0.25 → 64 dims).
        # theta=10_000_000 per Qwen3.5 config (vs 1_000_000 in Qwen3).
        cos, sin = rope_rotate(rope_dim, context_length, theta=10_000_000)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.dtype = dtype

    def forward(self, inp):
        emb = self.tok_emb(inp)
        n   = emb.shape[1]
        mask = torch.triu(torch.ones(n, n, device=inp.device, dtype=torch.bool), diagonal=1)
        x = emb

        for block in self.blocks:
            x = block(x, self.cos, self.sin, mask)

        x = self.final_norm(x)
        x = self.final_proj(x.to(self.dtype))
        return x

if __name__ == "__main__":

    QWEN3_5_CONFIG = {
        "vocab_size":              248_320,
        "context_length":          262_144,
        "emb_dim":                 1024,
        # Full attention (GQA) params
        "n_heads":                 8,
        "n_kv_groups":             2,
        "head_dim":                256,
        "rope_dim":                64,       # head_dim * partial_rotary_factor (0.25)
        # Linear attention (DeltaNet) params
        "linear_n_heads":          16,
        "linear_key_head_dim":     128,
        "linear_value_head_dim":   128,
        "linear_conv_kernel_dim":  4,
        # Shared
        "n_layers":                24,
        "hidden_dim":              3584,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
        "dtype": torch.bfloat16,
    }

    model = Qwen35Model(
        dim=QWEN3_5_CONFIG["emb_dim"],
        depth=QWEN3_5_CONFIG["n_layers"],
        n_heads=QWEN3_5_CONFIG["n_heads"],
        num_groups=QWEN3_5_CONFIG["n_kv_groups"],
        head_dim=QWEN3_5_CONFIG["head_dim"],
        rope_dim=QWEN3_5_CONFIG["rope_dim"],
        linear_n_heads=QWEN3_5_CONFIG["linear_n_heads"],
        linear_key_head_dim=QWEN3_5_CONFIG["linear_key_head_dim"],
        linear_value_head_dim=QWEN3_5_CONFIG["linear_value_head_dim"],
        linear_conv_kernel_dim=QWEN3_5_CONFIG["linear_conv_kernel_dim"],
        mlp_dim=QWEN3_5_CONFIG["hidden_dim"],
        vocab_size=QWEN3_5_CONFIG["vocab_size"],
        context_length=QWEN3_5_CONFIG["context_length"],
        layer_types=QWEN3_5_CONFIG["layer_types"],
        dtype=QWEN3_5_CONFIG["dtype"],
    )
    out = model(torch.tensor([1, 2, 3]).unsqueeze(0))

    print("Model output shape : ", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

    # print("\nModel : \n", model)

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")