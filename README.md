# Qwen3.5 Playground

A clean PyTorch implementation of the **Qwen3.5-0.8B** language model, built from scratch and aligned with the official [HuggingFace checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B).

## Architecture

Qwen3.5-0.8B uses a **hybrid linear / full attention** design:

| Component | Config |
|---|---|
| Total layers | 24 (6 × [3 linear + 1 full]) |
| Linear attention | GatedDeltaNet (delta-rule with learned decay) |
| Full attention | Grouped Query Attention (GQA) |
| Hidden size | 1024 |
| GQA heads / KV groups | 8 / 2, head_dim=256 |
| GatedDeltaNet heads | 16, key/value head_dim=128 |
| RoPE | Partial (25% of head_dim = 64 dims), θ=10M |
| FFN intermediate | 3584 (SiLU-gated) |
| Vocabulary | 248 320 |

## Project Structure

```
qwen3_5.py              # Model definition (Qwen35Model)
test.py                 # Download weights + run inference
modules/
  attention.py          # GQAttention + GatedDeltaNetAttention
  rmsnorm.py            # RMSNorm (zero-init) + RMSNormGated
  pos_enc.py            # Rotary position encoding
  mapping.py            # HF checkpoint → model weight loader
  sampling.py           # Token sampling / decoding loop
  tokenizer.py          # Qwen3.5 tokenizer wrapper
  llm_utils.py          # Utility helpers
```

## Usage

```python
import torch
from qwen3_5 import Qwen35Model

QWEN3_5_CONFIG = {
    "vocab_size": 248_320, "context_length": 262_144, "emb_dim": 1024,
    "n_layers": 24, "hidden_dim": 3584,
    "n_heads": 8, "n_kv_groups": 2, "head_dim": 256, "rope_dim": 64,
    "linear_n_heads": 16, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "dtype": torch.bfloat16,
}

layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 6

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
    layer_types=layer_types,
    dtype=QWEN3_5_CONFIG["dtype"],
)
```

Run the full inference test (downloads weights from HuggingFace Hub):

```bash
python3 test.py
```

## Key Implementation Details

- **GatedDeltaNet** — linear attention with Mamba-style input-dependent memory decay (`A_log` + `dt_bias`), combined Q/K/V depthwise conv, and `RMSNormGated` output.
- **GQAttention** — standard scaled dot-product attention with partial RoPE, grouped KV heads, and a sigmoid output gate (Q and gate are produced by a single 2× linear projection).
- **Weight loading** — `modules/mapping.py` maps HuggingFace `model.language_model.*` checkpoint keys directly into this implementation's parameter names.

## License

See [LICENSE](LICENSE).
