import torch


def load_weights_into_qwen3_5(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    if "model.embed_tokens.weight" in params:
        model_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in params:
        model_prefix = "model.language_model"
    else:
        raise KeyError("Could not find embed token weights in checkpoint.")

    def pkey(suffix):
        return f"{model_prefix}.{suffix}"

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params[pkey("embed_tokens.weight")],
        pkey("embed_tokens.weight"),
    )

    n_layers = param_config["n_layers"]
    layer_types = param_config.get("layer_types", ["full_attention"] * n_layers)

    for l in range(n_layers):
        block = model.blocks[l]
        layer_type = layer_types[l]

        if layer_type == "full_attention":
            att = block.attn
            att.W_query.weight = assign(
                att.W_query.weight,
                params[pkey(f"layers.{l}.self_attn.q_proj.weight")],
                pkey(f"layers.{l}.self_attn.q_proj.weight"),
            )
            att.k_proj.weight = assign(
                att.k_proj.weight,
                params[pkey(f"layers.{l}.self_attn.k_proj.weight")],
                pkey(f"layers.{l}.self_attn.k_proj.weight"),
            )
            att.v_proj.weight = assign(
                att.v_proj.weight,
                params[pkey(f"layers.{l}.self_attn.v_proj.weight")],
                pkey(f"layers.{l}.self_attn.v_proj.weight"),
            )
            att.o_proj.weight = assign(
                att.o_proj.weight,
                params[pkey(f"layers.{l}.self_attn.o_proj.weight")],
                pkey(f"layers.{l}.self_attn.o_proj.weight"),
            )
            if hasattr(att, "q_norm") and att.q_norm is not None:
                att.q_norm.weight = assign(
                    att.q_norm.weight,
                    params[pkey(f"layers.{l}.self_attn.q_norm.weight")],
                    pkey(f"layers.{l}.self_attn.q_norm.weight"),
                )
            if hasattr(att, "k_norm") and att.k_norm is not None:
                att.k_norm.weight = assign(
                    att.k_norm.weight,
                    params[pkey(f"layers.{l}.self_attn.k_norm.weight")],
                    pkey(f"layers.{l}.self_attn.k_norm.weight"),
                )

        elif layer_type == "linear_attention":
            lat = block.attn
            lat.dt_bias = assign(
                lat.dt_bias,
                params[pkey(f"layers.{l}.linear_attn.dt_bias")],
                pkey(f"layers.{l}.linear_attn.dt_bias"),
            )
            lat.A_log = assign(
                lat.A_log,
                params[pkey(f"layers.{l}.linear_attn.A_log")],
                pkey(f"layers.{l}.linear_attn.A_log"),
            )
            lat.conv1d.weight = assign(
                lat.conv1d.weight,
                params[pkey(f"layers.{l}.linear_attn.conv1d.weight")],
                pkey(f"layers.{l}.linear_attn.conv1d.weight"),
            )
            lat.norm.weight = assign(
                lat.norm.weight,
                params[pkey(f"layers.{l}.linear_attn.norm.weight")],
                pkey(f"layers.{l}.linear_attn.norm.weight"),
            )
            lat.out_proj.weight = assign(
                lat.out_proj.weight,
                params[pkey(f"layers.{l}.linear_attn.out_proj.weight")],
                pkey(f"layers.{l}.linear_attn.out_proj.weight"),
            )
            lat.in_proj_qkv.weight = assign(
                lat.in_proj_qkv.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_qkv.weight"),
            )
            lat.in_proj_z.weight = assign(
                lat.in_proj_z.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_z.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_z.weight"),
            )
            lat.in_proj_b.weight = assign(
                lat.in_proj_b.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_b.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_b.weight"),
            )
            lat.in_proj_a.weight = assign(
                lat.in_proj_a.weight,
                params[pkey(f"layers.{l}.linear_attn.in_proj_a.weight")],
                pkey(f"layers.{l}.linear_attn.in_proj_a.weight"),
            )

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        block.norm1.weight = assign(
            block.norm1.weight,
            params[pkey(f"layers.{l}.input_layernorm.weight")],
            pkey(f"layers.{l}.input_layernorm.weight"),
        )

        block.ff.gate_proj.weight = assign(
            block.ff.gate_proj.weight,
            params[pkey(f"layers.{l}.mlp.gate_proj.weight")],
            pkey(f"layers.{l}.mlp.gate_proj.weight"),
        )
        block.ff.up_proj.weight = assign(
            block.ff.up_proj.weight,
            params[pkey(f"layers.{l}.mlp.up_proj.weight")],
            pkey(f"layers.{l}.mlp.up_proj.weight"),
        )
        block.ff.down_proj.weight = assign(
            block.ff.down_proj.weight,
            params[pkey(f"layers.{l}.mlp.down_proj.weight")],
            pkey(f"layers.{l}.mlp.down_proj.weight"),
        )
        block.norm2.weight = assign(
            block.norm2.weight,
            params[pkey(f"layers.{l}.post_attention_layernorm.weight")],
            pkey(f"layers.{l}.post_attention_layernorm.weight"),
        )

    model.final_norm.weight = assign(
        model.final_norm.weight,
        params[pkey("norm.weight")],
        pkey("norm.weight"),
    )

    if "lm_head.weight" in params:
        model.final_proj.weight = assign(model.final_proj.weight, params["lm_head.weight"], "lm_head.weight")
    elif pkey("lm_head.weight") in params:
        model.final_proj.weight = assign(model.final_proj.weight, params[pkey("lm_head.weight")], pkey("lm_head.weight"))
    else:
        model.final_proj.weight = model.tok_emb.weight
        print("Model uses weight tying.")