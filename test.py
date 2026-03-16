import torch
import os, json
from huggingface_hub import snapshot_download
from qwen3_5 import Qwen35Model
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules.mapping import load_weights_into_qwen3_5
from modules.sampling import advance_decoding
from modules.tokenizer import Qwen3_5Tokenizer


def test_qwen3_5_0_8B(prompt, config):

    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 6  # 24 layers
    config["layer_types"] = layer_types

    model = Qwen35Model(
        dim=config["emb_dim"],
        depth=config["n_layers"],
        n_heads=config["n_heads"],
        num_groups=config["n_kv_groups"],
        head_dim=config["head_dim"],
        rope_dim=config["rope_dim"],
        linear_n_heads=config["linear_n_heads"],
        linear_key_head_dim=config["linear_key_head_dim"],
        linear_value_head_dim=config["linear_value_head_dim"],
        linear_conv_kernel_dim=config["linear_conv_kernel_dim"],
        mlp_dim=config["hidden_dim"],
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        layer_types=layer_types,
        dtype=config["dtype"],
    )
    device = torch.device("cpu")



    repo_id = "Qwen/Qwen3.5-0.8B"
    local_dir = Path(repo_id).parts[-1]

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)


    load_weights_into_qwen3_5(model, config, weights_dict)
    model.to(device)
    del weights_dict

    tokenizer_file_path = "Qwen3.5-0.8B/tokenizer.json"

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )

    tokenizer = Qwen3_5Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=True,
    )


    print(f"Prompt : {prompt}")
    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(f"Decoded Text: {text}")

    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token in advance_decoding(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=8192,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            window_size=50
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )


if __name__ == "__main__":
    prompt = "Please explain the climate change and how it impacts our future."

    QWEN3_5_CONFIG = {
        # Qwen3.5-0.8B text_config values
        "vocab_size":             248_320,
        "context_length":         262_144,
        "emb_dim":                1024,
        "n_layers":               24,
        "hidden_dim":             3584,
        # Full attention (GQA) — every 4th layer
        "n_heads":                8,
        "n_kv_groups":            2,
        "head_dim":               256,
        "rope_dim":               64,      # head_dim * partial_rotary_factor (0.25)
        # Linear attention (GatedDeltaNet) — remaining 3/4 layers
        "linear_n_heads":         16,
        "linear_key_head_dim":    128,
        "linear_value_head_dim":  128,
        "linear_conv_kernel_dim": 4,
        "dtype": torch.bfloat16,
    }
    test_qwen3_5_0_8B(prompt, QWEN3_5_CONFIG)