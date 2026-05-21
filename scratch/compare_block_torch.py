"""Layer-level parity check: a single AE TransformerBlock, MLX vs torch sa3, same weights.

Loads decoder.layers.3.transformers.0 from sa3 medium safetensors, builds both:
  - torch: stable_audio_3.models.transformer.TransformerBlock (built via sa3's config)
  - mlx: our AETransformerBlock

Runs the same input through both, compares.
Important: torch path here will use the SDPA SWA fallback that allegedly breaks audio.
For a *short* T like 256 the fallback may behave OK structurally though it may still
diverge at edges. This is a sanity check on weight loading + arch matching, not a
proof-of-correctness vs a flash-attn ground truth (we don't have flash-attn locally).
"""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import json
import numpy as np
import torch
import mlx.core as mx
from safetensors import safe_open

# torch sa3
from stable_audio_3.models.transformer import TransformerBlock as TorchTB

# our mlx
from mlx_sa3.nn_blocks import AETransformerBlock, sdpa_band_mask


CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
PREFIX = "pretransform.model.decoder.layers.3.transformers.0."


def main():
    # 1. Load just block-0 tensors from safetensors
    raw = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                raw[k[len(PREFIX):]] = f.get_tensor(k)

    print(f"loaded {len(raw)} tensors for block 0")

    # 2. Build torch TransformerBlock with the AE block's config
    #    From research_log: norm_type=dyt, attn_kwargs={qk_norm:dyt, differential:True},
    #    ff_kwargs={mult:3, no_bias:False}, add_rope=True, cross_attend=False, dim=1536, dim_heads=64
    dim = 1536; dim_heads = 64
    tb_torch = TorchTB(
        dim=dim,
        dim_heads=dim_heads,
        cross_attend=False,
        causal=False,
        zero_init_branch_outputs=True,
        add_rope=True,
        layer_scale=False,
        norm_type="dyt",
        attn_kwargs={"qk_norm": "dyt", "differential": True},
        ff_kwargs={"mult": 3, "no_bias": False, "sinusoidal": False},
        norm_kwargs={"eps": 1e-3},
    )
    # Strict load — fail loud on mismatch
    msd = tb_torch.state_dict()
    matched = 0
    missing = []
    for k, v in raw.items():
        if k in msd:
            assert msd[k].shape == v.shape, f"shape mismatch on {k}: {msd[k].shape} vs {v.shape}"
            msd[k] = v
            matched += 1
        else:
            missing.append(k)
    print(f"matched {matched}/{len(raw)} torch params; missing: {missing}")
    tb_torch.load_state_dict(msd, strict=False)
    tb_torch.eval()

    # 3. Build MLX block; load weights via direct attribute update
    tb_mlx = AETransformerBlock(dim=dim, dim_heads=dim_heads, ff_mult=3.0, sinusoidal=False)
    # Construct the parameter tree
    mlx_params = {
        "pre_norm": {"alpha": mx.array(raw["pre_norm.alpha"].numpy()),
                      "beta": mx.array(raw["pre_norm.beta"].numpy()),
                      "gamma": mx.array(raw["pre_norm.gamma"].numpy())},
        "ff_norm": {"alpha": mx.array(raw["ff_norm.alpha"].numpy()),
                     "beta": mx.array(raw["ff_norm.beta"].numpy()),
                     "gamma": mx.array(raw["ff_norm.gamma"].numpy())},
        "self_attn": {
            "to_qkv": {"weight": mx.array(raw["self_attn.to_qkv.weight"].numpy())},
            "to_out": {"weight": mx.array(raw["self_attn.to_out.weight"].numpy())},
            "q_norm": {"alpha": mx.array(raw["self_attn.q_norm.alpha"].numpy()),
                        "beta": mx.array(raw["self_attn.q_norm.beta"].numpy()),
                        "gamma": mx.array(raw["self_attn.q_norm.gamma"].numpy())},
            "k_norm": {"alpha": mx.array(raw["self_attn.k_norm.alpha"].numpy()),
                        "beta": mx.array(raw["self_attn.k_norm.beta"].numpy()),
                        "gamma": mx.array(raw["self_attn.k_norm.gamma"].numpy())},
        },
        "ff": {
            "ff_0": {"proj": {"weight": mx.array(raw["ff.ff.0.proj.weight"].numpy()),
                                "bias": mx.array(raw["ff.ff.0.proj.bias"].numpy())}},
            "ff_2": {"weight": mx.array(raw["ff.ff.2.weight"].numpy()),
                      "bias": mx.array(raw["ff.ff.2.bias"].numpy())},
        },
        "rope": {"inv_freq": mx.array(raw["rope.inv_freq"].numpy())},
    }
    tb_mlx.update(mlx_params)

    # 4. Run identical input through both. Use a packed-seq-shape consistent with the AE:
    #    sub_chunk_size=17, latent_count=8 -> T=136
    B, T = 1, 136
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, T, dim)).astype(np.float32) * 0.1

    # window for AE: [1,1] * 17 = [17, 17]
    wl, wr = 17, 17

    # torch forward
    x_t = torch.from_numpy(x_np)
    with torch.no_grad():
        y_t = tb_torch(x_t, self_attention_flash_sliding_window=[wl, wr])
    out_t = y_t.numpy()

    # mlx forward
    x_m = mx.array(x_np)
    sw_mask = sdpa_band_mask(T, T, wl, wr, dtype=mx.float32)
    y_m = tb_mlx(x_m, sw_mask=sw_mask)
    mx.eval(y_m)
    out_m = np.array(y_m)

    diff = np.abs(out_t - out_m)
    print(f"out_t shape: {out_t.shape}, out_m shape: {out_m.shape}")
    print(f"max abs diff: {diff.max():.4e}")
    print(f"mean abs diff: {diff.mean():.4e}")
    print(f"torch norm: {np.linalg.norm(out_t):.4f}, mlx norm: {np.linalg.norm(out_m):.4f}")
    # rel
    print(f"max rel diff: {(diff / (np.abs(out_t) + 1e-6)).max():.4e}")

    # cosine similarity
    cs = (out_t * out_m).sum() / (np.linalg.norm(out_t) * np.linalg.norm(out_m))
    print(f"cosine sim: {cs:.6f}")


if __name__ == "__main__":
    main()
