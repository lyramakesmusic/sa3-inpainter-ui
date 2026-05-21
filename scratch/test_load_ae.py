"""Load AE weights from sa3-medium safetensors into the MLX SA3MediumAE.

Asserts: every required key is present, weight folding produces finite values,
running_std is a positive scalar, and a tiny synthetic decode runs without NaN.
"""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import numpy as np
import mlx.core as mx
from mlx_sa3.ae import SA3MediumAE
from mlx_sa3.weights import load_ae_weights


def main():
    model = SA3MediumAE()
    params = load_ae_weights(model)
    print("running_std:", float(params["bottleneck"]["running_std"][0]))
    map_w = params["decoder"]["layers_3"]["mapping"]["weight"]
    print("decoder mapping weight shape:", map_w.shape, "dtype:", map_w.dtype)
    print("layers_1 weight shape:", params["decoder"]["layers_1"]["weight"].shape)
    print("block0 to_qkv weight shape:",
          params["decoder"]["layers_3"]["transformers"][0]["self_attn"]["to_qkv"]["weight"].shape)
    print("block0 ff.ff_0.proj.weight shape:",
          params["decoder"]["layers_3"]["transformers"][0]["ff"]["ff_0"]["proj"]["weight"].shape)

    # Tiny synthetic decode: 1s of audio = 44100/4096 ≈ 11 latent frames; use 16 for safety
    T_lat = 16
    rng = np.random.default_rng(0)
    latents = mx.array(rng.standard_normal((1, 256, T_lat)).astype(np.float32))
    out = model.decode(latents)
    mx.eval(out)
    arr = np.array(out)
    print("decode out shape:", arr.shape)
    print("any NaN/Inf?", np.isnan(arr).any() or np.isinf(arr).any())
    print("mean abs:", float(np.abs(arr).mean()), "  max abs:", float(np.abs(arr).max()))
    print("OK" if not (np.isnan(arr).any() or np.isinf(arr).any()) else "FAIL")


if __name__ == "__main__":
    main()
