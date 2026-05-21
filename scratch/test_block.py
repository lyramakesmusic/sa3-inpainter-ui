"""Smoke test: a single AETransformerBlock runs end-to-end with random weights.

Just shape/dtype/no-NaN check; verifies the building blocks are wired correctly
before we wrestle with weight loading from the safetensors.
"""
import numpy as np
import mlx.core as mx
from mlx_sa3.nn_blocks import AETransformerBlock, sdpa_band_mask


def main():
    B, T, dim = 1, 256, 1536
    block = AETransformerBlock(dim=dim, dim_heads=64, ff_mult=3.0, sinusoidal=False)
    x = mx.random.normal((B, T, dim))
    mask = sdpa_band_mask(T, T, 17, 17)
    y = block(x, sw_mask=mask)
    mx.eval(y)
    arr = np.array(y)
    print("output shape:", arr.shape)
    print("any NaN?", np.isnan(arr).any())
    print("any Inf?", np.isinf(arr).any())
    print("mean abs:", np.abs(arr).mean())
    print("OK" if (not np.isnan(arr).any() and not np.isinf(arr).any()) else "FAIL")


if __name__ == "__main__":
    main()
