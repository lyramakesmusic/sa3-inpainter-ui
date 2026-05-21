"""Verify MLX SDPA + additive band-mask SWA matches a brute-force torch reference.

The goal: prove that `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask=band_mask)`
with an additive band mask produces the same output as a hand-computed sliding-window
attention. If this passes, our AE fix is sound at the kernel level.
"""

import numpy as np
import mlx.core as mx
import torch
import torch.nn.functional as F


def torch_sliding_window_ref(q, k, v, w_left, w_right):
    """Brute-force ground truth: standard masked attention with an explicit band mask.
    q, k, v: (B, H, T, D) torch tensors in fp32.
    """
    B, H, T, D = q.shape
    scale = D ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale  # (B, H, T, T)
    ii = torch.arange(T)
    jj = torch.arange(T)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -w_left) & (delta <= w_right)
    mask = torch.where(in_band, torch.zeros_like(scores[0, 0]), torch.full_like(scores[0, 0], float("-inf")))
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    return out


def mlx_sliding_window_sdpa(q_np, k_np, v_np, w_left, w_right):
    """Run MLX SDPA with an additive band mask."""
    q = mx.array(q_np)
    k = mx.array(k_np)
    v = mx.array(v_np)
    T = q.shape[-2]
    ii = mx.arange(T)
    jj = mx.arange(T)
    delta = jj[None, :] - ii[:, None]  # (T, T)
    in_band = (delta >= -w_left) & (delta <= w_right)
    # additive mask: 0 inside, -inf outside
    mask = mx.where(in_band, mx.zeros_like(delta).astype(mx.float32), mx.full(delta.shape, -mx.inf, dtype=mx.float32))
    D = q.shape[-1]
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=float(D) ** -0.5, mask=mask)
    mx.eval(out)
    return np.array(out)


def main():
    rng = np.random.default_rng(0)
    # Match the SAME-L decoder shape regime: large T, small window
    B, H, T, D = 1, 4, 256, 64
    w_left, w_right = 17, 17

    q_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    v_np = rng.standard_normal((B, H, T, D)).astype(np.float32)

    # torch ground truth
    qt = torch.from_numpy(q_np); kt = torch.from_numpy(k_np); vt = torch.from_numpy(v_np)
    out_ref = torch_sliding_window_ref(qt, kt, vt, w_left, w_right).numpy()

    # MLX
    out_mlx = mlx_sliding_window_sdpa(q_np, k_np, v_np, w_left, w_right)

    diff = np.abs(out_ref - out_mlx)
    print(f"shape ref={out_ref.shape} mlx={out_mlx.shape}")
    print(f"max abs diff: {diff.max():.2e}")
    print(f"mean abs diff: {diff.mean():.2e}")
    print(f"ref norm: {np.linalg.norm(out_ref):.4f}, mlx norm: {np.linalg.norm(out_mlx):.4f}")
    assert diff.max() < 1e-3, "MLX band-mask SDPA diverges from torch reference"
    print("PASS")


if __name__ == "__main__":
    main()
