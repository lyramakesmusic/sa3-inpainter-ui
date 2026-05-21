"""MLX building blocks matching the stable-audio-3 transformer.

Faithful re-implementations of the torch classes used by the SAME-L (TAAE_v2)
autoencoder transformer blocks. Differential attention + DyT norm + RoPE +
GLU feedforward, with SWA realized via an additive band mask passed to
`mx.fast.scaled_dot_product_attention`.
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class DynamicTanh(nn.Module):
    """tanh(alpha * x) * gamma + beta. Matches torch DynamicTanh."""

    def __init__(self, dim: int, init_alpha: float = 4.0):
        super().__init__()
        self.alpha = mx.ones((1,)) * init_alpha
        self.gamma = mx.ones((dim,))
        self.beta = mx.zeros((dim,))

    def __call__(self, x):
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class RotaryEmbedding(nn.Module):
    """RoPE freqs precomputed on demand. Matches torch RotaryEmbedding (no xpos, no scale)."""

    def __init__(self, dim: int, base: float = 10000.0, base_rescale_factor: float = 1.0):
        super().__init__()
        base = base * (base_rescale_factor ** (dim / (dim - 2)))
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def forward_from_seq_len(self, seq_len: int):
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.einsum("i,j->ij", t, self.inv_freq)
        # match torch: concat (freqs, freqs) along -1 so apply_rope can split as (cos, sin) of half-dim
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs


def rotate_half(x):
    # x has even last dim D; split into (x1, x2) of D/2 each, return concat(-x2, x1)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(t, freqs):
    """Apply rotary to t along its sequence axis.

    t: (..., n, d) with d = head_dim
    freqs: (n, rot_dim)  -- rot_dim == d in our config since RotaryEmbedding(dim_heads//2) → inv_freq[d/4]
                              and freqs = concat(freqs, freqs) gives last dim = d/2; here we apply to first d/2 of t
    """
    out_dtype = t.dtype
    # promote to fp32 for stability
    t_f32 = t.astype(mx.float32)
    freqs_f32 = freqs.astype(mx.float32)
    rot_dim = freqs_f32.shape[-1]
    seq_len = t_f32.shape[-2]
    freqs_f32 = freqs_f32[-seq_len:, :]

    # Broadcast freqs to t shape: (..., n, d). t has heads dim so we need to expand.
    # Torch path: if t.ndim==4 and freqs.ndim==3: freqs = rearrange(freqs, 'b n d -> b 1 n d')
    # MLX: freqs is (n, d) in our usage; we just rely on broadcasting since t's trailing dims are (..., n, d)
    # The rot_dim slice + concat handles partial rotation if rot_dim < d.
    t_rot = t_f32[..., :rot_dim]
    t_pass = t_f32[..., rot_dim:]

    cos = mx.cos(freqs_f32)
    sin = mx.sin(freqs_f32)
    t_rot = (t_rot * cos) + (rotate_half(t_rot) * sin)
    out = mx.concatenate([t_rot, t_pass], axis=-1)
    return out.astype(out_dtype)


def sdpa_band_mask(seq_q: int, seq_k: int, w_left: int, w_right: int, dtype=mx.float32):
    """Additive band mask: 0 inside [-w_left, +w_right] from each query, -inf outside.

    This is the SWA fix: passing this to mx.fast.scaled_dot_product_attention reproduces
    flash-attn's window_size semantics for self-attention with matching q/k lengths.
    """
    ii = mx.arange(seq_q)
    jj = mx.arange(seq_k)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -w_left) & (delta <= w_right)
    mask = mx.where(in_band, mx.zeros(delta.shape, dtype=dtype), mx.full(delta.shape, -mx.inf, dtype=dtype))
    return mask


class DifferentialAttention(nn.Module):
    """Differential self-attention with DyT qk-norm, RoPE, optional SWA via additive mask.

    Matches the AE TransformerResamplingBlock attention. Self-attend only (cross_attend=False).
    """

    def __init__(self, dim: int, dim_heads: int = 64, qk_norm: str = "dyt"):
        super().__init__()
        assert dim % dim_heads == 0
        self.dim = dim
        self.dim_heads = dim_heads
        self.num_heads = dim // dim_heads

        # Differential: to_qkv produces 5 * dim (q, k, v, q_diff, k_diff)
        self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        if qk_norm == "dyt":
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)
        else:
            raise NotImplementedError(f"qk_norm={qk_norm} not implemented for AE block")

    def __call__(self, x, rotary_freqs=None, sw_mask=None):
        B, T, _ = x.shape
        H, D = self.num_heads, self.dim_heads

        qkv = self.to_qkv(x)  # (B, T, 5*dim)
        q, k, v, q_diff, k_diff = mx.split(qkv, 5, axis=-1)

        # (B, T, H*D) -> (B, H, T, D)
        def shape_heads(t):
            return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        q = shape_heads(q); k = shape_heads(k); v = shape_heads(v)
        q_diff = shape_heads(q_diff); k_diff = shape_heads(k_diff)

        # qk-norm per head
        q = self.q_norm(q); k = self.k_norm(k)
        q_diff = self.q_norm(q_diff); k_diff = self.k_norm(k_diff)

        # RoPE: apply to q/k (and q_diff/k_diff). Freqs cover head_dim.
        if rotary_freqs is not None:
            q = apply_rotary_pos_emb(q, rotary_freqs)
            k = apply_rotary_pos_emb(k, rotary_freqs)
            q_diff = apply_rotary_pos_emb(q_diff, rotary_freqs)
            k_diff = apply_rotary_pos_emb(k_diff, rotary_freqs)

        scale = float(D) ** -0.5

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=sw_mask)
        out_diff = mx.fast.scaled_dot_product_attention(q_diff, k_diff, v, scale=scale, mask=sw_mask)
        out = out - out_diff  # (B, H, T, D)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.to_out(out)


class GLU(nn.Module):
    """SwiGLU / SinGLU: x = proj(x); a, gate = chunk(2); return a * act(gate)."""

    def __init__(self, dim_in: int, dim_out: int, activation: str = "silu"):
        super().__init__()
        # proj has bias per safetensors (ff.0.proj.bias present)
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)
        if activation == "silu":
            self._act = lambda g: g * mx.sigmoid(g)
        elif activation == "sin":
            self._act = lambda g: mx.sin(math.pi * g)
        else:
            raise NotImplementedError(activation)

    def __call__(self, x):
        x = self.proj(x)
        a, gate = mx.split(x, 2, axis=-1)
        return a * self._act(gate)


class FeedForward(nn.Module):
    """FF GLU: dim -> inner=dim*mult (via GLU) -> dim. Matches torch FeedForward(glu=True)."""

    def __init__(self, dim: int, mult: float = 3.0, sinusoidal: bool = False):
        super().__init__()
        inner = int(dim * mult)
        act = "sin" if sinusoidal else "silu"
        # nn.Sequential mapping in torch: ff[0]=GLU, ff[1]=Identity, ff[2]=Linear(inner, dim)
        # safetensors names: ff.ff.0.proj.{weight,bias}, ff.ff.2.{weight,bias}
        self.ff_0 = GLU(dim, inner, activation=act)
        self.ff_2 = nn.Linear(inner, dim, bias=True)

    def __call__(self, x):
        return self.ff_2(self.ff_0(x))


class AETransformerBlock(nn.Module):
    """The autoencoder TransformerBlock (no global_cond, no cross_attend, self-attn only).

    pre_norm -> self_attn -> + residual
    ff_norm -> ff -> + residual
    """

    def __init__(self, dim: int = 1536, dim_heads: int = 64, ff_mult: float = 3.0, sinusoidal: bool = False):
        super().__init__()
        self.pre_norm = DynamicTanh(dim)
        self.self_attn = DifferentialAttention(dim, dim_heads=dim_heads, qk_norm="dyt")
        self.ff_norm = DynamicTanh(dim)
        self.ff = FeedForward(dim, mult=ff_mult, sinusoidal=sinusoidal)
        # rope is per-block in the torch impl, with dim = dim_heads // 2
        self.rope = RotaryEmbedding(dim_heads // 2)

    def __call__(self, x, sw_mask=None):
        # cache rotary for this seq length
        freqs = self.rope.forward_from_seq_len(x.shape[-2])
        x = x + self.self_attn(self.pre_norm(x), rotary_freqs=freqs, sw_mask=sw_mask)
        x = x + self.ff(self.ff_norm(x))
        return x
