"""MLX port of the SA3 medium SAME-L (TAAE_v2) autoencoder decoder path.

Layout mirrors `stable_audio_3.models.autoencoders.SAMEDecoder` +
`TransformerResamplingBlock` (decoder mode) + `bottleneck.SoftNormBottleneck.decode`
+ `pretransforms.PatchedPretransform.decode`.

Only the decode direction is implemented (encoder isn't needed for text-to-audio).
SWA goes through `mx.fast.scaled_dot_product_attention` with an additive band mask
(see nn_blocks.sdpa_band_mask) — this is the fix that replaces the broken torch
SDPA fallback for the [17, 17] window used here.
"""
from __future__ import annotations

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .nn_blocks import AETransformerBlock, sdpa_band_mask


# ---------- bottleneck ----------

class SoftnormBottleneck(nn.Module):
    """Inference-only bottleneck.

    encode = (x * scaling_factor + bias) / running_std
    decode = x * running_std    (does NOT undo the scaling_factor/bias — the AE
                                  decoder's first projection is trained against scaled+biased input)
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.running_std = mx.ones((1,))
        self.scaling_factor = mx.ones((1, dim, 1))
        self.bias = mx.zeros((1, dim, 1))
        self.noise_scaling_factor = mx.zeros((1, 0, 1))  # noise_augment_dim=0

    def encode(self, x):
        x = x * self.scaling_factor + self.bias
        return x / self.running_std

    def decode(self, x):
        return x * self.running_std

    def __call__(self, x):  # default = decode for legacy callers
        return self.decode(x)


# ---------- patched pretransform ----------

class PatchedPretransform(nn.Module):
    """encode: "b c (l h) -> b (c h) l" with h=patch_size  (pad to multiple of h with zeros)
    decode: "b (c h) l -> b c (l h)"
    No postfilter / oversampling in sa3-medium config.
    """

    def __init__(self, channels: int = 2, patch_size: int = 256):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

    def encode(self, x):
        B, c, T = x.shape
        h = self.patch_size
        assert c == self.channels
        pad = (-T) % h
        if pad:
            x = mx.concatenate([x, mx.zeros((B, c, pad), dtype=x.dtype)], axis=-1)
        L = (T + pad) // h
        # "b c (l h) -> b (c h) l": split T -> (L, h), then move h next to c
        x = x.reshape(B, c, L, h)          # b c l h
        x = x.transpose(0, 1, 3, 2)        # b c h l
        x = x.reshape(B, c * h, L)         # b (c h) l
        return x

    def decode(self, x):
        B, CH, L = x.shape
        c, h = self.channels, self.patch_size
        assert CH == c * h
        # "b (c h) l -> b c (l h)"
        x = x.reshape(B, c, h, L)          # b c h l
        x = x.transpose(0, 1, 3, 2)        # b c l h
        x = x.reshape(B, c, L * h)         # b c (l h)
        return x


# ---------- TRB decoder ----------

class TransformerResamplingBlockDecoder(nn.Module):
    """Decoder-mode TRB: TAAE_v2 with stride=16, variable_stride=True, sliding_window=[1,1].

    Forward (matches torch SAMEEncoder/Decoder path):
      x : (B, C=1536, T_lat)   -- after the upstream Linear+Transpose
      1. rearrange to (B, T_lat, 1536)
      2. input_seg_size=1 (decoder), output_seg_size=16, sub_chunk_size=17
      3. zero-pad to multiple of input_seg_size (no-op since 1)
      4. rearrange '(b n) c d', c=1 -> (B*T_lat, 1, 1536)
      5. cat with new_tokens [B*T_lat, 16, 1536] -> (B*T_lat, 17, 1536)
      6. rearrange '(b n) c d -> b (n c) d', b=B -> (B, T_lat*17, 1536)
      7. for each transformer layer: apply with band-mask SWA (window [17, 17])
      8. rearrange back -> (B*T_lat, 17, 1536), slice last 16 -> (B*T_lat, 16, 1536)
      9. rearrange '(b n) c d -> b d (n c)', b=B -> (B, 1536, T_lat*16)
      10. mapping (1x1 conv 1536->512) -> (B, 512, T_lat*16)
    """

    def __init__(
        self,
        in_channels: int = 1536,
        out_channels: int = 512,
        stride: int = 16,
        transformer_depth: int = 12,
        sliding_window: List[int] = (1, 1),
        sinusoidal_blocks: int = 8,
        dim_heads: int = 64,
        ff_mult: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.input_seg_size = 1            # decoder
        self.output_seg_size = stride       # decoder
        self.sub_chunk_size = stride + 1   # = 17 (with no prepend_cond)

        # sliding_window in latent units; effective on packed seq = window * (stride+1)
        self.sliding_window_latents = tuple(sliding_window) if sliding_window else None
        if self.sliding_window_latents is not None:
            wl, wr = self.sliding_window_latents
            self.sliding_window_seq = (wl * self.sub_chunk_size, wr * self.sub_chunk_size)
        else:
            self.sliding_window_seq = None

        # new_tokens: variable_stride=True → shape [1, 1, dim] broadcast at runtime
        self.new_tokens = mx.zeros((1, 1, in_channels))

        # 12 transformer blocks; sinusoidal=True for the last `sinusoidal_blocks` (=8)
        self.transformers: List[AETransformerBlock] = []
        for i in range(transformer_depth):
            # match torch: `sinusoidal = (transformer_depth - i) < sinusoidal_blocks`  (strict <)
            sinusoidal = (transformer_depth - i) < sinusoidal_blocks
            self.transformers.append(
                AETransformerBlock(dim=in_channels, dim_heads=dim_heads, ff_mult=ff_mult, sinusoidal=sinusoidal)
            )

        # mapping = WNConv1d(in, out, k=1, bias=True)  -- absorb weight_g/weight_v into a Linear
        # (1x1 conv == Linear on channel dim). Implemented here as Linear; we store weights with
        # canonical names so the loader fills them after folding weight_g/weight_v.
        self.mapping = nn.Linear(in_channels, out_channels, bias=True)

    def __call__(self, x):
        B, C, T_lat = x.shape
        assert C == self.in_channels, (C, self.in_channels)

        # (1) (B, C, T_lat) -> (B, T_lat, C)
        x = x.transpose(0, 2, 1)

        # (2-6) inflate to (B, T_lat * sub_chunk_size, C)
        # input_seg_size=1: each latent frame becomes a "chunk" with [1 input frame + 16 new tokens]
        # = 17 tokens. Concretely: x is (B, T_lat, 1, C) -> cat new_tokens (B, T_lat, 16, C) ->
        # (B, T_lat, 17, C) -> (B, T_lat*17, C)
        x = x.reshape(B, T_lat, self.input_seg_size, C)  # (B, T_lat, 1, C)

        nt = mx.broadcast_to(self.new_tokens, (B, T_lat, self.output_seg_size, C))  # (B, T_lat, 16, C)
        # mask_noise applies only during training (0.1 for decoder); inference -> no-op
        x = mx.concatenate([x, nt], axis=2)  # (B, T_lat, 17, C)
        x = x.reshape(B, T_lat * self.sub_chunk_size, C)  # (B, T_lat*17, C)

        # (7) transformer stack with SWA band mask
        T = x.shape[1]
        if self.sliding_window_seq is not None:
            wl, wr = self.sliding_window_seq
            sw_mask = sdpa_band_mask(T, T, wl, wr, dtype=mx.float32)
        else:
            sw_mask = None

        for layer in self.transformers:
            x = layer(x, sw_mask=sw_mask)

        # (8) reshape to (B*T_lat, 17, C), slice last 16, reshape to (B, C, T_lat*16)
        x = x.reshape(B, T_lat, self.sub_chunk_size, C)  # (B, T_lat, 17, C)
        x = x[:, :, -self.output_seg_size:, :]            # (B, T_lat, 16, C)
        x = x.reshape(B, T_lat * self.output_seg_size, C)  # (B, T_lat*16, C)
        # to (B, C, T_lat*16) for mapping (Linear acts on last dim) -> (B, T_lat*16, out_C)
        x = self.mapping(x)                                  # (B, T_lat*16, out_C)
        x = x.transpose(0, 2, 1)                              # (B, out_C, T_lat*16)
        return x


# ---------- SAMEDecoder ----------

class SAMEDecoder(nn.Module):
    """layers = [Transpose, Linear(latent_dim, channels[-1]), Transpose, TRB decoder]."""

    def __init__(self, latent_dim: int = 256, ae_dim: int = 1536, out_channels: int = 512):
        super().__init__()
        # layers[0] = Transpose -> handled inline below
        self.layers_1 = nn.Linear(latent_dim, ae_dim, bias=True)   # safetensors: decoder.layers.1.{weight,bias}
        self.layers_3 = TransformerResamplingBlockDecoder(in_channels=ae_dim, out_channels=out_channels)

    def __call__(self, x):
        # x: (B, latent_dim, T_lat)
        x = x.transpose(0, 2, 1)        # (B, T_lat, latent_dim)
        x = self.layers_1(x)             # (B, T_lat, ae_dim)
        x = x.transpose(0, 2, 1)        # (B, ae_dim, T_lat)
        x = self.layers_3(x)             # (B, 512, T_lat*16)
        return x


# ---------- TRB encoder ----------

class TransformerResamplingBlockEncoder(nn.Module):
    """Encoder-mode TRB. Mirror of decoder; key differences:
      input_seg_size = stride (16), output_seg_size = 1
      mapping is applied BEFORE the transformer stack (with pre-pad)
      mapping: WNConv1d(in=512, out=1536, k=1)
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 1536,
        stride: int = 16,
        transformer_depth: int = 12,
        sliding_window=(1, 1),
        dim_heads: int = 64,
        ff_mult: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.input_seg_size = stride      # encoder
        self.output_seg_size = 1           # encoder
        self.sub_chunk_size = stride + 1

        self.sliding_window_latents = tuple(sliding_window) if sliding_window else None
        if self.sliding_window_latents is not None:
            wl, wr = self.sliding_window_latents
            self.sliding_window_seq = (wl * self.sub_chunk_size, wr * self.sub_chunk_size)
        else:
            self.sliding_window_seq = None

        self.mapping = nn.Linear(in_channels, out_channels, bias=True)  # WNConv1d k=1 -> Linear after fold
        self.new_tokens = mx.zeros((1, 1, out_channels))

        self.transformers: List[AETransformerBlock] = []
        for _ in range(transformer_depth):
            self.transformers.append(
                AETransformerBlock(dim=out_channels, dim_heads=dim_heads, ff_mult=ff_mult, sinusoidal=False)
            )

    def __call__(self, x):
        B, C, T = x.shape
        assert C == self.in_channels

        # pad to multiple of input_seg_size
        pad = (-T) % self.input_seg_size
        if pad:
            x = mx.concatenate([x, mx.zeros((B, C, pad), dtype=x.dtype)], axis=-1)
            T = x.shape[-1]

        # mapping (Linear acts on last dim): transpose -> apply -> back
        x = x.transpose(0, 2, 1)      # (B, T, C)
        x = self.mapping(x)            # (B, T, out_C)
        # back to (B, out_C, T)? No — we want the transformer to see (B, T, out_C). So leave as (B, T, out_C).
        # Then rearrange into chunks
        D = self.out_channels
        n_chunks = T // self.input_seg_size  # T / 16
        x = x.reshape(B, n_chunks, self.input_seg_size, D)  # (B, n_chunks, 16, D)
        nt = mx.broadcast_to(self.new_tokens, (B, n_chunks, self.output_seg_size, D))  # (B, n_chunks, 1, D)
        x = mx.concatenate([x, nt], axis=2)  # (B, n_chunks, 17, D)
        x = x.reshape(B, n_chunks * self.sub_chunk_size, D)

        # SWA
        Tp = x.shape[1]
        if self.sliding_window_seq is not None:
            wl, wr = self.sliding_window_seq
            sw_mask = sdpa_band_mask(Tp, Tp, wl, wr, dtype=mx.float32)
        else:
            sw_mask = None

        for layer in self.transformers:
            x = layer(x, sw_mask=sw_mask)

        # reshape & take last output_seg_size=1 from each chunk
        x = x.reshape(B, n_chunks, self.sub_chunk_size, D)
        x = x[:, :, -self.output_seg_size:, :]   # (B, n_chunks, 1, D)
        x = x.reshape(B, n_chunks * self.output_seg_size, D)  # (B, n_chunks, D)
        x = x.transpose(0, 2, 1)                                # (B, D, n_chunks) = (B, 1536, T/16)
        return x


class SAMEEncoder(nn.Module):
    """layers = [TRB encoder, Transpose, Linear(channels[-1], latent_dim), Transpose]."""

    def __init__(self, in_channels: int = 512, ae_dim: int = 1536, latent_dim: int = 256):
        super().__init__()
        self.layers_0 = TransformerResamplingBlockEncoder(in_channels=in_channels, out_channels=ae_dim)
        self.layers_2 = nn.Linear(ae_dim, latent_dim, bias=True)

    def __call__(self, x):
        # x: (B, in_channels=512, T_pre)
        x = self.layers_0(x)         # (B, 1536, T_lat)
        x = x.transpose(0, 2, 1)    # (B, T_lat, 1536)
        x = self.layers_2(x)         # (B, T_lat, 256)
        x = x.transpose(0, 2, 1)    # (B, 256, T_lat)
        return x


# ---------- top-level decoder ----------

class SA3MediumAE(nn.Module):
    """Full SA3-medium AE: bottleneck + SAMEEncoder/Decoder + PatchedPretransform.

    Decode: latents (B, 256, T_lat)  ->  waveform (B, 2, T_lat * 4096)
    Encode: waveform (B, 2, T_audio)  ->  latents (B, 256, T_lat)
    """

    def __init__(self):
        super().__init__()
        self.bottleneck = SoftnormBottleneck(dim=256)
        self.encoder = SAMEEncoder(in_channels=512, ae_dim=1536, latent_dim=256)
        self.decoder = SAMEDecoder(latent_dim=256, ae_dim=1536, out_channels=512)
        self.pretransform = PatchedPretransform(channels=2, patch_size=256)

    def decode(self, latents):
        x = self.bottleneck.decode(latents)
        x = self.decoder(x)
        x = self.pretransform.decode(x)
        return x

    def encode(self, audio):
        x = self.pretransform.encode(audio)
        x = self.encoder(x)
        x = self.bottleneck.encode(x)
        return x


def decode_chunked(ae: SA3MediumAE, latents, chunk_size: int = 128, overlap: int = 32):
    """Match `AudioAutoencoder.decode_audio(chunked=True)` semantics, in MLX.

    Each chunk decodes `chunk_size` latents -> `chunk_size * 4096` samples.
    On internal edges we trim `(overlap // 2) * samples_per_latent` samples so
    consecutive chunks butt up cleanly. Final chunk is anchored to the end.

    Memory: each chunk needs the AE state for a packed seq of `chunk_size * 17`
    tokens (e.g. 128 -> 2176 tokens, band mask ~18MB). Total RAM ~ peak chunk
    activations + the assembled output. For 3-min audio: T_lat ~= 1937,
    20 chunks of ~524k samples each, decoded sequentially, final buffer ~64MB.

    No in-place writes — assembled by concatenation in order.
    """
    samples_per_latent = 4096
    T_lat = latents.shape[-1]
    if T_lat <= chunk_size:
        return ae.decode(latents)

    hop = chunk_size - overlap
    starts = list(range(0, T_lat - chunk_size + 1, hop))
    if starts[-1] != T_lat - chunk_size:
        starts.append(T_lat - chunk_size)

    chunk_samples = chunk_size * samples_per_latent
    half_overlap_samples = (overlap // 2) * samples_per_latent
    n = len(starts)

    pieces = []
    for i, s_lat in enumerate(starts):
        chunk = ae.decode(latents[..., s_lat:s_lat + chunk_size])
        mx.eval(chunk)   # release transient activations between chunks
        is_first = i == 0
        is_last = i == n - 1
        left = 0 if is_first else half_overlap_samples
        right = chunk_samples if is_last else chunk_samples - half_overlap_samples
        pieces.append(chunk[..., left:right])
    return mx.concatenate(pieces, axis=-1)
