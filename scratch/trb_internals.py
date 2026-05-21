"""Trace inside the TRB(decoder) layer-by-layer to localize the divergence."""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import numpy as np
import torch
import mlx.core as mx
from einops import rearrange

from stable_audio_3.loading_utils import load_autoencoder
from mlx_sa3.ae import SA3MediumAE
from mlx_sa3.nn_blocks import sdpa_band_mask
from mlx_sa3.weights import load_ae_weights

CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
CFG  = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model_config.json"


def diff(name, a, b):
    if isinstance(b, mx.array):
        b = np.array(b)
    if hasattr(a, 'detach'):
        a = a.detach().numpy() if not isinstance(a, np.ndarray) else a
    d = np.abs(a - b)
    print(f"{name:60s}  shape={a.shape}  maxabs={d.max():.4e}  torch_norm={np.linalg.norm(a):.3f}  mlx_norm={np.linalg.norm(b):.3f}")


def main():
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
    t_ae = load_autoencoder(CFG, CKPT, device="cpu")
    t_ae.eval()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, C, T_lat = 1, 1536, 16
    # Use deterministic input at TRB-decoder input shape
    x_np = rng.standard_normal((B, C, T_lat)).astype(np.float32) * 0.1

    # ---- Step 1: prepare packed sequence
    # torch path: TRB.forward when stride=None and type=decoder + sliding_window
    trb_t = t_ae.decoder.layers[3]
    trb_m = mlx_ae.decoder.layers_3

    # mask the noise inside TRB: temporarily zero mask_noise on torch so we can compare
    trb_t.mask_noise = 0.0

    # Walk torch internals manually mirroring the forward:
    sub_chunk = 17
    input_seg = 1
    output_seg = 16
    sliding_window = [17, 17]

    x_t = torch.from_numpy(x_np)            # (B, 1536, T_lat)
    x_t_p = rearrange(x_t, '... a b -> ... b a')   # (B, T_lat, 1536)
    x_t_p = rearrange(x_t_p, 'b (n c) d -> (b n) c d', c=input_seg)  # (B*T_lat, 1, 1536)
    new_tokens = trb_t.new_tokens.expand([x_t_p.shape[0], output_seg, -1])  # (B*T_lat, 16, 1536)
    x_t_p = torch.cat([x_t_p, new_tokens], dim=-2)   # (B*T_lat, 17, 1536)
    x_t_p = rearrange(x_t_p, '(b n) c d -> b (n c) d', b=B)  # (B, T_lat*17, 1536)
    print(f"torch packed seq shape: {x_t_p.shape}")

    # MLX equivalent
    x_m = mx.array(x_np)
    x_m_p = x_m.transpose(0, 2, 1)                          # (B, T_lat, 1536)
    x_m_p = x_m_p.reshape(B, T_lat, input_seg, C)          # (B, T_lat, 1, 1536)
    nt = mx.broadcast_to(trb_m.new_tokens, (B, T_lat, output_seg, C))
    x_m_p = mx.concatenate([x_m_p, nt], axis=2)             # (B, T_lat, 17, 1536)
    x_m_p = x_m_p.reshape(B, T_lat * sub_chunk, C)
    mx.eval(x_m_p)
    diff("packed seq (pre-transformers)", x_t_p.detach().numpy(), x_m_p)

    # Run each transformer layer in torch
    x_t_run = x_t_p.clone()
    x_m_run = x_m_p
    Tp = x_t_run.shape[1]
    sw_mask = sdpa_band_mask(Tp, Tp, 17, 17, dtype=mx.float32)
    for i, layer in enumerate(trb_t.transformers):
        with torch.no_grad():
            x_t_run = layer(x_t_run, self_attention_flash_sliding_window=sliding_window)
        x_m_run = trb_m.transformers[i](x_m_run, sw_mask=sw_mask)
        if i == 0 or i == 5 or i == 11:
            mx.eval(x_m_run)
            diff(f"after transformer[{i}]", x_t_run.detach().numpy(), x_m_run)

    # After transformers
    x_t_post = rearrange(x_t_run, 'b (n c) d -> (b n) c d', c=sub_chunk)
    x_t_post = x_t_post[:, -output_seg:, :]
    x_t_post = rearrange(x_t_post, '(b n) c d -> b d (n c)', b=B)   # (B, 1536, T_lat*16)
    print(f"torch post-transformer reshape: {x_t_post.shape}")
    with torch.no_grad():
        x_t_map = trb_t.mapping(x_t_post)   # (B, 512, T_lat*16)
    print(f"torch after mapping shape: {x_t_map.shape}")

    # MLX post
    x_m_post = x_m_run.reshape(B, T_lat, sub_chunk, C)
    x_m_post = x_m_post[:, :, -output_seg:, :]
    x_m_post = x_m_post.reshape(B, T_lat * output_seg, C)
    x_m_map = trb_m.mapping(x_m_post)
    x_m_map = x_m_map.transpose(0, 2, 1)
    mx.eval(x_m_map)
    diff("after mapping", x_t_map.detach().numpy(), x_m_map)


if __name__ == "__main__":
    main()
