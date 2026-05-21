"""Full AE-level parity check: torch sa3 AE decode vs MLX AE decode, same weights, same latents.

This catches: weight remapping bugs (WNConv1d fold, name mismatches), sequence-layout bugs
(packed-seq reshape, new_tokens injection), bottleneck mismatch, patched pretransform mismatch.

If output diff is ~1e-5: the MLX AE is faithful. Discrepancies vs the "AE makes static" claim
likely live at larger T (chunked) or MPS-only paths.
"""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import json
import numpy as np
import torch
import mlx.core as mx

from stable_audio_3.loading_utils import load_autoencoder

from mlx_sa3.ae import SA3MediumAE
from mlx_sa3.weights import load_ae_weights


CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
CFG = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model_config.json"


def main():
    # MLX
    print("loading MLX AE...")
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)

    # torch (cpu)
    print("loading torch AE on CPU...")
    torch_ae = load_autoencoder(CFG, CKPT, device="cpu")
    torch_ae.eval()

    # Latents: 16 frames (~1.5s) of zero-mean unit-variance noise
    T_lat = 16
    rng = np.random.default_rng(0)
    lat_np = rng.standard_normal((1, 256, T_lat)).astype(np.float32)

    # torch decode (note: this goes through the SDPA SWA fallback since no flash-attn)
    with torch.no_grad():
        y_t = torch_ae.decode(torch.from_numpy(lat_np))
    out_t = y_t.numpy()

    # mlx decode
    lat_m = mx.array(lat_np)
    y_m = mlx_ae.decode(lat_m)
    mx.eval(y_m)
    out_m = np.array(y_m)

    print(f"out_t shape: {out_t.shape}, out_m shape: {out_m.shape}")
    diff = np.abs(out_t - out_m)
    print(f"max abs diff: {diff.max():.4e}")
    print(f"mean abs diff: {diff.mean():.4e}")
    print(f"torch out range: [{out_t.min():.4f}, {out_t.max():.4f}]  norm: {np.linalg.norm(out_t):.4f}")
    print(f"mlx   out range: [{out_m.min():.4f}, {out_m.max():.4f}]  norm: {np.linalg.norm(out_m):.4f}")
    cs = (out_t * out_m).sum() / max(1e-12, (np.linalg.norm(out_t) * np.linalg.norm(out_m)))
    print(f"cosine sim: {cs:.6f}")


if __name__ == "__main__":
    main()
