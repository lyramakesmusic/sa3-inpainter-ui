"""Compare torch vs MLX AE decode with torch's stochastic noise sources disabled.

If diff drops to ~1e-5, MLX impl is exact-equivalent to torch.
"""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import numpy as np
import torch
import mlx.core as mx

from stable_audio_3.loading_utils import load_autoencoder
from mlx_sa3.ae import SA3MediumAE
from mlx_sa3.weights import load_ae_weights

CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
CFG  = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model_config.json"


def main():
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
    t_ae = load_autoencoder(CFG, CKPT, device="cpu")
    t_ae.eval()
    # Disable torch's eval-mode stochastic noise sources for a deterministic comparison
    t_ae.bottleneck.noise_regularize = False
    t_ae.decoder.layers[3].mask_noise = 0.0

    T_lat = 16
    rng = np.random.default_rng(0)
    lat_np = rng.standard_normal((1, 256, T_lat)).astype(np.float32)

    with torch.no_grad():
        y_t = t_ae.decode(torch.from_numpy(lat_np))
    out_t = y_t.numpy()

    y_m = mlx_ae.decode(mx.array(lat_np))
    mx.eval(y_m)
    out_m = np.array(y_m)

    diff = np.abs(out_t - out_m)
    print(f"shape: {out_t.shape}")
    print(f"max abs diff: {diff.max():.4e}")
    print(f"mean abs diff: {diff.mean():.4e}")
    print(f"max rel diff (where torch>1e-3): {(diff[np.abs(out_t)>1e-3]/np.abs(out_t)[np.abs(out_t)>1e-3]).max():.4e}")
    print(f"cosine sim: {(out_t*out_m).sum()/(np.linalg.norm(out_t)*np.linalg.norm(out_m)):.8f}")


if __name__ == "__main__":
    main()
