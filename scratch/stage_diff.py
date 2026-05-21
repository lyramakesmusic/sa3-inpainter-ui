"""Stage-by-stage diff between torch and MLX AE decode pipeline."""
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


def diff(name, a, b):
    d = np.abs(a - b)
    print(f"{name:40s}  shape={a.shape} | maxabs={d.max():.4e} | rel={(d/(np.abs(a)+1e-6)).max():.4e} | torch_norm={np.linalg.norm(a):.3f} | mlx_norm={np.linalg.norm(b):.3f}")


def main():
    torch.manual_seed(0)
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
    t_ae = load_autoencoder(CFG, CKPT, device="cpu")
    t_ae.eval()

    T_lat = 16
    rng = np.random.default_rng(0)
    lat_np = rng.standard_normal((1, 256, T_lat)).astype(np.float32)

    # Stage 1: bottleneck.decode
    with torch.no_grad():
        lat_t1 = t_ae.bottleneck.decode(torch.from_numpy(lat_np))
    lat_m1_arr = mlx_ae.bottleneck.decode(mx.array(lat_np))
    mx.eval(lat_m1_arr)
    diff("bottleneck.decode", lat_t1.numpy(), np.array(lat_m1_arr))

    # Stage 2: SAMEDecoder layers_1 (Linear 256->1536 with transposes)
    with torch.no_grad():
        l1_t = t_ae.decoder.layers[1](lat_t1.transpose(-1, -2))   # (1, T_lat, 1536)
    l1_m = mlx_ae.decoder.layers_1(lat_m1_arr.transpose(0, 2, 1))
    mx.eval(l1_m)
    diff("decoder.layers[1] (Linear)", l1_t.numpy(), np.array(l1_m))

    # Now feed l1_t/l1_m through TRB. Compare the output of TRB.
    # torch TRB input shape: (B, 1536, T_lat)
    x_t = l1_t.transpose(-1, -2)
    with torch.no_grad():
        trb_t = t_ae.decoder.layers[3](x_t)   # (1, 512, T_lat*16)
    x_m = l1_m.transpose(0, 2, 1)
    trb_m = mlx_ae.decoder.layers_3(x_m)
    mx.eval(trb_m)
    diff("TRB(decoder) full", trb_t.numpy(), np.array(trb_m))

    # final pretransform
    with torch.no_grad():
        wav_t = t_ae.pretransform.decode(trb_t)
    wav_m = mlx_ae.pretransform.decode(trb_m)
    mx.eval(wav_m)
    diff("pretransform.decode", wav_t.numpy(), np.array(wav_m))

    # full end-to-end
    with torch.no_grad():
        full_t = t_ae.decode(torch.from_numpy(lat_np))
    full_m = mlx_ae.decode(mx.array(lat_np))
    mx.eval(full_m)
    diff("FULL decode", full_t.numpy(), np.array(full_m))


if __name__ == "__main__":
    main()
