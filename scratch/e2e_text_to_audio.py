"""End-to-end text-to-audio on Mac: torch+MPS DIT → MLX AE → WAV.

Loads sa3-medium with `StableAudioModel.from_pretrained("medium", device="mps")`,
runs diffusion to get latents (no flash-attn; SDPA works fine because DIT is full-attn),
then hands the latents to the MLX AE decoder for the bit-equivalent decode.
"""
import os, sys, time
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")
os.environ["HF_HUB_OFFLINE"] = "0"

import numpy as np
import torch
import torchaudio
import mlx.core as mx

from mlx_sa3.ae import SA3MediumAE
from mlx_sa3.weights import load_ae_weights


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="House music, 124 BPM, energetic festival vibes")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/sa3_mlx_e2e.wav")
    args = p.parse_args()

    # 1) Hook up the sa3 medium model on MPS, get latents
    from stable_audio_3 import StableAudioModel

    print("Loading sa3 medium on MPS (no flash-attn; SDPA fallback for DIT full-attn)...")
    sa = StableAudioModel.from_pretrained("medium", device="mps")
    print("Model loaded. Generating latents...")
    t0 = time.time()
    latents = sa.generate(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg,
        seed=args.seed,
        return_latents=True,
    )
    print(f"DIT done in {time.time()-t0:.1f}s. Latents shape: {tuple(latents.shape)}, dtype: {latents.dtype}")

    # 2) MLX AE decode
    print("Loading MLX AE...")
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae)
    print("Decoding via MLX AE (with band-mask SWA)...")
    lat_np = latents.detach().to(torch.float32).cpu().numpy()
    t1 = time.time()
    wav_m = mlx_ae.decode(mx.array(lat_np))
    mx.eval(wav_m)
    print(f"MLX decode in {time.time()-t1:.1f}s. Wav shape: {tuple(wav_m.shape)}")

    # 3) Save WAV
    wav_np = np.array(wav_m)[0]   # (2, T)
    sr = 44100
    # Truncate to requested duration
    n_samples = int(args.duration * sr)
    wav_np = wav_np[:, :n_samples]
    wav_t = torch.from_numpy(wav_np.clip(-1, 1))
    torchaudio.save(args.out, wav_t, sr)
    print(f"Saved {args.out}  ({wav_np.shape[-1] / sr:.2f}s @ {sr}Hz)")
    print(f"audio stats: min={wav_np.min():.3f} max={wav_np.max():.3f} std={wav_np.std():.3f}")


if __name__ == "__main__":
    main()
