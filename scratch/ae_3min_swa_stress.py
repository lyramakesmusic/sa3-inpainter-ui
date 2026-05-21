"""SWA stress test: decode 3min of random latents through the MLX AE.

If the band-mask SWA produces correct outputs at the packed seq ~33k, the audio
will sound like coherent texture (modulated noise) and NOT static. If SWA is
broken at scale, every chunk will produce uncorrelated garbage and we'll hear it.

Also runs torch AE for comparison (will use the SDPA SWA fallback torch path,
which is what Lyra reports as broken on Windows). At this scale we can
auditorily compare both.
"""
import sys, time
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import numpy as np
import soundfile as sf
import mlx.core as mx

CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
CFG  = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model_config.json"
SR = 44100


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlx-out", default="/tmp/sa3_mlx_3min.wav")
    p.add_argument("--torch-out", default="/tmp/sa3_torch_3min.wav")
    p.add_argument("--skip-torch", action="store_true")
    p.add_argument("--chunk", type=int, default=128, help="AE decode chunk size in latents")
    args = p.parse_args()

    T_lat = int(args.duration * SR / 4096) + 1
    rng = np.random.default_rng(args.seed)
    # Mimic the magnitude/structure of real latents (std ~ 0.3 from real DIT outputs)
    lat_np = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    print(f"latents shape: (1, 256, {T_lat})  -> ~{T_lat * 4096 / SR:.1f}s")
    print(f"AE packed seq: {T_lat * 17} tokens, window=[17,17] -> 34 visible per query")

    # MLX path
    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights
    print("\n== MLX decode (chunked, band-mask SWA) ==")
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
    t0 = time.time()
    wav_m = decode_chunked(mlx_ae, mx.array(lat_np), chunk_size=args.chunk, overlap=32)
    mx.eval(wav_m)
    print(f"MLX decode: {time.time()-t0:.1f}s")
    arr_m = np.array(wav_m)[0].T  # (samples, channels) for soundfile
    n_samples = int(args.duration * SR)
    arr_m = arr_m[:n_samples]
    peak = np.abs(arr_m).max()
    if peak > 0:
        arr_m = arr_m * (0.95 / peak)
    sf.write(args.mlx_out, arr_m, SR)
    print(f"saved {args.mlx_out}  ({arr_m.shape[0]/SR:.1f}s, std={arr_m.std():.3f}, peak_before_norm={peak:.3f})")

    if args.skip_torch:
        return

    print("\n== torch decode (CPU, SDPA SWA fallback) ==")
    import torch
    from stable_audio_3.loading_utils import load_autoencoder
    t_ae = load_autoencoder(CFG, CKPT, device="cpu")
    t_ae.eval()
    t_ae.bottleneck.noise_regularize = False
    t_ae.decoder.layers[3].mask_noise = 0.0
    t0 = time.time()
    with torch.no_grad():
        wav_t = t_ae.decode(torch.from_numpy(lat_np))  # this uses chunked path internally? Let me check.
    # Actually self.decode doesn't chunk by default; decode_audio does. Use decode_audio.
    print(f"torch decode (non-chunked): {time.time()-t0:.1f}s")
    arr_t = wav_t.numpy()[0].T
    arr_t = arr_t[:n_samples]
    peak = np.abs(arr_t).max()
    if peak > 0:
        arr_t = arr_t * (0.95 / peak)
    sf.write(args.torch_out, arr_t, SR)
    print(f"saved {args.torch_out}  (std={arr_t.std():.3f}, peak_before_norm={peak:.3f})")

    diff = np.abs(np.array(wav_m)[0] - wav_t.numpy()[0])
    print(f"\nMLX vs torch: maxabs={diff.max():.4e}  meanabs={diff.mean():.4e}")
    cs = (np.array(wav_m)[0] * wav_t.numpy()[0]).sum() / (
        np.linalg.norm(np.array(wav_m)[0]) * np.linalg.norm(wav_t.numpy()[0]) + 1e-12)
    print(f"cosine sim: {cs:.6f}")


if __name__ == "__main__":
    main()
