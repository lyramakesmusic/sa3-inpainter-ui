"""E2E text-to-audio using LOCAL paths (no hf_hub_download hangs on xet).

Patches sa3 to read t5gemma from a local subfolder and loads the DIT weights directly.
"""
import sys, os, time, json
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")
os.environ["HF_HUB_OFFLINE"] = "1"        # block remote hub calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torchaudio
import mlx.core as mx

LOCAL_MEDIUM = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium"
CFG = f"{LOCAL_MEDIUM}/model_config.json"
CKPT = f"{LOCAL_MEDIUM}/model.safetensors"


def patch_t5gemma_to_local(model_config):
    """Override the t5gemma conditioner's repo_id to a local path so transformers
    loads from disk instead of hitting HF."""
    for c in model_config["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            c["config"]["repo_id"] = LOCAL_MEDIUM   # transformers will use <repo_id>/<subfolder>
    return model_config


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="House music, 124 BPM, energetic festival vibes")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/sa3_mlx_e2e.wav")
    p.add_argument("--ae-only", action="store_true", help="Skip DIT; use random latents (sanity check)")
    args = p.parse_args()

    if args.ae_only:
        # Quick path: decode random latents through both torch and MLX AE, save both
        T_lat = int(args.duration * 44100 / 4096) + 1
        rng = np.random.default_rng(args.seed)
        lat_np = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
        # MLX AE
        from mlx_sa3.ae import SA3MediumAE
        from mlx_sa3.weights import load_ae_weights
        mlx_ae = SA3MediumAE()
        load_ae_weights(mlx_ae, CKPT)
        wav_m = mlx_ae.decode(mx.array(lat_np))
        mx.eval(wav_m)
        wav_np = np.array(wav_m)[0]
        sr = 44100
        n_samples = int(args.duration * sr)
        wav_np = wav_np[:, :n_samples]
        torchaudio.save(args.out, torch.from_numpy(wav_np.clip(-1, 1)), sr)
        print(f"saved {args.out} ({n_samples/sr:.2f}s @ {sr}Hz)")
        print(f"stats: min={wav_np.min():.3f} max={wav_np.max():.3f} std={wav_np.std():.3f}")
        return

    # Full path: torch DIT on MPS + MLX AE
    model_config = json.load(open(CFG))
    model_config = patch_t5gemma_to_local(model_config)
    print("Building model from local config...")
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from safetensors.torch import load_file
    model = create_diffusion_cond_from_config(model_config)
    print("Loading weights from local safetensors...")
    sd = load_file(CKPT)
    model.load_state_dict(sd, strict=False)
    model.eval().requires_grad_(False)
    model.to("mps")
    print("Model on MPS. Building StableAudioModel wrapper...")

    # The wrapper class needs the model_config to know shapes etc
    from stable_audio_3 import StableAudioModel
    sa = StableAudioModel(model, model_config, device="mps", model_half=False)

    print(f"Generating {args.duration}s in {args.steps} steps...")
    t0 = time.time()
    latents = sa.generate(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        return_latents=True,
    )
    print(f"DIT done in {time.time()-t0:.1f}s. Latents shape: {tuple(latents.shape)}")

    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights
    print("Loading MLX AE...")
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
    print("MLX AE decode (band-mask SWA, chunked)...")
    lat_np = latents.detach().to(torch.float32).cpu().numpy()
    t1 = time.time()
    if lat_np.shape[-1] > 128:
        wav_m = decode_chunked(mlx_ae, mx.array(lat_np), chunk_size=128, overlap=32)
    else:
        wav_m = mlx_ae.decode(mx.array(lat_np))
    mx.eval(wav_m)
    print(f"AE decode in {time.time()-t1:.1f}s.")
    wav_np = np.array(wav_m)[0]
    sr = 44100
    n_samples = int(args.duration * sr)
    wav_np = wav_np[:, :n_samples]
    torchaudio.save(args.out, torch.from_numpy(wav_np.clip(-1, 1)), sr)
    print(f"saved {args.out} ({n_samples/sr:.2f}s @ {sr}Hz)")
    print(f"stats: min={wav_np.min():.3f} max={wav_np.max():.3f} std={wav_np.std():.3f}")


if __name__ == "__main__":
    main()
