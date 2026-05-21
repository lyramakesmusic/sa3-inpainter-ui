"""SA3 medium gradio UI with: (a) local-file model load (no hf_hub hang), and
(b) MLX-backed AE decode so longer audio doesn't go static on Mac.

Forwards through `stable_audio_3.interface.diffusion_cond.create_diffusion_cond_ui`,
which already wires up text-to-audio, init-audio, inpaint, LoRA stack, etc.

Usage:
  uv run python run_gradio.py
  uv run python run_gradio.py --lora-ckpt-path /path/to/lora.safetensors
"""
import argparse, json, os, sys, time

sys.path.insert(0, str(os.path.dirname(__file__)))

# Suppress noisy warnings unless --verbose. We do NOT force offline mode here —
# we already bypass hf_hub for the sa3 medium load by building directly via the
# factory + safetensors load_file, so the Prompt Assistant (and any future HF
# downloads) can still pull what they need.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import mlx.core as mx

LOCAL_MEDIUM = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium"
CFG = f"{LOCAL_MEDIUM}/model_config.json"
CKPT = f"{LOCAL_MEDIUM}/model.safetensors"


def build_local_model(device: str = "mps"):
    """Build StableAudioModel without touching hf_hub (which hangs on xet for some)."""
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3 import StableAudioModel
    from safetensors.torch import load_file

    model_config = json.load(open(CFG))
    # Force t5gemma to load from the local repo
    for c in model_config["model"]["conditioning"]["configs"]:
        if c["type"] == "t5gemma":
            c["config"]["repo_id"] = LOCAL_MEDIUM
    print("[sa3-mlx] Building model from local config...")
    model = create_diffusion_cond_from_config(model_config)
    print("[sa3-mlx] Loading weights from", CKPT)
    sd = load_file(CKPT)
    model.load_state_dict(sd, strict=False)
    model.eval().requires_grad_(False)
    model.to(device)
    return StableAudioModel(model, model_config, device=device, model_half=False)


def patch_ae_to_mlx(sa_model, chunk_size: int = 128, overlap: int = 32):
    """Replace `model.pretransform.decode` with a wrapper that runs decode in MLX.

    The sa3 pipeline expects `pretransform.decode(z)` to return a (B, C, T) torch
    tensor on the same device as z. We:
      1. detach + .cpu().float() z (small overhead vs DIT step cost)
      2. convert to mlx.array
      3. call MLX AE decode (chunked for long sequences)
      4. wrap output back to a torch tensor on z's original device
    """
    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights

    print("[sa3-mlx] Loading MLX AE...")
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)

    # Sa3's AutoencoderPretransform.decode does `z * self.scale` before passing to
    # the inner model's decode_audio. We preserve that:
    pre = sa_model.model.pretransform
    inner_scale = float(getattr(pre, "scale", 1.0))
    print(f"[sa3-mlx] Patching pretransform.decode -> MLX (chunk={chunk_size}, overlap={overlap}, scale={inner_scale})")

    def mlx_decode(z, *args, **kwargs):
        device = z.device
        z_np = (z * inner_scale).detach().to(torch.float32).cpu().numpy()
        z_mx = mx.array(z_np)
        t0 = time.time()
        if z_mx.shape[-1] > chunk_size:
            wav_mx = decode_chunked(mlx_ae, z_mx, chunk_size=chunk_size, overlap=overlap)
        else:
            wav_mx = mlx_ae.decode(z_mx)
        mx.eval(wav_mx)
        wav_np = np.array(wav_mx)
        elapsed = time.time() - t0
        T_lat = z_mx.shape[-1]
        T_samp = wav_np.shape[-1]
        print(f"[sa3-mlx] AE decode: {T_lat} latents -> {T_samp} samples in {elapsed:.2f}s ({T_samp/44100/elapsed:.1f}x realtime)")
        return torch.from_numpy(wav_np).to(device)

    pre.decode = mlx_decode
    return sa_model


def main():
    parser = argparse.ArgumentParser(description="SA3 medium with MLX AE, Gradio UI")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"], help="Torch device for DIT")
    parser.add_argument("--lora-ckpt-path", type=str, nargs="*", default=None, help="LoRA checkpoint(s) to apply")
    parser.add_argument("--default-prompt", type=str, default="House music, 124 BPM, energetic festival vibes")
    parser.add_argument("--title", type=str, default="SA3 medium · MLX AE")
    parser.add_argument("--chunk-size", type=int, default=128, help="AE chunk size in latents")
    parser.add_argument("--overlap", type=int, default=32, help="AE chunk overlap in latents")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
    from stable_audio_3.verbose import set_verbose
    set_verbose(args.verbose)

    torch.manual_seed(42)
    model = build_local_model(device=args.device)
    if args.lora_ckpt_path:
        print("[sa3-mlx] Loading LoRA(s):", args.lora_ckpt_path)
        model.load_lora(args.lora_ckpt_path)
    patch_ae_to_mlx(model, chunk_size=args.chunk_size, overlap=args.overlap)

    interface = create_diffusion_cond_ui(
        model,
        gradio_title=args.title,
        default_prompt=args.default_prompt,
    )
    interface.queue()
    interface.launch(
        share=args.share,
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
    )


if __name__ == "__main__":
    main()
