"""Pre-encode audio files to latents using MLX SAME encoder.

Produces the same .npy + .json format as the PyTorch pre_encode.py,
so MLX-encoded latents work with both training backends.

Usage:
    python mlx_sa3/pre_encode_mlx.py \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output/_encoded \
        --model-ckpt /path/to/model.safetensors
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _add_sa3_to_path():
    sa3_root = Path(os.environ.get("SA3_ROOT", ""))
    if not sa3_root.is_dir():
        raise RuntimeError("SA3_ROOT not set or invalid")
    mlx_root = sa3_root / "optimized" / "mlx"
    for p in [str(sa3_root), str(mlx_root), str(mlx_root / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return sa3_root, mlx_root


def encode_audio_files(
    audio_dir: str,
    output_dir: str,
    encoder_weights: str | None = None,
    sample_rate: int = 44100,
    max_seconds: float = 47.0,
):
    sa3_root, mlx_root = _add_sa3_to_path()

    import mlx.core as mx
    from models.defs.same_l_encoder import SAMELEncoder, load_model as load_encoder
    from models.defs.same_l_decoder import STRIDE

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No audio files in {audio_dir}")

    # Find encoder weights — try explicit path, then auto-search/download
    enc_path = encoder_weights
    if enc_path is None:
        candidates = [
            mlx_root / "models" / "mlx" / "same_l_encoder_f32.npz",
            sa3_root / "optimized" / "mlx" / "models" / "mlx" / "same_l_encoder_f32.npz",
        ]
        for c in candidates:
            if c.exists():
                enc_path = str(c)
                break
        if enc_path is None:
            try:
                from weights import ensure_local
                enc_path = str(ensure_local("models/mlx/same_l_encoder_f32.npz"))
            except Exception:
                pass
    print(json.dumps({"status": "loading_encoder", "weights": str(enc_path)}), flush=True)
    encoder = load_encoder(enc_path, dtype=mx.float32)
    mx.eval(encoder.parameters())

    # Patched pretransform params for sa3-medium
    patch_size = 256
    channels = 2
    downsampling_ratio = patch_size * STRIDE  # 256 * 16 = 4096

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile required: pip install soundfile")

    for idx, audio_path in enumerate(files):
        t0 = time.time()

        # Load audio
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != sample_rate:
            # Simple resample via scipy
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * sample_rate / sr))
            sr = sample_rate

        # Stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        elif audio.shape[-1] > 2:
            audio = audio[:, :2]

        # Trim to max length
        max_samples = int(max_seconds * sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        total_seconds = len(audio) / sample_rate

        # Pad to multiple of downsampling_ratio
        target_len = int(np.ceil(len(audio) / downsampling_ratio)) * downsampling_ratio
        if len(audio) < target_len:
            pad = np.zeros((target_len - len(audio), audio.shape[-1]), dtype=np.float32)
            audio_padded = np.concatenate([audio, pad], axis=0)
        else:
            audio_padded = audio

        # Create padding mask at latent resolution
        latent_len = target_len // downsampling_ratio
        real_latent_len = int(np.ceil(len(audio) / downsampling_ratio))
        padding_mask = [1] * real_latent_len + [0] * (latent_len - real_latent_len)

        # Patched pretransform encode: [samples, 2] → [1, 2, samples] → [1, 512, T_patches]
        x = mx.array(audio_padded.T[None, :, :])  # [1, 2, samples]
        # rearrange "b c (l h) -> b (c h) l" with h=patch_size
        B, C, L = x.shape
        T_patches = L // patch_size
        x = x.reshape(B, C, T_patches, patch_size)  # [1, 2, T, 256]
        x = x.transpose(0, 1, 3, 2)  # [1, 2, 256, T]
        x = x.reshape(B, C * patch_size, T_patches)  # [1, 512, T]

        # Encode through SAME-L
        # encoder expects channels-last: [B, T, 512]
        x_cl = x.transpose(0, 2, 1)  # [1, T, 512]
        latent_cl = encoder(x_cl)  # [1, T_lat, 256]
        latent = latent_cl.transpose(0, 2, 1)  # [1, 256, T_lat]

        mx.eval(latent)
        latent_np = np.array(latent[0]).astype(np.float16)  # [256, T_lat]

        # Save
        npy_path = output_dir / f"{idx:010d}.npy"
        json_path = output_dir / f"{idx:010d}.json"

        np.save(str(npy_path), latent_np)

        meta = {
            "path": str(audio_path),
            "relpath": audio_path.name,
            "seconds_total": round(total_seconds, 3),
            "padding_mask": padding_mask,
        }
        json_path.write_text(json.dumps(meta, indent=2))

        elapsed = time.time() - t0
        print(json.dumps({
            "status": "encoded",
            "idx": idx,
            "file": audio_path.name,
            "shape": list(latent_np.shape),
            "seconds": round(total_seconds, 1),
            "elapsed": round(elapsed, 2),
        }), flush=True)

    print(json.dumps({"status": "done", "total": len(files)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--encoder-weights", default=None,
                   help="Path to SAME-L encoder .npz (auto-downloaded if not set)")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--max-seconds", type=float, default=47.0)
    args = p.parse_args()

    encode_audio_files(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        encoder_weights=args.encoder_weights,
        sample_rate=args.sample_rate,
        max_seconds=args.max_seconds,
    )
