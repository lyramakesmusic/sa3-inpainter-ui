"""MLX LoRA training for SA3 DiT.

Standalone script run as subprocess — same JSON-line progress protocol
as the PyTorch train_lora.py so the backend/UI work unchanged.

Requires:
  - SA3_ROOT env var pointing to stable-audio-3 repo (for upstream MLX model code)
  - Pre-encoded latents (from CUDA pre-encode step or MLX encoder)
  - T5Gemma .npz weights (auto-downloaded from HuggingFace if missing)

Supports: lora, dora-rows, dora-cols, bora adapter types;
  T5Gemma conditioner LoRA; LogSNR distribution shift; gradient checkpointing.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def _add_sa3_to_path():
    sa3_root = Path(os.environ.get("SA3_ROOT", ""))
    if not sa3_root.is_dir():
        raise RuntimeError(
            "SA3_ROOT not set or invalid. Point it at the stable-audio-3 repo clone."
        )
    mlx_root = sa3_root / "optimized" / "mlx"
    if not mlx_root.is_dir():
        raise RuntimeError(f"MLX code not found at {mlx_root}")
    for p in [str(sa3_root), str(mlx_root), str(mlx_root / "scripts")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return sa3_root, mlx_root


def _find_model_ckpt(model_name: str, models_dir: str) -> Path:
    """Locate the SA3 safetensors checkpoint."""
    name_map = {
        "medium": "stable-audio-3-medium",
        "medium-base": "stable-audio-3-medium-base",
        "small-music": "stable-audio-3-small-music",
        "sm-music": "stable-audio-3-small-music",
        "small-sfx": "stable-audio-3-small-sfx",
        "sm-sfx": "stable-audio-3-small-sfx",
    }
    folder = name_map.get(model_name, f"stable-audio-3-{model_name}")
    ckpt = Path(models_dir) / folder / "model.safetensors"
    if not ckpt.exists():
        for suffix in ["ARC.safetensors", "model.safetensors"]:
            alt = Path(models_dir) / folder / f"stable-audio-3-{model_name}-{suffix}"
            if alt.exists():
                return alt
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt}")
    return ckpt


class PreEncodedDataset:
    """Loads pre-encoded latents and metadata from disk."""

    def __init__(self, encoded_dir: str, caption: str = ""):
        self.encoded_dir = Path(encoded_dir)
        self.caption = caption

        self.files = sorted(self.encoded_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {encoded_dir}")

        self.metadata = []
        for f in self.files:
            meta_path = f.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as fp:
                    self.metadata.append(json.load(fp))
            else:
                self.metadata.append({"seconds_total": 30.0})

        dataset_dir = self.encoded_dir.parent / "_dataset"
        if not caption and dataset_dir.is_dir():
            txt_files = sorted(dataset_dir.glob("*.txt"))
            if txt_files:
                self.caption = txt_files[0].read_text().strip()

    def __len__(self):
        return len(self.files)

    def sample_batch(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, len(self.files), size=batch_size)
        latents = []
        seconds = []
        for i in indices:
            lat = np.load(self.files[i]).astype(np.float32)
            latents.append(lat)
            seconds.append(float(self.metadata[i].get("seconds_total", 30.0)))
        return latents, seconds


# ── LogSNR Distribution Shift ──


def logsnr_shift_timesteps(t, seq_len, anchor_length=2000,
                           anchor_logsnr=-6.2, rate=1.0, logsnr_end=2.0):
    """Map t∈[0,1] through LogSNR space with adaptive bounds.

    Ports SA3's LogSNRShift.shift() to numpy for use before mx conversion.
    """
    log2_ratio = math.log2(seq_len / anchor_length)
    logsnr_start = anchor_logsnr - rate * log2_ratio
    logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
    t_out = 1.0 / (1.0 + np.exp(logsnr))  # sigmoid(-logsnr)
    t_out = np.where(t <= 0, 0.0, t_out)
    t_out = np.where(t >= 1, 1.0, t_out)
    return t_out


# ── T5Gemma conditioner wrapper for LoRA ──

def _make_combined_model_class():
    """Deferred class creation — mlx.nn must be importable."""
    import mlx.nn as nn

    class CombinedModel(nn.Module):
        """Wraps DiT + optional T5Gemma encoder as a single nn.Module
        so nn.value_and_grad can differentiate both."""

        def __init__(self, dit, t5_encoder=None):
            super().__init__()
            self.dit = dit
            self.t5_encoder = t5_encoder

        def __call__(self, x, t_arr, cross_cond, global_cond):
            return self.dit(x, t_arr, cross_cond, global_cond)

        def forward_with_text(self, x, t_arr, caption_ids, caption_mask,
                              padding_emb, global_cond):
            from models.defs.sa3_pipeline import apply_prompt_padding
            embeds = self.t5_encoder(caption_ids, caption_mask)
            cross_cond = apply_prompt_padding(embeds, caption_mask, padding_emb)
            return self.dit(x, t_arr, cross_cond, global_cond)

    return CombinedModel


def train(args):
    sa3_root, mlx_root = _add_sa3_to_path()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from models.defs.dit_mlx_medium import DiT, load_dit
    from models.defs.t5gemma_mlx import T5Gemma
    from models.defs.sa3_pipeline import (
        load_conditioner_from_sa3_ckpt,
        apply_prompt_padding,
    )

    lora_dir = Path(__file__).resolve().parent
    if str(lora_dir) not in sys.path:
        sys.path.insert(0, str(lora_dir.parent))
    from mlx_sa3.lora import (
        inject_lora,
        freeze_base_unfreeze_lora,
        get_lora_state_dict,
        save_lora_safetensors,
    )

    _log({"status": "init", "model": args.model_name, "rank": args.rank,
          "adapter_type": args.adapter_type})

    # ── Load dataset ──
    dataset = PreEncodedDataset(args.encoded_dir, caption=args.caption)
    _log({"status": "dataset_loaded", "samples": len(dataset),
          "caption": dataset.caption})

    sample_lat = np.load(dataset.files[0])
    T_lat = sample_lat.shape[-1]
    _log({"status": "latent_info", "shape": list(sample_lat.shape),
          "T_lat": T_lat})

    # ── Load DiT ──
    _log({"status": "loading_dit"})
    ckpt_path = _find_model_ckpt(args.model_name, args.models_dir)
    model = load_dit(str(ckpt_path), T_lat=T_lat, dtype=mx.float32)
    mx.eval(model.parameters())
    _log({"status": "dit_loaded", "ckpt": str(ckpt_path)})

    # ── Load T5Gemma + conditioner ──
    _log({"status": "loading_t5gemma"})
    t5_npz = _find_t5gemma_npz(mlx_root, sa3_root)
    t5 = T5Gemma.from_npz(str(t5_npz))
    _log({"status": "t5gemma_loaded"})

    padding_emb, sec_embedder = load_conditioner_from_sa3_ckpt(str(ckpt_path))

    caption = dataset.caption or args.caption or ""
    if not caption:
        _log({"status": "warning", "msg": "No caption provided, using empty string"})

    # ── Set up conditioner LoRA + combined model ──
    if args.train_conditioner:
        n_cond_lora = inject_lora(
            t5.encoder, rank=args.rank, alpha=args.alpha,
            adapter_type=args.adapter_type,
        )
        _log({"status": "conditioner_lora", "adapters": n_cond_lora})
        cross_attn_cond = None
    else:
        embeds, mask = t5.encode([caption], max_len=256)
        cross_attn_cond = apply_prompt_padding(embeds, mask, padding_emb)
        mx.eval(cross_attn_cond)
        n_cond_lora = 0
        del t5
        import gc; gc.collect()
        _log({"status": "text_encoded", "caption": caption,
              "embed_shape": list(cross_attn_cond.shape)})

    # ── Inject LoRA into DiT ──
    n_lora = inject_lora(
        model,
        rank=args.rank,
        alpha=args.alpha,
        adapter_type=args.adapter_type,
        include=args.include,
        exclude=args.exclude,
    )
    _log({"status": "lora_injected", "adapters": n_lora,
          "adapter_type": args.adapter_type})

    # ── Wrap in CombinedModel for joint gradient computation ──
    CombinedModel = _make_combined_model_class()
    combined = CombinedModel(
        model,
        t5_encoder=t5.encoder if args.train_conditioner else None,
    )
    freeze_base_unfreeze_lora(combined)

    # ── Apply gradient checkpointing ──
    if args.grad_checkpoint:
        _apply_gradient_checkpointing(model)
        _log({"status": "grad_checkpoint_enabled"})

    # ── Count params ──
    n_train = sum(p.size for _, p in nn.utils.tree_flatten(combined.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(combined.parameters()))
    _log({"status": "params", "trainable": n_train, "total": n_total,
          "pct": round(100 * n_train / max(n_total, 1), 2)})

    # ── Optimizer ──
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    # ── Training state ──
    rng = np.random.default_rng(42)
    save_dir = Path(args.output_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_config = {
        "rank": args.rank,
        "alpha": args.alpha,
        "adapter_type": args.adapter_type,
    }
    if args.exclude:
        lora_config["exclude"] = args.exclude
    if args.train_conditioner:
        lora_config["train_conditioner"] = True

    _log({"status": "training", "steps": args.steps,
          "batch_size": args.batch_size, "lr": args.lr,
          "dist_shift": args.dist_shift})

    # ── Loss function ──
    # nn.value_and_grad differentiates w.r.t. the first arg.
    # When training conditioner, we pass `combined` so both DiT and T5 get grads.

    if args.train_conditioner:
        def loss_fn(combined, x, noise, t_arr, caption_ids,
                    caption_mask, global_cond):
            t = t_arr[:, None, None]
            noised = x * (1.0 - t) + noise * t
            v = combined.forward_with_text(
                noised, t_arr, caption_ids, caption_mask,
                padding_emb, global_cond,
            )
            target = noise - x
            return mx.mean((v - target) ** 2)

        loss_and_grad = nn.value_and_grad(combined, loss_fn)
    else:
        def loss_fn(combined, x, noise, t_arr, cross_cond, global_cond):
            t = t_arr[:, None, None]
            noised = x * (1.0 - t) + noise * t
            v = combined(noised, t_arr, cross_cond, global_cond)
            target = noise - x
            return mx.mean((v - target) ** 2)

        loss_and_grad = nn.value_and_grad(combined, loss_fn)

    t0 = time.time()
    running_loss = 0.0

    # Pre-tokenize caption for conditioner training
    if args.train_conditioner:
        caption_ids, caption_mask = t5.tokenize([caption], max_len=256)
        mx.eval(caption_ids, caption_mask)

    for step in range(1, args.steps + 1):
        latents, seconds = dataset.sample_batch(args.batch_size, rng)
        x = mx.array(np.stack(latents, axis=0))
        noise = mx.random.normal(x.shape)

        # Timestep sampling
        t_np = rng.uniform(0.01, 0.99, size=(args.batch_size,)).astype(np.float32)
        if args.dist_shift:
            t_np = logsnr_shift_timesteps(
                t_np, seq_len=T_lat,
                anchor_logsnr=args.anchor_logsnr,
                rate=args.dist_shift_rate,
                logsnr_end=args.logsnr_end,
            )
        t_arr = mx.array(t_np)

        # Global conditioning
        global_cond = sec_embedder(seconds)[:, 0, :]

        # Forward + backward
        if args.train_conditioner:
            batch_ids = mx.broadcast_to(caption_ids, (args.batch_size,) + caption_ids.shape[1:])
            batch_mask = mx.broadcast_to(caption_mask, (args.batch_size,) + caption_mask.shape[1:])
            loss, grads = loss_and_grad(
                combined, x, noise, t_arr,
                batch_ids, batch_mask, global_cond
            )
            optimizer.update(combined, grads)
            mx.eval(combined.parameters(), optimizer.state, loss)
        else:
            cross_cond = mx.broadcast_to(
                cross_attn_cond, (args.batch_size,) + cross_attn_cond.shape[1:]
            )
            loss, grads = loss_and_grad(
                combined, x, noise, t_arr, cross_cond, global_cond
            )
            optimizer.update(combined, grads)
            mx.eval(combined.parameters(), optimizer.state, loss)

        loss_val = float(loss)
        running_loss = 0.95 * running_loss + 0.05 * loss_val if step > 1 else loss_val

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            it_s = step / elapsed if elapsed > 0 else 0
            _log({
                "status": "step", "step": step,
                "loss": round(loss_val, 6),
                "avg_loss": round(running_loss, 6),
                "it_s": round(it_s, 2),
            })

        if step % args.checkpoint_every == 0 or step == args.steps:
            ckpt_path_st = save_dir / f"lora-step{step:06d}.safetensors"
            dit_state = get_lora_state_dict(combined.dit)
            cond_state = get_lora_state_dict(combined.t5_encoder) if args.train_conditioner else None
            n_keys = save_lora_safetensors(
                dit_state, cond_state, lora_config, str(ckpt_path_st)
            )
            _log({
                "status": "checkpoint", "step": step,
                "path": str(ckpt_path_st), "keys": n_keys,
                "loss": round(running_loss, 6),
            })

    elapsed = time.time() - t0
    _log({"status": "done", "steps": args.steps,
          "final_loss": round(running_loss, 6),
          "elapsed_s": round(elapsed, 1)})

    lora_out_dir = Path(os.environ.get("SA3_LORA_DIR", str(Path.home() / "loras")))
    lora_out_dir.mkdir(parents=True, exist_ok=True)
    lora_name = Path(args.output_dir).name
    final_ckpt = save_dir / f"lora-step{args.steps:06d}.safetensors"
    if final_ckpt.exists():
        import shutil
        dst = lora_out_dir / f"{lora_name}.safetensors"
        shutil.copy2(str(final_ckpt), str(dst))
        _log({"status": "saved", "output": str(dst), "name": lora_name})


# ── Gradient checkpointing ──


def _apply_gradient_checkpointing(model):
    """Wrap each transformer block's forward in mx.checkpoint."""
    import mlx.core as mx

    if not hasattr(model, "transformer"):
        return
    transformer = model.transformer
    if not hasattr(transformer, "layers"):
        return

    for i, layer in enumerate(transformer.layers):
        original_call = layer.__call__

        def make_checkpointed(fn):
            def checkpointed_call(*args, **kwargs):
                return mx.checkpoint(fn)(*args, **kwargs)
            return checkpointed_call

        layer.__call__ = make_checkpointed(original_call)


def _find_t5gemma_npz(mlx_root: Path, sa3_root: Path) -> Path:
    candidates = [
        mlx_root / "models" / "mlx" / "t5gemma_f16.npz",
        sa3_root / "optimized" / "mlx" / "models" / "mlx" / "t5gemma_f16.npz",
    ]
    for c in candidates:
        if c.exists():
            return c

    try:
        from weights import ensure_local
        return ensure_local("models/mlx/t5gemma_f16.npz")
    except Exception as e:
        raise FileNotFoundError(
            f"T5Gemma .npz not found at any of: {candidates}. "
            f"Download from HuggingFace stabilityai/stable-audio-3-optimized MLX/t5gemma_f16.npz"
        ) from e


def _log(data: dict):
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MLX LoRA training for SA3")
    p.add_argument("--model-name", default="medium")
    p.add_argument("--encoded-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--adapter-type", default="lora",
                   choices=["lora", "dora-rows", "dora-cols", "bora"])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--include", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)
    # Conditioner LoRA
    p.add_argument("--train-conditioner", action="store_true", default=False,
                   help="Also train LoRA on T5Gemma conditioner")
    # Distribution shift
    p.add_argument("--dist-shift", action="store_true", default=False,
                   help="Apply LogSNR distribution shift to timestep sampling")
    p.add_argument("--anchor-logsnr", type=float, default=-6.2)
    p.add_argument("--dist-shift-rate", type=float, default=1.0)
    p.add_argument("--logsnr-end", type=float, default=2.0)
    # Gradient checkpointing
    p.add_argument("--grad-checkpoint", action="store_true", default=False,
                   help="Enable gradient checkpointing to reduce memory usage")
    args = p.parse_args()

    if args.alpha is None:
        args.alpha = float(args.rank)

    if args.models_dir is None:
        args.models_dir = os.environ.get(
            "SA3_MODELS_DIR", str(Path.home() / "sa3-inpainter" / "models")
        )

    train(args)
