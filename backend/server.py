"""SA3 Inpainter backend. FastAPI on :5174.

Loads the SA3 medium model once at startup (~30s), exposes JSON API for the
Svelte frontend.

Dual backend: auto-detects CUDA (torch AE) vs MPS (MLX AE).
Set SA3_BACKEND=cuda or SA3_BACKEND=mlx to force.
"""
import asyncio
import gc
import shutil
import os, sys, json, time, threading, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.distribution_shift import (
    IdentityDistributionShift, FluxDistributionShift, DistributionShift, LogSNRShift
)
from safetensors.torch import load_file

# -------- backend detection --------

def _detect_backend():
    forced = os.environ.get("SA3_BACKEND", "").lower()
    if forced == "cuda":
        return "cuda", "cuda"
    if forced == "mlx":
        return "mlx", "mps"
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mlx", "mps"
    return "cuda", "cpu"

BACKEND, DEVICE = _detect_backend()
HAS_MLX = BACKEND == "mlx"

if HAS_MLX:
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights

# -------- settings persistence --------

_SETTINGS_FILE = Path(os.environ.get(
    "SA3_SETTINGS_FILE",
    str(Path.home() / ".config" / "sa3-inpainter" / "settings.json"),
))

_SETTINGS_DEFAULTS = {
    "models_dir": str(Path.home() / "sa3-inpainter" / "models"),
    "lora_dir": str(Path.home() / "sa3-inpainter" / "loras"),
    "lora_train_dir": str(Path.home() / "sa3-inpainter" / "lora_training"),
    "embeddings_dir": str(Path.home() / "sa3-inpainter" / "embeddings"),
    "sa3_root": "",
    "hf_token": "",
}

def _load_settings():
    if _SETTINGS_FILE.exists():
        try:
            saved = json.loads(_SETTINGS_FILE.read_text())
            merged = dict(_SETTINGS_DEFAULTS)
            merged.update({k: v for k, v in saved.items() if k in _SETTINGS_DEFAULTS})
            return merged
        except Exception:
            pass
    return dict(_SETTINGS_DEFAULTS)

def _save_settings(s):
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_FILE.write_text(json.dumps(s, indent=2))

_settings = _load_settings()

def _apply_settings():
    global MODELS_DIR, LOCAL_MEDIUM, LORA_DIR, LORA_TRAIN_DIR, EMBED_DIR
    MODELS_DIR = Path(_settings["models_dir"])
    LOCAL_MEDIUM = str(MODELS_DIR / "stable-audio-3-medium")
    LORA_DIR = Path(_settings["lora_dir"])
    LORA_TRAIN_DIR = Path(_settings["lora_train_dir"])
    EMBED_DIR = Path(_settings["embeddings_dir"])

MODELS_DIR = Path(os.environ.get("SA3_MODELS_DIR", _settings["models_dir"]))
LOCAL_MEDIUM = os.environ.get("SA3_MODEL_DIR", str(MODELS_DIR / "stable-audio-3-medium"))
LORA_DIR = Path(os.environ.get("SA3_LORA_DIR", _settings["lora_dir"]))
LORA_TRAIN_DIR = Path(os.environ.get("SA3_LORA_TRAIN_DIR", _settings["lora_train_dir"]))
EMBED_DIR = Path(os.environ.get("SA3_EMBED_DIR", _settings["embeddings_dir"]))
DATA_DIR = Path("/tmp/sa3-inpainter"); DATA_DIR.mkdir(exist_ok=True)
SR = 44100
DOWNSAMPLE = 4096
BANDS = [(0, 250), (250, 2500), (2500, 22050)]

# map model names → HF repo IDs for download-once caching
_MODEL_REPOS = {
    "medium":       "stabilityai/stable-audio-3-medium",
    "medium-base":  "stabilityai/stable-audio-3-medium-base",
    "small-music":  "stabilityai/stable-audio-3-small-music",
    "small-sfx":    "stabilityai/stable-audio-3-small-sfx",
}


def _resolve_local_path(name):
    """Return a local directory for a model, downloading from HF once if needed."""
    from huggingface_hub import snapshot_download
    local_dir = MODELS_DIR / f"stable-audio-3-{name}"
    if (local_dir / "model.safetensors").exists() and (local_dir / "model_config.json").exists():
        print(f"[backend] {name} found at {local_dir}")
        return str(local_dir)
    repo_id = _MODEL_REPOS.get(name)
    if not repo_id:
        return None
    print(f"[backend] downloading {name} from {repo_id} → {local_dir} (one-time)")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[backend] download complete: {local_dir}")
    return str(local_dir)


# -------- model state --------

sa = None               # StableAudioModel instance
mlx_ae = None           # MLX AE, only for medium on MPS
_use_mlx_ae = False     # whether current model uses MLX AE decode path
_current_model = None   # name of loaded model
_use_fp16 = os.environ.get("SA3_FP16", "0") == "1"
_cancel_event = threading.Event()
_loaded_lora_name = None
_default_memory_tokens = None   # snapshot of original memory tokens
_memory_token_strength = 1.0    # user-controllable scale factor
_training_unloaded_model = None # model name to reload after training


def _load_model(name, local_path=None):
    """Load a model by name. Resolves to a local path, downloading once if needed."""
    global sa, mlx_ae, _use_mlx_ae, _current_model, _loaded_lora_name

    # cleanup old model
    sa = None
    mlx_ae = None
    _use_mlx_ae = False
    _loaded_lora_name = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if not local_path:
        local_path = _resolve_local_path(name)

    print(f"[backend] loading {name} on {DEVICE}...")
    want_half = _use_fp16 and DEVICE == "cuda"

    if local_path:
        cfg_path = f"{local_path}/model_config.json"
        ckpt_path = f"{local_path}/model.safetensors"
        cfg = json.load(open(cfg_path))
        for c in cfg["model"]["conditioning"]["configs"]:
            if c["type"] == "t5gemma":
                c["config"]["repo_id"] = local_path
        model = create_diffusion_cond_from_config(cfg)
        model.load_state_dict(load_file(ckpt_path), strict=False)
        model.eval().requires_grad_(False).to(DEVICE)
        if want_half:
            model.half()
        sa = StableAudioModel(model, cfg, device=DEVICE, model_half=want_half)

        if HAS_MLX and name.startswith("medium"):
            print("[backend] loading MLX AE...")
            mlx_ae = SA3MediumAE()
            load_ae_weights(mlx_ae, ckpt_path)
            _use_mlx_ae = True
            print("[backend] MLX AE loaded")
    else:
        sa = StableAudioModel.from_pretrained(name, device=DEVICE, model_half=want_half)

    _current_model = name

    # snapshot default memory tokens for strength scaling
    global _default_memory_tokens, _memory_token_strength
    _memory_token_strength = 1.0
    try:
        mt = sa.model.model.transformer.memory_tokens
        _default_memory_tokens = mt.data.clone()
        print(f"[backend] memory tokens: {mt.shape}")
    except AttributeError:
        _default_memory_tokens = None

    print(f"[backend] {name} ready (mlx_ae={'yes' if _use_mlx_ae else 'no'})")


def _unload_model():
    global sa, mlx_ae, _use_mlx_ae, _loaded_lora_name, _default_memory_tokens
    sa = None
    mlx_ae = None
    _use_mlx_ae = False
    _loaded_lora_name = None
    _default_memory_tokens = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("[backend] model unloaded")


# initial load
_load_model("medium", local_path=LOCAL_MEDIUM)
if _use_fp16 and DEVICE == "cuda":
    print("[backend] fp16 enabled")

# register RES4LYF exponential RK samplers
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
from res4lyf.sampler import register_samplers, SAMPLER_NAMES as RES4LYF_NAMES
register_samplers()

from kv_cache import enable_kv_cache, disable_kv_cache, clear_kv_cache
from tome import apply_tome, remove_tome


# -------- decode helpers --------

_decode_fp32 = True
_decode_overlap = 32

def _decode_latents(lat_np):
    """Decode latent numpy array → waveform numpy array.
    lat_np: (1, 256, T_lat) float32
    Returns: (channels, T_audio) float32 numpy
    """
    if _use_mlx_ae and mlx_ae is not None:
        lat_mx = mx.array(lat_np)
        if lat_np.shape[-1] > 128:
            wav_m = decode_chunked(mlx_ae, lat_mx, chunk_size=128, overlap=_decode_overlap)
        else:
            wav_m = mlx_ae.decode(lat_mx)
        mx.eval(wav_m)
        return np.array(wav_m)[0]
    else:
        lat_t = torch.from_numpy(lat_np).to(DEVICE)
        use_fp32 = _decode_fp32 and DEVICE == "cuda" and _use_fp16
        with torch.inference_mode():
            if use_fp32:
                sa.same.float()
                lat_t = lat_t.float()
                wav_t = sa.same.decode(lat_t, chunked=True, overlap=_decode_overlap, chunk_size=128)
                sa.same.half()
            else:
                if _use_fp16 and DEVICE == "cuda":
                    lat_t = lat_t.half()
                wav_t = sa.same.decode(lat_t, chunked=True, overlap=_decode_overlap, chunk_size=128)
        return wav_t.float().cpu().numpy()[0]


def render_noise_spec_once():
    out_path = DATA_DIR / "noise_spec.png"
    if out_path.exists(): return
    T_lat = int(30 * SR / DOWNSAMPLE) + 1
    rng = np.random.default_rng(7)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    wav_np = _decode_latents(lat)
    render_spec_png(wav_np, out_path)

state = {"audio_path": None, "version": 0, "bpm": None}

# -------- audio history (undo/redo) --------

HISTORY_DIR = DATA_DIR / "history"; HISTORY_DIR.mkdir(exist_ok=True)
_audio_undo = []   # stack of (wav_path, bpm)
_audio_redo = []   # stack of (wav_path, bpm)
MAX_HISTORY = 30


def _snapshot_current():
    """Save the current audio as a history entry. Call BEFORE replacing it."""
    if state["audio_path"] is None:
        return
    idx = len(_audio_undo)
    dst = HISTORY_DIR / f"snap_{idx}.wav"
    shutil.copy2(state["audio_path"], dst)
    _audio_undo.append((str(dst), state.get("bpm")))
    if len(_audio_undo) > MAX_HISTORY:
        old_path, _ = _audio_undo.pop(0)
        Path(old_path).unlink(missing_ok=True)
    _audio_redo.clear()


def _restore_snapshot(entry):
    """Restore audio from a history entry. Returns envelope info."""
    wav_path, bpm = entry
    audio, _ = sf.read(wav_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    env = persist_audio(audio.T)
    state["bpm"] = bpm
    return env


app = FastAPI()


def compute_envelope(audio_np):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    N = len(mono) // DOWNSAMPLE
    freqs = np.fft.rfftfreq(DOWNSAMPLE, 1.0 / SR)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    data = []
    for i in range(N):
        seg = mono[i*DOWNSAMPLE:(i+1)*DOWNSAMPLE]
        peak = float(np.abs(seg).max())
        spec = np.abs(np.fft.rfft(seg)) ** 2
        e = [float(spec[m].sum()) for m in masks]
        total = sum(e) + 1e-12
        rgb = [(v/total) ** 0.6 for v in e]
        mx_ = max(rgb) + 1e-12
        rgb = [v/mx_ for v in rgb]
        data.append([round(peak, 4)] + [round(c, 3) for c in rgb])
    return {"sr": SR, "downsample": DOWNSAMPLE, "count": N, "data": data}


def render_spec_png(audio_np, out_path):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    n_fft = 8192
    hop = DOWNSAMPLE
    f, t, Z = stft(mono, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    P_db = 10.0 * np.log10(P + 1e-12)
    P_db = np.clip(P_db, -55, P_db.max()); P_db -= P_db.min()
    if P_db.max() < 1e-6:
        P_db = np.zeros_like(P_db)
    else:
        P_db /= P_db.max()
        P_db = P_db ** 0.55
    out_h = 600
    log_f = np.geomspace(30, 16000, out_h)
    spec_log = np.zeros((out_h, P_db.shape[1]), dtype=np.float32)
    for j in range(P_db.shape[1]):
        spec_log[:, j] = np.interp(log_f, f, P_db[:, j])
    fig = plt.figure(figsize=(20, 6), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off()
    ax.imshow(spec_log[::-1], aspect="auto", origin="upper", cmap="magma", interpolation="nearest", extent=(0,1,0,1))
    fig.savefig(out_path, dpi=100, facecolor="black")
    plt.close(fig)


def render_overview_png(audio_np, out_path, W=2000, H=80):
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    bin_sz = max(1, len(mono) // W)
    peaks = np.zeros(W)
    for i in range(W):
        s = i * bin_sz
        peaks[i] = np.max(np.abs(mono[s:s+bin_sz])) if s < len(mono) else 0
    peaks /= peaks.max() + 1e-9
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor("#000000")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, W); ax.set_ylim(-1.05, 1.05)
    ax.vlines(np.arange(W), -peaks, peaks, color="#666666", linewidth=0.7)
    fig.savefig(out_path, dpi=100, facecolor="#000000")
    plt.close(fig)


def persist_audio(audio_np):
    """audio_np: (2, T) float in [-1, 1]."""
    p = DATA_DIR / "current.wav"
    sf.write(p, audio_np.T, SR)
    state["audio_path"] = str(p)
    state["version"] += 1
    env = compute_envelope(audio_np)
    with open(DATA_DIR / "envelope.json", "w") as fh:
        json.dump(env, fh)
    threading.Thread(target=render_spec_png, args=(audio_np, DATA_DIR / "current_spec.png"), daemon=True).start()
    threading.Thread(target=render_overview_png, args=(audio_np, DATA_DIR / "current_overview.png"), daemon=True).start()
    return env


# -------- LoRA helpers --------

def _apply_loras(loras: list[dict]) -> None:
    global _loaded_lora_name
    if not loras:
        return
    for entry in loras:
        name = entry["name"]
        strength = float(entry.get("strength", 1.0))
        lora_path = LORA_DIR / name
        if not lora_path.exists():
            print(f"[lora] not found: {lora_path}, skipping")
            continue
        if _loaded_lora_name != name:
            sa.load_lora([str(lora_path)])
            _loaded_lora_name = name
            print(f"[lora] loaded {name}")
        sa.set_lora_strength(strength)
        print(f"[lora] strength {name} @ {strength}")
        return


def _unload_loras(loras: list[dict]) -> None:
    if _loaded_lora_name is None:
        return
    sa.set_lora_strength(0.0)
    print(f"[lora] deactivated {_loaded_lora_name}")


# -------- embedding injection --------

_embed_cache = {}  # name → (trigger, tensor)


def _load_embedding(name):
    """Load a textual inversion embedding from EMBED_DIR. Returns (trigger, tensor) or None."""
    if name in _embed_cache:
        return _embed_cache[name]
    path = EMBED_DIR / f"{name}.safetensors"
    if not path.exists():
        return None
    data = load_file(str(path))
    emb = data.get("embedding", data.get("emb", next(iter(data.values()))))
    emb = emb.to(DEVICE)
    if _use_fp16 and DEVICE == "cuda":
        emb = emb.half()
    trigger = f"<{name}>"
    _embed_cache[name] = (trigger, emb)
    print(f"[embed] loaded {name}: {emb.shape}")
    return trigger, emb


def _find_embeddings_in_prompt(prompt):
    """Find <name> or <name:strength> triggers in prompt, return list of (name, tensor, strength)."""
    import re
    matches = re.findall(r"<([^>]+)>", prompt)
    found = []
    for m in matches:
        parts = m.split(":")
        name = parts[0]
        strength = float(parts[1]) if len(parts) > 1 else 1.0
        strength = max(0.01, min(2.0, strength))
        result = _load_embedding(name)
        if result:
            found.append((name, result[1], strength))
    return found


def _strip_embed_triggers(prompt):
    """Remove <name> and <name:strength> triggers from prompt text."""
    import re
    return re.sub(r"<[^>]+>", "", prompt).strip()


def _inject_embeddings(prompt, conditioning_tensors):
    """Inject learned embeddings into conditioning, replacing padding at the end.
    Modifies conditioning_tensors['prompt'][0] in-place."""
    embeds = _find_embeddings_in_prompt(prompt)
    if not embeds:
        return
    cond = conditioning_tensors.get("prompt")
    if cond is None:
        return
    embed_tensor, mask = cond
    real_len = mask[0].sum().item() if mask is not None else embed_tensor.shape[1]
    insert_pos = int(real_len)
    for name, emb, strength in embeds:
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        n_tok = emb.shape[0]
        if insert_pos + n_tok > embed_tensor.shape[1]:
            continue
        embed_tensor[0, insert_pos:insert_pos + n_tok] = emb[:n_tok] * strength
        if mask is not None:
            mask[0, insert_pos:insert_pos + n_tok] = True
        print(f"[embed] injected {name} ({n_tok} tokens, strength={strength:.2f}) at pos {insert_pos}")
        insert_pos += n_tok


# -------- endpoints --------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    _snapshot_current()
    raw = DATA_DIR / ("upload_raw" + Path(file.filename or ".wav").suffix)
    with open(raw, "wb") as f: f.write(await file.read())
    tag_bpm = read_bpm_tag(str(raw))
    audio, sr = sf.read(raw)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=-1)
    if sr != SR:
        import torchaudio
        a = torch.from_numpy(audio.T).float()
        a = torchaudio.transforms.Resample(sr, SR)(a)
        audio = a.numpy().T
    env = persist_audio(audio.T)
    bpm_source = "tag" if tag_bpm else "detect"
    bpm = tag_bpm if tag_bpm else detect_bpm(audio.T, SR)
    print(f"[bpm] {bpm_source}: {bpm}")
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm,
            "bpm_source": bpm_source}


@app.post("/api/detect_bpm")
async def redetect_bpm():
    if state["audio_path"] is None:
        raise HTTPException(400, "no audio loaded")
    audio, _ = sf.read(state["audio_path"])
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    bpm = detect_bpm(audio.T, SR)
    state["bpm"] = bpm
    print(f"[bpm] re-detect: {bpm}")
    return {"bpm": bpm, "bpm_source": "detect"}


class LoraEntry(BaseModel):
    name: str
    strength: float = 1.0


class GenBody(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    mask: list[int] = []
    settings: dict = {}
    loras: list[LoraEntry] = []


@app.post("/api/cancel")
async def cancel():
    _cancel_event.set()
    return {"status": "cancelling"}


@app.post("/api/undo")
async def undo_audio():
    if not _audio_undo:
        raise HTTPException(400, "nothing to undo")
    if state["audio_path"]:
        idx = len(_audio_undo) + len(_audio_redo)
        dst = HISTORY_DIR / f"snap_{idx}.wav"
        shutil.copy2(state["audio_path"], dst)
        _audio_redo.append((str(dst), state.get("bpm")))
    entry = _audio_undo.pop()
    env = _restore_snapshot(entry)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": state["bpm"],
            "can_undo": len(_audio_undo) > 0, "can_redo": len(_audio_redo) > 0}


@app.post("/api/redo")
async def redo_audio():
    if not _audio_redo:
        raise HTTPException(400, "nothing to redo")
    if state["audio_path"]:
        idx = len(_audio_undo) + len(_audio_redo)
        dst = HISTORY_DIR / f"snap_{idx}.wav"
        shutil.copy2(state["audio_path"], dst)
        _audio_undo.append((str(dst), state.get("bpm")))
    entry = _audio_redo.pop()
    env = _restore_snapshot(entry)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": state["bpm"],
            "can_undo": len(_audio_undo) > 0, "can_redo": len(_audio_redo) > 0}


@app.post("/api/generate")
async def generate(body: GenBody):
    _snapshot_current()
    _cancel_event.clear()

    s = body.settings
    steps = int(s.get("steps", 8))
    cfg = float(s.get("cfg", 1.0))
    seed = int(s.get("seed", 42))
    if seed == -1:
        seed = int(np.random.randint(0, 999999))
    noise = float(s.get("noise", 1.0))
    duration = float(s.get("duration", 30.0))
    sampler_type = s.get("sampler_type")
    apg_scale = float(s.get("apg_scale", 1.0))

    # advanced params
    scale_phi = float(s.get("scale_phi", 0.0))
    cfg_interval = s.get("cfg_interval", [0.0, 1.0])
    cfg_norm_threshold = float(s.get("cfg_norm_threshold", 0.0))
    exit_layer_ix_raw = s.get("exit_layer_ix")
    exit_layer_ix = int(exit_layer_ix_raw) if exit_layer_ix_raw else None
    duration_padding_sec = float(s.get("duration_padding_sec", 6.0))

    # inference speedup params
    kv_cache_enabled = bool(s.get("kvCache", False))
    tome_ratio = float(s.get("tomeRatio", 0.0))

    # dist_shift construction
    dist_shift_type = s.get("dist_shift_type", "default")
    dist_shift = None
    if dist_shift_type == "none":
        dist_shift = IdentityDistributionShift()
    elif dist_shift_type == "logsnr":
        dist_shift = LogSNRShift(
            anchor_logsnr=float(s.get("dist_shift_anchor_logsnr", -6.2)),
            rate=float(s.get("dist_shift_rate", 0.0)),
            logsnr_end=float(s.get("dist_shift_logsnr_end", 2.0)),
        )
    elif dist_shift_type == "flux":
        dist_shift = FluxDistributionShift(
            alpha_min=float(s.get("dist_shift_alpha_min", 1.0)),
            alpha_max=float(s.get("dist_shift_alpha_max", 1.0)),
        )
    elif dist_shift_type == "full":
        dist_shift = DistributionShift(
            base_shift=float(s.get("dist_shift_base_shift", 0.5)),
            max_shift=float(s.get("dist_shift_max_shift", 1.15)),
        )

    has_source = state["audio_path"] is not None
    has_mask = any(body.mask) if body.mask else False
    n_regen = sum(body.mask) if body.mask else 0
    print(f"[generate] source={has_source} mask_len={len(body.mask) if body.mask else 0} regen_latents={n_regen} mode={('inpaint' if has_source and has_mask else 'vary' if has_source else 't2a')}")

    neg_prompt = body.negative_prompt or None
    kwargs = dict(prompt=body.prompt, negative_prompt=neg_prompt, steps=steps,
                  cfg_scale=cfg, seed=seed, apg_scale=apg_scale,
                  duration_padding_sec=duration_padding_sec,
                  return_latents=True,
                  chunked_decode=False)
    if sampler_type:
        kwargs["sampler_type"] = sampler_type
    if dist_shift is not None:
        kwargs["dist_shift"] = dist_shift
    if scale_phi != 0.0:
        kwargs["scale_phi"] = scale_phi
    if cfg_interval != [0.0, 1.0]:
        kwargs["cfg_interval"] = tuple(cfg_interval)
    if cfg_norm_threshold > 0:
        kwargs["cfg_norm_threshold"] = cfg_norm_threshold
    if exit_layer_ix is not None:
        kwargs["exit_layer_ix"] = exit_layer_ix

    audio_mask = None
    if has_source and has_mask:
        audio, _ = sf.read(state["audio_path"])
        audio_t = torch.from_numpy(audio.T).float().to(DEVICE)
        actual_lat = audio.shape[0] // DOWNSAMPLE
        mask_lat = np.asarray(body.mask, dtype=np.float32)
        if len(mask_lat) > actual_lat:
            mask_lat = mask_lat[:actual_lat]
        elif len(mask_lat) < actual_lat:
            mask_lat = np.pad(mask_lat, (0, actual_lat - len(mask_lat)), constant_values=0)
        inv = 1.0 - mask_lat
        audio_mask = np.repeat(inv, DOWNSAMPLE)
        audio_mask = audio_mask[:audio.shape[0]]
        print(f"[inpaint] mask aligned: {len(mask_lat)} latents, {int(mask_lat.sum())} regen, {audio.shape[0]} samples")
        kwargs["duration"] = audio.shape[0] / SR
        kwargs["sample_size"] = audio.shape[0]
        kwargs["inpaint_audio"] = (SR, audio_t)
        kwargs["inpaint_mask"] = torch.from_numpy(audio_mask).unsqueeze(0).to(DEVICE)
    elif has_source:
        audio, _ = sf.read(state["audio_path"])
        audio_t = torch.from_numpy(audio.T).float().to(DEVICE)
        kwargs["duration"] = audio.shape[0] / SR
        kwargs["sample_size"] = audio.shape[0]
        kwargs["init_audio"] = (SR, audio_t)
        kwargs["init_noise_level"] = noise
    else:
        kwargs["duration"] = duration
        kwargs["sample_size"] = int(duration * SR)

    # pre-encode conditioning if embeddings are used
    has_embeds = bool(_find_embeddings_in_prompt(body.prompt))
    if has_embeds:
        clean_prompt = _strip_embed_triggers(body.prompt)
        cond_dicts, neg_cond_dicts = sa._build_conditioning_dicts(
            clean_prompt, neg_prompt,
            kwargs.get("duration", duration), 1
        )
        cond_tensors = sa.model.conditioner(cond_dicts, DEVICE)
        _inject_embeddings(body.prompt, cond_tensors)
        neg_cond_tensors = sa.model.conditioner(neg_cond_dicts, DEVICE) if neg_cond_dicts else {}
        kwargs.pop("prompt", None)
        kwargs.pop("negative_prompt", None)
        kwargs["conditioning"] = cond_dicts
        kwargs["conditioning_tensors"] = cond_tensors
        kwargs["negative_conditioning_tensors"] = neg_cond_tensors

    loras_list = [l.model_dump() for l in body.loras]

    def _run_generate():
        _apply_loras(loras_list)
        _dit = sa.model.model
        try:
            if _cancel_event.is_set():
                return None

            # Apply inference speedups
            if kv_cache_enabled:
                enable_kv_cache(_dit)
            if tome_ratio > 0.0:
                apply_tome(_dit, ratio=tome_ratio)

            t0 = time.time()
            result = sa.generate(**kwargs)

            lat_np = result.detach().to(torch.float32).cpu().numpy()
            print(f"[backend] DIT {time.time()-t0:.1f}s, latents shape {lat_np.shape}")
            t1 = time.time()
            wav_np = _decode_latents(lat_np)
            print(f"[backend] AE {time.time()-t1:.1f}s, decode_fp32={_decode_fp32} overlap={_decode_overlap}")

            if _cancel_event.is_set():
                return None

            target_dur = float(kwargs.get("duration", duration))
            max_samples = int(target_dur * SR)
            print(f"[truncate] mode={'inpaint' if (has_source and has_mask) else 'vary' if has_source else 't2a'} target_dur={target_dur:.2f}s max_samples={max_samples} wav_len={wav_np.shape[-1]}")
            if wav_np.shape[-1] > max_samples:
                wav_np = wav_np[:, :max_samples]

            if has_source and has_mask and audio_mask is not None:
                orig, _ = sf.read(state["audio_path"])
                orig_t = orig.T
                if orig_t.ndim == 1:
                    orig_t = np.stack([orig_t, orig_t], axis=0)
                T = min(orig_t.shape[-1], wav_np.shape[-1], len(audio_mask))
                m = audio_mask[:T].astype(np.float32)
                m2 = np.stack([m, m], axis=0)
                wav_np = wav_np[:, :T]
                orig_t = orig_t[:, :T]
                XF = 256
                m_eased = m2.copy()
                edges = np.where(np.abs(np.diff(m)) > 0)[0]
                for e in edges:
                    lo = max(0, e - XF // 2)
                    hi = min(T, e + XF // 2)
                    w = np.linspace(0, 1, hi - lo)
                    if (m[e] > m[e + 1]) if (e + 1 < T) else False:
                        m_eased[:, lo:hi] = np.minimum(m_eased[:, lo:hi], 1 - w)
                    else:
                        m_eased[:, lo:hi] = np.maximum(m_eased[:, lo:hi], w)
                wav_np = m_eased * orig_t + (1.0 - m_eased) * wav_np

            return wav_np
        finally:
            # Clean up speedup patches
            clear_kv_cache(_dit)
            disable_kv_cache(_dit)
            remove_tome(_dit)
            _unload_loras(loras_list)

    wav_np = await asyncio.to_thread(_run_generate)

    if wav_np is None:
        raise HTTPException(status_code=499, detail="generation cancelled")

    env = persist_audio(wav_np)
    bpm = detect_bpm(wav_np, SR)
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm,
            "seed": seed}


def read_bpm_tag(file_path):
    """Read BPM from audio file metadata (ID3 TBPM, Vorbis, etc). Returns float or None."""
    try:
        import mutagen
        f = mutagen.File(file_path, easy=True)
        if f is None:
            return None
        bpm_str = None
        if hasattr(f, "get"):
            for key in ("bpm", "TBPM", "tempo"):
                val = f.get(key)
                if val:
                    bpm_str = val[0] if isinstance(val, list) else val
                    break
        if not bpm_str:
            f2 = mutagen.File(file_path)
            if f2 and hasattr(f2, "tags") and f2.tags:
                for key in ("TBPM", "TXXX:BPM", "TXXX:bpm"):
                    frame = f2.tags.get(key)
                    if frame:
                        bpm_str = str(frame.text[0]) if hasattr(frame, "text") else str(frame)
                        break
        if bpm_str:
            bpm = float(bpm_str)
            if 20 < bpm < 400:
                return round(bpm, 1)
    except Exception as e:
        print(f"[bpm] tag read failed: {e}")
    return None


def detect_bpm(audio_np, sr=44100):
    """Detect BPM via onset strength autocorrelation. audio_np: (channels, T) or (T,)."""
    from scipy.signal import find_peaks
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    # compute spectral flux onset strength
    hop = 512
    n_fft = 2048
    n_frames = (len(mono) - n_fft) // hop + 1
    if n_frames < 16:
        return 120.0
    onset = np.zeros(n_frames)
    prev_spec = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        frame = mono[i * hop : i * hop + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        diff = spec - prev_spec
        onset[i] = np.sum(np.maximum(0, diff))
        prev_spec = spec
    # autocorrelation in BPM range 40-220
    min_lag = int(60.0 / 220 * sr / hop)
    max_lag = int(60.0 / 40 * sr / hop)
    max_lag = min(max_lag, len(onset) // 2)
    if min_lag >= max_lag:
        return 120.0
    onset_norm = onset - onset.mean()
    corr = np.correlate(onset_norm, onset_norm, mode='full')
    corr = corr[len(onset_norm) - 1:]  # positive lags only
    corr_range = corr[min_lag:max_lag]
    if len(corr_range) == 0:
        return 120.0
    peaks, props = find_peaks(corr_range, distance=int(0.3 * sr / hop))
    if len(peaks) == 0:
        best_lag = min_lag + np.argmax(corr_range)
    else:
        best_idx = np.argmax(corr_range[peaks])
        best_lag = min_lag + peaks[best_idx]
    bpm = 60.0 * sr / hop / best_lag
    # snap to reasonable range, handle octave errors
    if bpm > 180:
        bpm /= 2
    elif bpm < 60:
        bpm *= 2
    return round(bpm, 1)


class TempoBody(BaseModel):
    factor: float = 1.0  # >1 = faster, <1 = slower
    target_bpm: float | None = None


@app.post("/api/tempo")
async def tempo_change(body: TempoBody):
    if state["audio_path"] is None:
        raise HTTPException(400, "no audio loaded")
    _snapshot_current()
    factor = body.factor
    if factor <= 0.1 or factor > 10.0:
        raise HTTPException(400, "factor must be in (0.1, 10.0]")
    if abs(factor - 1.0) < 0.001:
        return {"version": state["version"], "count": 0, "duration": 0}

    src = state["audio_path"]
    dst = str(DATA_DIR / "tempo_out.wav")
    proc = await asyncio.create_subprocess_exec(
        "rubberband", "--fine", "--tempo", str(factor), src, dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(500, f"rubberband failed: {stderr.decode()[:200]}")

    audio, sr_out = sf.read(dst)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    env = persist_audio(audio.T)
    if body.target_bpm and body.target_bpm > 0:
        bpm = round(body.target_bpm, 1)
    else:
        bpm = detect_bpm(audio.T, SR)
    state["bpm"] = bpm
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR, "bpm": bpm}


@app.post("/api/clear")
async def clear():
    state["audio_path"] = None
    state["version"] += 1
    return {"version": state["version"]}


@app.get("/api/state")
async def get_state():
    return {"has_audio": state["audio_path"] is not None, "version": state["version"],
            "model_loaded": sa is not None, "backend": BACKEND, "model": _current_model,
            "bpm": state.get("bpm")}


@app.get("/api/settings")
async def get_settings():
    safe = {**_settings, "first_run": not _SETTINGS_FILE.exists()}
    if safe.get("hf_token"):
        safe["hf_token"] = "hf_****" + safe["hf_token"][-4:]
    return safe

class SettingsBody(BaseModel):
    models_dir: str | None = None
    lora_dir: str | None = None
    lora_train_dir: str | None = None
    embeddings_dir: str | None = None
    sa3_root: str | None = None
    hf_token: str | None = None

@app.post("/api/settings")
async def update_settings(body: SettingsBody):
    changed = {k: v for k, v in body.model_dump().items() if v is not None}
    if changed.get("hf_token", "").startswith("hf_****"):
        del changed["hf_token"]
    _settings.update(changed)
    _save_settings(_settings)
    _apply_settings()
    safe = {**_settings}
    if safe.get("hf_token"):
        safe["hf_token"] = "hf_****" + safe["hf_token"][-4:]
    return safe


import psutil
_proc = psutil.Process()
psutil.cpu_percent(interval=None)

@app.get("/api/speedups")
async def get_speedups():
    """Return available inference speedup options and their current state."""
    return {
        "kv_cache": {
            "available": True,
            "description": "Cache cross-attention K/V projections across diffusion steps",
            "param": "kvCache",
            "type": "bool",
        },
        "tome": {
            "available": True,
            "description": "Token Merging: reduce self-attention sequence length per block",
            "param": "tomeRatio",
            "type": "float",
            "range": [0.0, 0.5],
            "default": 0.0,
        },
    }


def _load_training_losses():
    """Parse loss from PL metrics CSVs in training dirs. Returns {step: loss}."""
    losses = {}
    if not LORA_TRAIN_DIR.exists():
        return losses
    import csv
    for train_dir in LORA_TRAIN_DIR.iterdir():
        if not train_dir.is_dir():
            continue
        logs_dir = train_dir / "checkpoints" / "lightning_logs"
        if not logs_dir.exists():
            continue
        for version_dir in sorted(logs_dir.iterdir(), reverse=True):
            csv_path = version_dir / "metrics.csv"
            if not csv_path.exists():
                continue
            try:
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        step = int(row.get("step", -1))
                        loss = float(row.get("train/loss", 0))
                        losses[(train_dir.name, step)] = round(loss, 4)
            except Exception:
                pass
    return losses

@app.get("/api/loras")
async def list_loras():
    if not LORA_DIR.exists(): return {"dir": str(LORA_DIR), "files": []}
    import re
    losses = _load_training_losses()
    files = []
    for p in sorted(LORA_DIR.iterdir()):
        if not p.is_file() or p.suffix != ".safetensors":
            continue
        entry = {"name": p.name}
        m = re.match(r"(.+?)-step(\d+)\.safetensors$", p.name)
        if m:
            train_name, step = m.group(1), int(m.group(2))
            loss = losses.get((train_name, step - 1)) or losses.get((train_name, step))
            if loss is not None:
                entry["loss"] = loss
                entry["step"] = step
        files.append(entry)
    return {"dir": str(LORA_DIR), "files": files}


# -------- memory tokens --------

MEMTOK_DIR = Path(os.environ.get("SA3_MEMTOK_DIR", str(Path.home() / "models/sa3-memory-tokens")))


def _get_memory_tokens():
    """Get the live memory_tokens parameter, or None."""
    try:
        return sa.model.model.transformer.memory_tokens
    except AttributeError:
        return None


@app.get("/api/memory_tokens")
async def get_memory_tokens():
    mt = _get_memory_tokens()
    if mt is None:
        return {"available": False}
    presets = []
    if MEMTOK_DIR.exists():
        presets = sorted(p.stem for p in MEMTOK_DIR.glob("*.safetensors"))
    return {
        "available": True,
        "count": mt.shape[0],
        "dim": mt.shape[1],
        "strength": _memory_token_strength,
        "presets": presets,
    }


class MemtokBody(BaseModel):
    strength: float | None = None
    preset: str | None = None
    action: str = "set"   # "set" | "save" | "load" | "reset"
    name: str | None = None


@app.post("/api/memory_tokens")
async def set_memory_tokens(body: MemtokBody):
    global _memory_token_strength
    mt = _get_memory_tokens()
    if mt is None:
        raise HTTPException(400, "model has no memory tokens")

    if body.action == "reset":
        if _default_memory_tokens is not None:
            mt.data.copy_(_default_memory_tokens)
        _memory_token_strength = 1.0
        return {"strength": 1.0, "status": "reset"}

    if body.action == "save":
        name = body.name or "custom"
        MEMTOK_DIR.mkdir(parents=True, exist_ok=True)
        from safetensors.torch import save_file
        save_file({"memory_tokens": mt.data.clone().cpu()}, str(MEMTOK_DIR / f"{name}.safetensors"))
        print(f"[memtok] saved {name}")
        return {"status": "saved", "name": name}

    if body.action == "load":
        preset = body.preset or body.name
        if not preset:
            raise HTTPException(400, "preset name required")
        path = MEMTOK_DIR / f"{preset}.safetensors"
        if not path.exists():
            raise HTTPException(404, f"preset not found: {preset}")
        data = load_file(str(path))
        mt.data.copy_(data["memory_tokens"].to(mt.device, mt.dtype))
        _default_memory_tokens.copy_(mt.data)
        _memory_token_strength = 1.0
        print(f"[memtok] loaded {preset}")
        return {"status": "loaded", "preset": preset, "strength": 1.0}

    if body.strength is not None:
        s = max(0.0, min(3.0, body.strength))
        if _default_memory_tokens is not None:
            mt.data.copy_(_default_memory_tokens * s)
        _memory_token_strength = s
        return {"strength": s, "status": "ok"}

    return {"strength": _memory_token_strength, "status": "ok"}


# -------- embeddings (textual inversion) --------

@app.get("/api/embeddings")
async def list_embeddings():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(EMBED_DIR.iterdir()):
        if p.suffix == ".safetensors":
            ckpt_dir = EMBED_DIR / p.stem
            has_ckpts = ckpt_dir.is_dir() and any(ckpt_dir.glob("*.safetensors"))
            files.append({"name": p.stem, "file": p.name, "has_checkpoints": has_ckpts})
    return {"dir": str(EMBED_DIR), "files": files}


@app.get("/api/embeddings/{name}/checkpoints")
async def list_checkpoints(name: str):
    ckpt_dir = EMBED_DIR / name
    if not ckpt_dir.is_dir():
        return {"checkpoints": []}
    items = []
    for p in sorted(ckpt_dir.iterdir()):
        if p.suffix != ".safetensors":
            continue
        parts = p.stem.split("_")
        try:
            step = int(parts[1])
            loss = float(parts[3])
            items.append({"file": p.name, "step": step, "loss": loss})
        except (IndexError, ValueError):
            continue
    return {"checkpoints": items}


class ApplyCheckpointBody(BaseModel):
    file: str


@app.post("/api/embeddings/{name}/apply_checkpoint")
async def apply_checkpoint(name: str, body: ApplyCheckpointBody):
    ckpt_path = EMBED_DIR / name / body.file
    if not ckpt_path.exists():
        raise HTTPException(404, f"checkpoint not found: {body.file}")
    active_path = EMBED_DIR / f"{name}.safetensors"
    shutil.copy2(str(ckpt_path), str(active_path))
    _embed_cache.pop(name, None)
    print(f"[embed] applied checkpoint {body.file} for {name}")
    return {"status": "applied", "name": name, "checkpoint": body.file}


class TrainEmbedBody(BaseModel):
    folder: str
    name: str
    tokens: int = 4
    steps: int = 500
    lr: float = 0.005
    batch_size: int = 0  # 0 = auto


_train_proc = None
_train_last_result = None


@app.get("/api/browse_folder")
async def browse_folder(start: str = "~"):
    """Open a native folder picker dialog and return the selected path."""
    import subprocess
    start_path = str(Path(start).expanduser())
    try:
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--filename={start_path}/"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {"path": result.stdout.strip()}
        return {"path": None}
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["kdialog", "--getexistingdirectory", start_path],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                return {"path": result.stdout.strip()}
            return {"path": None}
        except FileNotFoundError:
            raise HTTPException(status_code=501, detail="No dialog tool found (zenity or kdialog)")


@app.post("/api/train_embedding")
async def train_embedding(body: TrainEmbedBody):
    global _train_proc
    if _train_proc and _train_proc.returncode is None:
        raise HTTPException(409, "training already in progress")

    folder = Path(body.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {folder}")

    output = str(EMBED_DIR / f"{body.name}.safetensors")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # resolve model path
    model_path = LOCAL_MEDIUM
    if _current_model and _current_model != "medium":
        resolved = _resolve_local_path(_current_model)
        if resolved:
            model_path = resolved

    script = str(Path(__file__).resolve().parent / "train_embedding.py")
    cmd = [
        sys.executable, script,
        "--model-path", model_path,
        "--audio-folder", str(folder),
        "--output", output,
        "--tokens", str(body.tokens),
        "--steps", str(body.steps),
        "--lr", str(body.lr),
        "--device", DEVICE,
        "--checkpoint-dir", str(EMBED_DIR / body.name),
        "--checkpoint-every", "50",
        "--batch-size", str(body.batch_size),
    ]
    if _use_fp16:
        cmd.append("--fp16")

    _train_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return {"status": "started", "name": body.name, "output": output}


@app.get("/api/train_embedding/status")
async def train_status():
    global _train_proc, _train_last_result
    if _train_proc is None:
        if _train_last_result is not None:
            result = _train_last_result
            _train_last_result = None
            return result
        return {"status": "idle"}
    if _train_proc.returncode is not None:
        stdout = (await _train_proc.stdout.read()).decode()
        stderr = (await _train_proc.stderr.read()).decode()
        lines = [l for l in stdout.strip().split("\n") if l.strip()]
        last = {}
        for l in reversed(lines):
            try:
                last = json.loads(l)
                break
            except Exception:
                continue
        ok = _train_proc.returncode == 0
        _train_proc = None
        _embed_cache.clear()
        if not ok:
            print(f"[train] FAILED (rc={1 if not ok else 0})")
            if stderr:
                print(f"[train] stderr: {stderr[:2000]}")
            if stdout:
                print(f"[train] stdout: {stdout[:2000]}")
        _train_last_result = {"status": "done" if ok else "error", "result": last,
                "error": stderr[:2000] if not ok else None,
                "stdout": "\n".join(lines[-10:]) if lines else None}
        return _train_last_result
    # still running — try to read latest progress line
    try:
        line = await asyncio.wait_for(_train_proc.stdout.readline(), timeout=0.1)
        if line:
            try:
                return {"status": "running", "progress": json.loads(line.decode())}
            except Exception:
                pass
    except asyncio.TimeoutError:
        pass
    return {"status": "running"}


# -------- LoRA training --------

class TrainLoraBody(BaseModel):
    folder: str
    name: str
    caption: str = ""
    rank: int = 16
    adapter_type: str = "dora-rows"
    steps: int = 1000
    lr: float = 1e-4
    batch_size: int = 1
    checkpoint_every: int = 100
    exclude: list[str] | None = None
    use_compile: bool = False
    train_conditioner: bool = False
    dist_shift: bool = True
    grad_checkpoint: bool = True

class PreEncodeBody(BaseModel):
    folder: str
    name: str
    caption: str = ""

_lora_train_proc = None
_lora_train_last_result = None
_pre_encode_proc = None
_pre_encode_last_result = None

@app.post("/api/pre_encode")
async def start_pre_encode(body: PreEncodeBody):
    global _pre_encode_proc
    if _pre_encode_proc and _pre_encode_proc.returncode is None:
        raise HTTPException(409, "Pre-encoding already in progress")

    folder = Path(body.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {folder}")

    output_dir = str(LORA_TRAIN_DIR / body.name)
    LORA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    ae_model = "same-l"
    if _current_model and "small" in _current_model:
        ae_model = "same-s"

    env = os.environ.copy()
    env["SA3_ROOT"] = _settings["sa3_root"]
    env["SA3_MODELS_DIR"] = str(MODELS_DIR)
    if _settings.get("hf_token"):
        env["HF_TOKEN"] = _settings["hf_token"]

    if HAS_MLX:
        # MLX pre-encoding path (Apple Silicon)
        script = str(Path(__file__).resolve().parent.parent / "mlx_sa3" / "pre_encode_mlx.py")
        enc_dir = str(Path(output_dir) / "_encoded")
        cmd = [
            sys.executable, script,
            "--audio-dir", str(folder),
            "--output-dir", enc_dir,
        ]
    else:
        # PyTorch pre-encoding path (CUDA)
        script = str(Path(__file__).resolve().parent / "pre_encode.py")
        cmd = [
            sys.executable, script,
            "--audio-folder", str(folder),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--ae-model", ae_model,
        ]

    _pre_encode_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    return {"status": "started", "name": body.name, "output_dir": output_dir}


@app.get("/api/pre_encode/status")
async def pre_encode_status():
    global _pre_encode_proc, _pre_encode_last_result
    if _pre_encode_proc is None:
        if _pre_encode_last_result is not None:
            result = _pre_encode_last_result
            _pre_encode_last_result = None
            return result
        return {"status": "idle"}
    if _pre_encode_proc.returncode is not None:
        stdout = (await _pre_encode_proc.stdout.read()).decode()
        stderr = (await _pre_encode_proc.stderr.read()).decode()
        lines = [l for l in stdout.strip().split("\n") if l.strip()]
        last = {}
        for l in reversed(lines):
            try:
                last = json.loads(l)
                break
            except Exception:
                continue
        rc = _pre_encode_proc.returncode
        ok = rc == 0
        _pre_encode_proc = None
        if not ok:
            print(f"[pre_encode] FAILED (rc={rc})")
            if stderr:
                print(f"[pre_encode] stderr: {stderr[:2000]}")
        _pre_encode_last_result = {
            "status": "done" if ok else "error",
            "result": last,
            "error": stderr[:2000] if not ok else None,
        }
        return _pre_encode_last_result
    try:
        line = await asyncio.wait_for(_pre_encode_proc.stdout.readline(), timeout=0.1)
        if line:
            try:
                return {"status": "running", "progress": json.loads(line.decode())}
            except Exception:
                pass
    except asyncio.TimeoutError:
        pass
    return {"status": "running"}


@app.get("/api/lora_training/{name}/has_encoded")
async def check_encoded(name: str):
    enc_dir = LORA_TRAIN_DIR / name / "_encoded"
    if enc_dir.is_dir():
        n = len(list(enc_dir.glob("*.npy")))
        return {"has_encoded": n > 0, "latents": n}
    return {"has_encoded": False, "latents": 0}


@app.post("/api/train_lora")
async def start_lora_training(body: TrainLoraBody):
    global _lora_train_proc, _training_unloaded_model
    if _lora_train_proc and _lora_train_proc.returncode is None:
        raise HTTPException(409, "LoRA training already in progress")

    folder = Path(body.folder).expanduser().resolve()
    if not folder.is_dir():
        raise HTTPException(400, f"folder not found: {folder}")

    output_dir = str(LORA_TRAIN_DIR / body.name)
    LORA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "medium-base"
    if _current_model and "small" in _current_model:
        model_name = _current_model

    # unload inference model to free VRAM for training
    _training_unloaded_model = _current_model
    _unload_model()

    # resolve auto batch size from free VRAM (~3GB per batch element for full LoRA training)
    batch_size = body.batch_size
    if batch_size <= 0 and DEVICE == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        batch_size = max(1, min(8, int(free_gb / 3.0)))
        print(f"[lora_train] auto batch: {free_gb:.1f}GB free → batch_size={batch_size}")
    elif batch_size <= 0:
        batch_size = 1

    # total_steps ÷ batch = optimizer steps (keeps wall time constant regardless of batch)
    optimizer_steps = max(1, body.steps // batch_size)
    checkpoint_every = max(1, body.checkpoint_every // batch_size)
    print(f"[lora_train] total_steps={body.steps} ÷ batch={batch_size} → optimizer_steps={optimizer_steps}")

    env = os.environ.copy()
    env["SA3_ROOT"] = _settings["sa3_root"]
    env["SA3_MODELS_DIR"] = str(MODELS_DIR)
    env["SA3_LORA_DIR"] = str(LORA_DIR)
    if _settings.get("hf_token"):
        env["HF_TOKEN"] = _settings["hf_token"]

    # Check for pre-encoded latents
    enc_dir = Path(output_dir) / "_encoded"
    has_encoded = enc_dir.is_dir() and list(enc_dir.glob("*.npy"))

    use_mlx_train = HAS_MLX
    if use_mlx_train:
        # MLX training path (Apple Silicon)
        script = str(Path(__file__).resolve().parent.parent / "mlx_sa3" / "train_lora_mlx.py")
        if not has_encoded:
            raise HTTPException(400,
                "MLX training requires pre-encoded latents. "
                "Run pre-encoding first or transfer pre-encoded data from a CUDA machine.")
        cmd = [
            sys.executable, script,
            "--model-name", model_name,
            "--encoded-dir", str(enc_dir),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--rank", str(body.rank),
            "--adapter-type", body.adapter_type,
            "--steps", str(optimizer_steps),
            "--lr", str(body.lr),
            "--batch-size", str(batch_size),
            "--checkpoint-every", str(checkpoint_every),
        ]
        if body.dist_shift:
            cmd.append("--dist-shift")
        if body.grad_checkpoint:
            cmd.append("--grad-checkpoint")
        if body.train_conditioner:
            cmd.append("--train-conditioner")
        if body.exclude:
            cmd.extend(["--exclude"] + body.exclude)
        print(f"[lora_train] using MLX training backend ({body.adapter_type})")
    else:
        # PyTorch training path (CUDA)
        script = str(Path(__file__).resolve().parent / "train_lora.py")
        cmd = [
            sys.executable, script,
            "--model-name", model_name,
            "--audio-folder", str(folder),
            "--output-dir", output_dir,
            "--caption", body.caption or body.name,
            "--rank", str(body.rank),
            "--adapter-type", body.adapter_type,
            "--steps", str(optimizer_steps),
            "--lr", str(body.lr),
            "--batch-size", str(batch_size),
            "--checkpoint-every", str(checkpoint_every),
        ]
        if body.exclude:
            cmd.extend(["--exclude"] + body.exclude)
        if body.use_compile:
            cmd.append("--compile")

    _lora_train_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    return {"status": "started", "name": body.name, "output_dir": output_dir,
            "batch_size": batch_size, "optimizer_steps": optimizer_steps,
            "backend": "mlx" if use_mlx_train else "cuda"}


@app.get("/api/train_lora/status")
async def lora_train_status():
    global _lora_train_proc, _lora_train_last_result, _training_unloaded_model
    if _lora_train_proc is None:
        if _lora_train_last_result is not None:
            result = _lora_train_last_result
            _lora_train_last_result = None
            return result
        return {"status": "idle"}
    if _lora_train_proc.returncode is not None:
        stdout = (await _lora_train_proc.stdout.read()).decode()
        stderr = ""
        lines = [l for l in stdout.strip().split("\n") if l.strip()]
        last = {}
        for l in reversed(lines):
            try:
                last = json.loads(l)
                break
            except Exception:
                continue
        rc = _lora_train_proc.returncode
        ok = rc == 0
        _lora_train_proc = None
        if not ok:
            print(f"[lora_train] FAILED (rc={rc})")
            if stderr:
                print(f"[lora_train] stderr: {stderr[:2000]}")
            if stdout:
                err_lines = [l for l in lines if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
                print(f"[lora_train] stdout errors: {err_lines[-3:] if err_lines else lines[-5:]}")
        error_detail = stderr[:2000] if stderr else "\n".join(lines[-10:]) if not ok else None
        _lora_train_last_result = {
            "status": "done" if ok else "error",
            "result": last,
            "error": error_detail,
        }
        # reload the inference model that was unloaded for training
        if _training_unloaded_model:
            reload_name = _training_unloaded_model
            _training_unloaded_model = None
            print(f"[backend] reloading {reload_name} after training...")
            try:
                _load_model(reload_name)
                _lora_train_last_result["model_reloaded"] = True
            except Exception as e:
                print(f"[backend] reload failed: {e}")
                _lora_train_last_result["model_reloaded"] = False
                _lora_train_last_result["reload_error"] = str(e)
        return _lora_train_last_result
    # still running
    try:
        line = await asyncio.wait_for(_lora_train_proc.stdout.readline(), timeout=0.1)
        if line:
            try:
                return {"status": "running", "progress": json.loads(line.decode())}
            except Exception:
                pass
    except asyncio.TimeoutError:
        pass
    return {"status": "running"}


VALID_MODELS = {"medium", "medium-base", "small-music", "small-sfx"}


class ModelBody(BaseModel):
    model: str


@app.post("/api/model")
async def switch_model(body: ModelBody):
    name = body.model
    if name not in VALID_MODELS:
        raise HTTPException(400, f"unknown model: {name}")
    if name == _current_model:
        return {"model": name, "status": "already_loaded"}

    try:
        await asyncio.to_thread(_load_model, name)
    except Exception as e:
        raise HTTPException(500, f"model load failed: {e}")

    return {"model": name, "status": "loaded", "backend": BACKEND,
            "mlx_ae": _use_mlx_ae}


class PrecisionBody(BaseModel):
    precision: str


@app.post("/api/precision")
async def set_precision(body: PrecisionBody):
    global _use_fp16
    want_fp16 = body.precision == "fp16"
    if want_fp16 == _use_fp16:
        return {"precision": "fp16" if _use_fp16 else "fp32"}
    if DEVICE != "cuda":
        raise HTTPException(400, "precision switching requires CUDA")
    if want_fp16:
        sa.model.half()
        sa.model_half = True
    else:
        sa.model.float()
        sa.model_half = False
    _use_fp16 = want_fp16
    torch.cuda.empty_cache()
    print(f"[backend] precision switched to {'fp16' if _use_fp16 else 'fp32'}")
    return {"precision": "fp16" if _use_fp16 else "fp32"}


class DecodeSettingsBody(BaseModel):
    decode_fp32: bool | None = None
    decode_overlap: int | None = None


@app.get("/api/decode_settings")
async def get_decode_settings():
    return {"decode_fp32": _decode_fp32, "decode_overlap": _decode_overlap}


@app.post("/api/decode_settings")
async def set_decode_settings(body: DecodeSettingsBody):
    global _decode_fp32, _decode_overlap
    if body.decode_fp32 is not None:
        _decode_fp32 = body.decode_fp32
    if body.decode_overlap is not None:
        _decode_overlap = max(0, min(128, body.decode_overlap))
    print(f"[backend] decode settings: fp32={_decode_fp32} overlap={_decode_overlap}")
    return {"decode_fp32": _decode_fp32, "decode_overlap": _decode_overlap}


@app.get("/api/stats")
async def get_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / 1e9
    ram_total_gb = vm.total / 1e9
    gpu_alloc_gb = 0.0
    try:
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / 1e9
        elif DEVICE == "mps" and hasattr(torch, "mps"):
            gpu_alloc_gb = torch.mps.current_allocated_memory() / 1e9
    except Exception: pass
    return {
        "cpu": round(cpu, 1),
        "ram_used": round(ram_used_gb, 1),
        "ram_total": round(ram_total_gb, 1),
        "gpu_alloc": round(gpu_alloc_gb, 2),
        "precision": "fp16" if _use_fp16 else "fp32",
        "backend": BACKEND,
        "model": _current_model,
        "model_loaded": sa is not None,
    }


@app.get("/api/audio")
async def get_audio():
    if not state["audio_path"]:
        raise HTTPException(404, "no audio")
    return FileResponse(state["audio_path"], media_type="audio/wav")


@app.get("/api/envelope.json")
async def get_env():
    p = DATA_DIR / "envelope.json"
    if not p.exists():
        return {"count": 0, "data": [], "downsample": DOWNSAMPLE, "sr": SR}
    return FileResponse(p, media_type="application/json")


@app.get("/api/spec.png")
async def get_spec():
    p = DATA_DIR / "current_spec.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/overview.png")
async def get_overview():
    p = DATA_DIR / "current_overview.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/noise_spec.png")
async def get_noise_spec():
    p = DATA_DIR / "noise_spec.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


render_noise_spec_once()
print(f"[backend] ready (backend={BACKEND}, device={DEVICE})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5174, log_level="info")
