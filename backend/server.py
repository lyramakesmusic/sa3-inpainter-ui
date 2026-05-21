"""SA3 Inpainter backend. FastAPI on :5174.

Loads the SA3 medium model once at startup (~30s), exposes JSON API for the
Svelte frontend.

CUDA port: uses SA3's built-in torch autoencoder instead of the MLX AE.
"""
import asyncio
import os, sys, json, time, threading, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3 import StableAudioModel
from safetensors.torch import load_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MEDIUM = os.environ.get("SA3_MODEL_DIR", str(Path.home() / "models/stable-audio-3-medium"))
CKPT = f"{LOCAL_MEDIUM}/model.safetensors"
CFG  = f"{LOCAL_MEDIUM}/model_config.json"
DATA_DIR = Path("/tmp/sa3-inpainter"); DATA_DIR.mkdir(exist_ok=True)
SR = 44100
DOWNSAMPLE = 4096
BANDS = [(0, 250), (250, 2500), (2500, 22050)]

print(f"[backend] loading sa3 medium on {DEVICE}...")
_cfg = json.load(open(CFG))
for c in _cfg["model"]["conditioning"]["configs"]:
    if c["type"] == "t5gemma":
        c["config"]["repo_id"] = LOCAL_MEDIUM
_model = create_diffusion_cond_from_config(_cfg)
_model.load_state_dict(load_file(CKPT), strict=False)
_model.eval().requires_grad_(False).to(DEVICE)

# fp16 on CUDA if SA3_FP16=1 (or default True on CUDA)
_use_fp16 = os.environ.get("SA3_FP16", "1" if DEVICE == "cuda" else "0") == "1"
if _use_fp16 and DEVICE == "cuda":
    _model = _model.half()
    print("[backend] fp16 enabled")

sa = StableAudioModel(_model, _cfg, device=DEVICE, model_half=_use_fp16)
print("[backend] model loaded")

# cancellation event — set by POST /api/cancel, cleared before each generate
_cancel_event = threading.Event()


def render_noise_spec_once():
    """Decode 30s of random latents into a noise spectrogram for the slider preview overlay."""
    out_path = DATA_DIR / "noise_spec.png"
    if out_path.exists(): return
    T_lat = int(30 * SR / DOWNSAMPLE) + 1
    rng = np.random.default_rng(7)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    lat_t = torch.from_numpy(lat).to(DEVICE)
    if _use_fp16 and DEVICE == "cuda":
        lat_t = lat_t.half()
    with torch.inference_mode():
        wav_t = sa.same.decode(lat_t)
    wav_np = wav_t.float().cpu().numpy()[0]
    render_spec_png(wav_np, out_path)

state = {"audio_path": None, "version": 0}
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
    """audio_np: (2, T) float in [-1, 1].

    Writes wav + envelope.json synchronously (fast path), then fires background
    threads for spec.png and overview.png.  Returns the envelope immediately.
    """
    p = DATA_DIR / "current.wav"
    sf.write(p, audio_np.T, SR)
    state["audio_path"] = str(p)
    state["version"] += 1
    env = compute_envelope(audio_np)
    with open(DATA_DIR / "envelope.json", "w") as fh:
        json.dump(env, fh)
    # background PNG renders — don't block the response
    threading.Thread(target=render_spec_png, args=(audio_np, DATA_DIR / "current_spec.png"), daemon=True).start()
    threading.Thread(target=render_overview_png, args=(audio_np, DATA_DIR / "current_overview.png"), daemon=True).start()
    return env


# -------- LoRA helpers --------

_loaded_lora_name: str | None = None  # name of the currently loaded LoRA (None = no LoRA loaded)

def _apply_loras(loras: list[dict]) -> None:
    """Load LoRA weights using SA3's native API.

    SA3 supports one LoRA at a time — loading a new one replaces the previous.
    We track which LoRA is loaded to avoid redundant load_lora() calls.
    """
    global _loaded_lora_name
    if not loras:
        return
    # SA3 only supports a single LoRA; use the first valid entry
    for entry in loras:
        name = entry["name"]
        strength = float(entry.get("strength", 1.0))
        lora_path = LORA_DIR / name
        if not lora_path.exists():
            print(f"[lora] not found: {lora_path}, skipping")
            continue
        if _loaded_lora_name != name:
            sa.load_lora(str(lora_path))
            _loaded_lora_name = name
            print(f"[lora] loaded {name}")
        sa.set_lora_strength(strength)
        print(f"[lora] strength {name} @ {strength}")
        return  # only one LoRA at a time


def _unload_loras(loras: list[dict]) -> None:
    """Deactivate LoRA by zeroing strength. Weights stay loaded but inactive."""
    if _loaded_lora_name is None:
        return
    sa.set_lora_strength(0.0)
    print(f"[lora] deactivated {_loaded_lora_name}")


# -------- endpoints --------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = DATA_DIR / "upload.wav"
    with open(raw, "wb") as f: f.write(await file.read())
    audio, sr = sf.read(raw)
    if audio.ndim == 1: audio = np.stack([audio, audio], axis=-1)
    if sr != SR:
        import torchaudio
        a = torch.from_numpy(audio.T).float()
        a = torchaudio.transforms.Resample(sr, SR)(a)
        audio = a.numpy().T
    env = persist_audio(audio.T)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR}


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


@app.post("/api/generate")
async def generate(body: GenBody):
    _cancel_event.clear()

    s = body.settings
    steps = int(s.get("steps", 8))
    cfg = float(s.get("cfg", 1.0))
    seed = int(s.get("seed", 42))
    noise = float(s.get("noise", 1.0))
    duration = float(s.get("duration", 30.0))

    has_source = state["audio_path"] is not None
    has_mask = any(body.mask) if body.mask else False
    n_regen = sum(body.mask) if body.mask else 0
    print(f"[generate] source={has_source} mask_len={len(body.mask) if body.mask else 0} regen_latents={n_regen} mode={('inpaint' if has_source and has_mask else 'vary' if has_source else 't2a')}")

    neg_prompt = body.negative_prompt or None
    kwargs = dict(prompt=body.prompt, negative_prompt=neg_prompt, steps=steps,
                  cfg_scale=cfg, seed=seed, return_latents=False, chunked_decode=True)

    # ---- build audio_mask for inpaint (needed in stitch step too) ----
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

    loras_list = [l.model_dump() for l in body.loras]

    def _run_generate():
        """Heavy computation: DIT + AE + stitch. Runs in a thread pool."""
        _apply_loras(loras_list)
        try:
            if _cancel_event.is_set():
                return None

            t0 = time.time()
            result = sa.generate(**kwargs)
            wav_np = result.detach().to(torch.float32).cpu().numpy()[0]
            print(f"[backend] DIT+AE {time.time()-t0:.1f}s, wav shape {wav_np.shape}")

            if _cancel_event.is_set():
                return None

            # truncate
            target_dur = float(kwargs.get("duration", duration))
            max_samples = int(target_dur * SR)
            print(f"[truncate] mode={'inpaint' if (has_source and has_mask) else 'vary' if has_source else 't2a'} target_dur={target_dur:.2f}s max_samples={max_samples} wav_len={wav_np.shape[-1]}")
            if wav_np.shape[-1] > max_samples:
                wav_np = wav_np[:, :max_samples]

            # crossfade stitch
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
            _unload_loras(loras_list)

    wav_np = await asyncio.to_thread(_run_generate)

    if wav_np is None:
        raise HTTPException(status_code=499, detail="generation cancelled")

    env = persist_audio(wav_np)
    return {"version": state["version"], "count": env["count"],
            "duration": env["count"] * DOWNSAMPLE / SR}


@app.post("/api/clear")
async def clear():
    state["audio_path"] = None
    state["version"] += 1
    return {"version": state["version"]}


@app.get("/api/state")
async def get_state():
    return {"has_audio": state["audio_path"] is not None, "version": state["version"], "model_loaded": True}


import psutil
_proc = psutil.Process()
psutil.cpu_percent(interval=None)
LORA_DIR = Path(os.environ.get("SA3_LORA_DIR", str(Path.home() / "loras")))

@app.get("/api/loras")
async def list_loras():
    if not LORA_DIR.exists(): return {"dir": str(LORA_DIR), "files": []}
    files = sorted(p.name for p in LORA_DIR.iterdir() if p.is_file() and p.suffix == ".safetensors")
    return {"dir": str(LORA_DIR), "files": files}


class PrecisionBody(BaseModel):
    precision: str  # "fp16" or "fp32"


@app.post("/api/precision")
async def set_precision(body: PrecisionBody):
    global _use_fp16
    want_fp16 = body.precision == "fp16"
    if want_fp16 == _use_fp16:
        return {"precision": "fp16" if _use_fp16 else "fp32"}
    if DEVICE != "cuda":
        raise HTTPException(400, "precision switching requires CUDA")
    if want_fp16:
        _model.half()
        sa.model_half = True
    else:
        _model.float()
        sa.model_half = False
    _use_fp16 = want_fp16
    torch.cuda.empty_cache()
    print(f"[backend] precision switched to {'fp16' if _use_fp16 else 'fp32'}")
    return {"precision": "fp16" if _use_fp16 else "fp32"}


@app.get("/api/stats")
async def get_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / 1e9
    ram_total_gb = vm.total / 1e9
    gpu_alloc_gb = 0.0
    try:
        if torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / 1e9
    except Exception: pass
    return {
        "cpu": round(cpu, 1),
        "ram_used": round(ram_used_gb, 1),
        "ram_total": round(ram_total_gb, 1),
        "gpu_alloc": round(gpu_alloc_gb, 2),
        "precision": "fp16" if _use_fp16 else "fp32",
        "model_loaded": True,
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
print("[backend] ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5174, log_level="info")
