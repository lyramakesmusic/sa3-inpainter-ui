"""SA3 Inpainter backend. FastAPI on :5174.

Loads the SA3 medium model once at startup (~30s), exposes JSON API for the
Svelte frontend.
"""
import os, sys, json, time, warnings, asyncio, threading, platform
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for mlx_sa3

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3 import StableAudioModel
from stable_audio_3.loading_utils import load_autoencoder
from safetensors.torch import load_file

DEFAULT_MODEL_DIR = str(Path.home() / "Projects/stable-audio-3/models/stable-audio-3-medium")
LOCAL_MEDIUM = os.environ.get("SA3_MODEL_DIR", DEFAULT_MODEL_DIR)
CKPT = f"{LOCAL_MEDIUM}/model.safetensors"
CFG  = f"{LOCAL_MEDIUM}/model_config.json"
DATA_DIR = Path("/tmp/sa3-inpainter"); DATA_DIR.mkdir(exist_ok=True)
SR = 44100
DOWNSAMPLE = 4096
BANDS = [(0, 250), (250, 2500), (2500, 22050)]


def pick_device():
    requested = os.environ.get("SA3_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        # ROCm PyTorch also reports AMD GPUs through the cuda device API.
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
MODEL_HALF = os.environ.get("SA3_MODEL_HALF", "0") == "1"
AE_BACKEND = os.environ.get("SA3_AE_BACKEND", "auto").lower()
if AE_BACKEND == "auto":
    AE_BACKEND = "mlx" if DEVICE == "mps" and platform.system() == "Darwin" else "torch"
AE_DEVICE = os.environ.get("SA3_AE_DEVICE")
if AE_DEVICE is None:
    AE_DEVICE = DEVICE

print(f"[backend] loading sa3 medium on {DEVICE}...")
_cfg = json.load(open(CFG))
for c in _cfg["model"]["conditioning"]["configs"]:
    if c["type"] == "t5gemma":
        c["config"]["repo_id"] = LOCAL_MEDIUM
_model = create_diffusion_cond_from_config(_cfg)
_model.load_state_dict(load_file(CKPT), strict=False)
_model.eval().requires_grad_(False).to(DEVICE)
sa = StableAudioModel(_model, _cfg, device=DEVICE, model_half=MODEL_HALF)

mlx_ae = None
torch_ae = None
if AE_BACKEND == "mlx":
    print("[backend] loading MLX AE...")
    import mlx.core as mx
    from mlx_sa3.ae import SA3MediumAE, decode_chunked
    from mlx_sa3.weights import load_ae_weights
    mlx_ae = SA3MediumAE()
    load_ae_weights(mlx_ae, CKPT)
elif AE_BACKEND == "torch":
    print(f"[backend] loading torch AE on {AE_DEVICE}...")
    torch_ae = load_autoencoder(CFG, CKPT, device=AE_DEVICE)
    torch_ae.eval().requires_grad_(False)
    try:
        torch_ae.bottleneck.noise_regularize = False
    except Exception:
        pass
else:
    raise RuntimeError(f"unsupported SA3_AE_BACKEND={AE_BACKEND!r}; use auto, mlx, or torch")


def decode_latents(lat_np):
    if AE_BACKEND == "mlx":
        lat = mx.array(lat_np)
        if lat_np.shape[-1] > 128:
            wav = decode_chunked(mlx_ae, lat, chunk_size=128, overlap=32)
        else:
            wav = mlx_ae.decode(lat)
        mx.eval(wav)
        return np.array(wav)[0]

    with torch.inference_mode():
        lat = torch.from_numpy(lat_np).to(AE_DEVICE)
        wav = torch_ae.decode(lat)
    return wav.detach().to(torch.float32).cpu().numpy()[0]


# defined after helpers below; called at bottom of file
def render_noise_spec_once():
    """Decode 30s of random latents into a noise spectrogram for the slider preview overlay."""
    out_path = DATA_DIR / "noise_spec.png"
    if out_path.exists(): return
    T_lat = int(30 * SR / DOWNSAMPLE) + 1
    rng = np.random.default_rng(7)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    render_spec_png(decode_latents(lat), out_path)

state = {"audio_path": None, "version": 0}
_gen_lock = threading.Lock()
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
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    mono = audio_np.mean(axis=0) if audio_np.ndim == 2 else audio_np
    # one STFT column per latent so spec time-axis quantizes to the same grid
    # the waveform uses
    n_fft = 8192
    hop = DOWNSAMPLE
    f, t, Z = stft(mono, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    P_db = 10.0 * np.log10(P + 1e-12)
    P_db = np.clip(P_db, -55, P_db.max()); P_db -= P_db.min(); P_db /= P_db.max() + 1e-12
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
    import matplotlib.pyplot as plt
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


def soft_limit(x, ceiling=0.97, knee=0.85):
    """Soft saturator: linear below `knee`, smoothly compressed up to `ceiling`.
    Cheap and clip-free even when the model returns peaks above 1.0."""
    s = np.sign(x)
    a = np.abs(x)
    out = np.where(
        a <= knee,
        a,
        knee + (ceiling - knee) * np.tanh((a - knee) / (ceiling - knee)),
    )
    return s * out


def persist_audio(audio_np):
    """audio_np: (2, T) float in [-1, 1]. Writes wav + envelope.json + spec.png + overview.png.
    Caller is responsible for limiting/scaling — we don't touch the levels here so
    inpaint-preserved regions stay bit-exact with the source."""
    p = DATA_DIR / "current.wav"
    sf.write(p, audio_np.T, SR)
    state["audio_path"] = str(p)
    state["version"] += 1
    env = compute_envelope(audio_np)
    with open(DATA_DIR / "envelope.json", "w") as f: json.dump(env, f)
    render_spec_png(audio_np, DATA_DIR / "current_spec.png")
    render_overview_png(audio_np, DATA_DIR / "current_overview.png")
    return env


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


class GenBody(BaseModel):
    prompt: str = ""
    mask: list[int] = []
    settings: dict = {}


@app.post("/api/generate")
async def generate(body: GenBody):
    # only one gen at a time; reject overlapping requests (lets the client retry).
    # the gen itself runs in a thread so the event loop stays responsive for stats/state/etc.
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "generation in progress")
    try:
        return await asyncio.to_thread(_run_generate, body)
    finally:
        _gen_lock.release()


def _run_generate(body):
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

    kwargs = dict(prompt=body.prompt, steps=steps, cfg_scale=cfg, seed=seed, return_latents=True)
    if has_source and has_mask:
        audio, _ = sf.read(state["audio_path"])
        audio_t = torch.from_numpy(audio.T).float().to(DEVICE)   # (channels, T) — no batch dim
        # align mask length to the actual latent count of the loaded audio.
        # frontend's mask may be stale (different count from earlier session state).
        actual_lat = audio.shape[0] // DOWNSAMPLE
        mask_lat = np.asarray(body.mask, dtype=np.float32)
        if len(mask_lat) > actual_lat:
            mask_lat = mask_lat[:actual_lat]
        elif len(mask_lat) < actual_lat:
            mask_lat = np.pad(mask_lat, (0, actual_lat - len(mask_lat)), constant_values=0)
        # mask: 1=regen, 0=preserve. sa3 convention: 1=preserve, 0=regen. invert.
        inv = 1.0 - mask_lat
        audio_mask = np.repeat(inv, DOWNSAMPLE)
        audio_mask = audio_mask[:audio.shape[0]]
        print(f"[inpaint] mask aligned: {len(mask_lat)} latents, {int(mask_lat.sum())} regen, {audio.shape[0]} samples")
        kwargs["duration"] = audio.shape[0] / SR
        kwargs["sample_size"] = audio.shape[0]   # cap output to actual source length, not sa3's default 120s
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

    t0 = time.time()
    latents = sa.generate(**kwargs)
    lat_np = latents.detach().to(torch.float32).cpu().numpy()
    print(f"[backend] DIT {time.time()-t0:.1f}s, latents shape {lat_np.shape}")
    t1 = time.time()
    wav_np = decode_latents(lat_np)
    print(f"[backend] AE {time.time()-t1:.1f}s")
    # (2, T) — raw, no limiter (visual cap handled in frontend)

    # sa3 pads internally to a power-of-2 latent count → output is longer than requested.
    target_dur = float(kwargs.get("duration", duration))
    max_samples = int(target_dur * SR)
    print(f"[truncate] mode={'inpaint' if (has_source and has_mask) else 'vary' if has_source else 't2a'} target_dur={target_dur:.2f}s max_samples={max_samples} wav_len={wav_np.shape[-1]}")
    if wav_np.shape[-1] > max_samples:
        wav_np = wav_np[:, :max_samples]

    # inpaint mode: stitch — preserve the original audio at unmasked sample positions
    # so unmasked regions are bit-exact, not lossily reconstructed by the AE+DIT
    if has_source and has_mask:
        orig, _ = sf.read(state["audio_path"])
        orig_t = orig.T   # (2, T)
        if orig_t.ndim == 1:
            orig_t = np.stack([orig_t, orig_t], axis=0)
        # align to the smallest of (orig audio, generated wav, mask) so shapes match exactly
        T = min(orig_t.shape[-1], wav_np.shape[-1], len(audio_mask))
        m = audio_mask[:T].astype(np.float32)        # 1=preserve, 0=regen
        m2 = np.stack([m, m], axis=0)
        wav_np = wav_np[:, :T]
        orig_t = orig_t[:, :T]
        # short cosine crossfade across boundaries to avoid clicks
        # (boundaries detected as positions where mask transitions)
        XF = 256   # samples
        m_eased = m2.copy()
        edges = np.where(np.abs(np.diff(m)) > 0)[0]
        for e in edges:
            lo = max(0, e - XF // 2)
            hi = min(T, e + XF // 2)
            w = np.linspace(0, 1, hi - lo)
            if m[e] > m[e + 1] if e + 1 < T else False:
                # going from preserve(1) to regen(0): fade preserve out
                m_eased[:, lo:hi] = np.minimum(m_eased[:, lo:hi], 1 - w)
            else:
                m_eased[:, lo:hi] = np.maximum(m_eased[:, lo:hi], w)
        wav_np = m_eased * orig_t + (1.0 - m_eased) * wav_np

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
psutil.cpu_percent(interval=None)  # prime
LORA_DIR = Path(os.environ.get("SA3_LORA_DIR", str(Path.home() / "loras")))

@app.get("/api/loras")
async def list_loras():
    if not LORA_DIR.exists(): return {"dir": str(LORA_DIR), "files": []}
    files = sorted(p.name for p in LORA_DIR.iterdir() if p.is_file() and p.suffix == ".safetensors")
    return {"dir": str(LORA_DIR), "files": files}


@app.get("/api/stats")
async def get_stats():
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / 1e9
    ram_total_gb = vm.total / 1e9
    device_alloc_gb = 0.0
    try:
        if DEVICE == "mps" and torch.backends.mps.is_available():
            device_alloc_gb = torch.mps.current_allocated_memory() / 1e9
        elif DEVICE == "cuda" and torch.cuda.is_available():
            device_alloc_gb = torch.cuda.memory_allocated() / 1e9
    except Exception: pass
    return {
        "cpu": round(cpu, 1),
        "ram_used": round(ram_used_gb, 1),
        "ram_total": round(ram_total_gb, 1),
        "mps_alloc": round(device_alloc_gb, 2),
        "device": DEVICE,
        "ae_backend": AE_BACKEND,
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
