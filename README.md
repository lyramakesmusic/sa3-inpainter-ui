# sa3-inpainter-ui

Browser UI for Stable Audio 3 medium — inpainting / vary / text-to-audio. MLX-backed SAME-L decoder so it runs on Apple Silicon without flash-attn.

Upstream: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) · [stabilityai/stable-audio-3-medium on HF](https://huggingface.co/stabilityai/stable-audio-3-medium)

![interface](interface.png)

## Install

```bash
# python deps
uv sync

# frontend deps
cd webui && npm install && cd ..
```

You'll also need the SA3 medium weights from HuggingFace at `~/Projects/stable-audio-3/models/stable-audio-3-medium/` (or edit the `LOCAL_MEDIUM` path in `backend/server.py`).

## Run

```bash
# backend on :5174 — ~30s to load the model
uv run python backend/server.py

# frontend on :5173 — Vite proxies /api → :5174
cd webui && npm run dev
```

Open http://localhost:5173.

LoRA library is read from `$SA3_LORA_DIR` (default `~/loras`).

---

## For an agent

Concrete things you'll trip on:

- **Apple Silicon or CUDA Linux/WSL.** Backend uses torch+MPS for the DIT and MLX for the decoder on Mac. On Linux/WSL with flash-attn installed, swap MPS for CUDA and it runs there too. Bare Windows / CUDA-on-Mac aren't on the path.
- **Model weights are gated.** Accept the license at https://huggingface.co/stabilityai/stable-audio-3-medium then `hf download stabilityai/stable-audio-3-medium --local-dir ~/Projects/stable-audio-3/models/stable-audio-3-medium`. The path is hard-coded as `LOCAL_MEDIUM` in `backend/server.py` — change it or symlink, your call.
- **`uv sync` covers everything.** Python 3.11. The `torch` + `mlx` + `mlx-metal` + `safetensors` + `fastapi` + `psutil` stack — let `uv` resolve it.
- **Backend before frontend.** Vite proxies `/api → :5174`; without the backend you get 502s and a red dot in the model status. The backend prints `[backend] ready` when the model finishes loading (~30s on first run, less on subsequent because of fs cache).
- **Port conflicts:** backend binds `127.0.0.1:5174`, frontend dev server `:5173`. Kill any other process on those ports first.
- **HuggingFace login** isn't required at runtime once weights are local, but the first `hf download` needs `hf auth login`.
- **Where to look when something breaks:** backend stderr is verbose (`[generate]`, `[truncate]`, `[inpaint]` log lines). Frontend logs `[play]`, `[vis-toggle]` etc. to the browser console.

The architecture is small enough to read end-to-end in an hour:

```
backend/server.py            FastAPI app, model lifecycle, viz rendering, /api routes
mlx_sa3/ae.py                top-level decoder chain
mlx_sa3/nn_blocks.py         transformer + differential attention + band-mask SWA
mlx_sa3/weights.py           safetensors → mlx weight remap
webui/src/lib/session.svelte.js   shared reactive state + api client
webui/src/lib/MainCanvas.svelte   spectrogram + paint + zoom interaction
webui/src/App.svelte         layout + audio graph + playback wiring
design.md                    the design spec
```

