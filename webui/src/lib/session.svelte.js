// Shared reactive session state.

function maskToRanges(mask) {
  const out = [];
  let start = -1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] && start < 0) start = i;
    else if (!mask[i] && start >= 0) { out.push([start, i]); start = -1; }
  }
  if (start >= 0) out.push([start, mask.length]);
  return out;
}

class Session {
  // track
  trackSeconds    = $state(0);
  sampleRate      = $state(44100);
  downsampleRatio = $state(4096);

  // backend session
  version  = $state(0);     // bumped by backend on every change; bust caches
  hasAudio = $state(false);

  // mask is the single source of truth for what's painted
  mask = $state(new Uint8Array(0));
  ghostMask = $state(new Uint8Array(0));   // last-inpainted regions, for visual recall

  // zoom window over full track, normalized 0..1
  zoomStart = $state(0.0);
  zoomEnd   = $state(1.0);

  // playhead, normalized 0..1 of full track
  playhead = $state(0.0);
  playing  = $state(false);
  volume   = $state(0.7);    // 0..1

  // prompt + settings
  prompt = $state("");
  model  = $state("Medium (ARC)");
  steps  = $state(8);
  cfg    = $state(1.0);
  noise  = $state(0.65);
  seed   = $state(-1);
  duration = $state(190);  // text-to-audio length (sec)

  loras = $state([]);

  generating  = $state(false);
  scrubbingNoise = $state(false);  // true while the user is actively dragging the A2A slider
  modelLoaded = $state(false);   // assume down until pollStats confirms otherwise
  stats = $state({ cpu: 0, vram: 0, ram: 0 });

  get latentCount() {
    return this.mask.length;
  }
  get paintedRanges() {
    return maskToRanges(this.mask);
  }
  get ghostRanges() {
    return maskToRanges(this.ghostMask);
  }
  get hasMask() {
    for (let i = 0; i < this.mask.length; i++) if (this.mask[i]) return true;
    return false;
  }

  setTrackInfo({ count, duration }) {
    this.trackSeconds = duration;
    // resize mask preserving existing values where possible
    const next = new Uint8Array(count);
    const old = this.mask;
    const lim = Math.min(old.length, count);
    for (let i = 0; i < lim; i++) next[i] = old[i];
    this.mask = next;
  }

  paint(startLatent, endLatent, mode) {
    if (endLatent < startLatent) [startLatent, endLatent] = [endLatent, startLatent];
    startLatent = Math.max(0, Math.floor(startLatent));
    endLatent = Math.min(this.mask.length, Math.ceil(endLatent));
    if (endLatent <= startLatent) return;
    const m = new Uint8Array(this.mask);
    const v = mode === "regen" ? 1 : 0;
    for (let i = startLatent; i < endLatent; i++) m[i] = v;
    this.mask = m;
  }

  clearMask() {
    this.mask = new Uint8Array(this.mask.length);
  }
}

export const session = new Session();


// ---------- backend api ----------

export async function apiState() {
  const r = await fetch("/api/state");
  const j = await r.json();
  session.hasAudio = j.has_audio;
  session.version = j.version;
  return j;
}

export async function apiUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  if (!r.ok) throw new Error("upload failed: " + r.status);
  const j = await r.json();
  session.hasAudio = true;
  session.version = j.version;
  session.setTrackInfo(j);
  session.duration = Math.round(j.duration);   // sync length slider to the loaded sample
  return j;
}

export async function apiClear() {
  const r = await fetch("/api/clear", { method: "POST" });
  const j = await r.json();
  session.hasAudio = false;
  session.version = j.version;
  session.mask = new Uint8Array(0);
  session.ghostMask = new Uint8Array(0);
  session.trackSeconds = 0;
  return j;
}

let _genAbort = null;

export function cancelGenerate() {
  if (_genAbort) _genAbort.abort();
  _genAbort = null;
  session.generating = false;
}

export async function apiGenerate() {
  cancelGenerate();
  session.generating = true;
  _genAbort = new AbortController();
  try {
    const body = {
      prompt: session.prompt,
      mask: Array.from(session.mask),
      settings: {
        steps: session.steps,
        cfg: session.cfg,
        seed: session.seed,
        noise: session.noise,
        duration: session.trackSeconds || session.duration,
      },
    };
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: _genAbort.signal,
    });
    if (!r.ok) throw new Error("generate failed: " + r.status);
    const j = await r.json();
    session.hasAudio = true;
    session.version = j.version;
    session.setTrackInfo(j);
    // remember the inpainted regions as ghost (visual recall), then clear the live mask
    if (body.mask.some(v => v)) {
      session.ghostMask = new Uint8Array(body.mask);
    }
    session.mask = new Uint8Array(session.mask.length);
    return j;
  } catch (e) {
    if (e.name === "AbortError") return null;
    throw e;
  } finally {
    _genAbort = null;
    session.generating = false;
  }
}
