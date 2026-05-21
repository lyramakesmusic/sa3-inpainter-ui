<script>
import { onMount } from "svelte";
import { session, apiState, apiUpload } from "./lib/session.svelte.js";
import TopBar from "./lib/TopBar.svelte";
import OverviewWave from "./lib/OverviewWave.svelte";
import TimeAxis from "./lib/TimeAxis.svelte";
import MainCanvas from "./lib/MainCanvas.svelte";
import RightRail from "./lib/RightRail.svelte";
import BottomBar from "./lib/BottomBar.svelte";

let audioEl = $state(null);
let isDragOver = $state(false);

// Web Audio graph for live volume + masked-region ducking
let audioCtx = null;
let srcNode = null;
let filterNode = null;
let gainNode = null;
const LP_CUTOFF_MASKED = 1500;   // Hz when playing inside an inpaint region
const LP_CUTOFF_BYPASS = 22000;  // effectively off
const MASKED_DB = -2;             // ducked level inside inpaint regions
const SMOOTH_S = 0.06;            // time-constant for setTargetAtTime

function ensureAudioGraph() {
  if (audioCtx || !audioEl) return;
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    srcNode = audioCtx.createMediaElementSource(audioEl);
    filterNode = audioCtx.createBiquadFilter();
    filterNode.type = "lowpass";
    filterNode.frequency.value = LP_CUTOFF_BYPASS;
    filterNode.Q.value = 0.5;     // shallow slope
    gainNode = audioCtx.createGain();
    gainNode.gain.value = session.volume;
    srcNode.connect(filterNode).connect(gainNode).connect(audioCtx.destination);
  } catch (e) { console.warn("audio graph init failed:", e); }
}

// keyboard shortcuts
function onKeyDown(e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  if (e.code === "Space") {
    e.preventDefault();
    session.playing = !session.playing;
  } else if (e.key === "Backspace") {
    if (session.hasMask) {
      e.preventDefault();
      session.clearMask();
    }
  }
}

// drag-drop file
function onDragOver(e) { e.preventDefault(); isDragOver = true; }
function onDragLeave(e) { isDragOver = false; }
async function onDrop(e) {
  e.preventDefault();
  isDragOver = false;
  const file = e.dataTransfer?.files?.[0];
  if (file) await apiUpload(file);
}

// playback wiring: session.playing ↔ audio element
$effect(() => {
  if (!audioEl) return;
  if (session.playing) {
    ensureAudioGraph();
    if (audioCtx?.state === "suspended") audioCtx.resume();
    audioEl.play().catch(() => session.playing = false);
  } else {
    audioEl.pause();
  }
});

// live volume: if audio graph active, use gain node; otherwise fall back to element volume
$effect(() => {
  if (gainNode && audioCtx) {
    gainNode.gain.setTargetAtTime(session.volume * currentMaskedGain(), audioCtx.currentTime, SMOOTH_S);
  } else if (audioEl) {
    audioEl.volume = session.volume;
  }
});

function currentMaskedGain() {
  return _maskedNow ? Math.pow(10, MASKED_DB / 20) : 1.0;
}
let _maskedNow = false;

// when audio source bumps version: swap src without interrupting playback
$effect(() => {
  if (!audioEl) return;
  session.version;
  if (!session.hasAudio) { audioEl.src = ""; return; }
  const wasPlaying = !audioEl.paused;
  const t = audioEl.currentTime || 0;
  const onReady = () => {
    audioEl.removeEventListener("loadedmetadata", onReady);
    try { audioEl.currentTime = Math.min(t, (audioEl.duration || t)); } catch {}
    if (wasPlaying) audioEl.play().catch(() => {});
  };
  audioEl.addEventListener("loadedmetadata", onReady);
  audioEl.src = `/api/audio?v=${session.version}`;
});

// when user moves playhead via UI, sync audio
$effect(() => {
  if (!audioEl || !session.hasAudio) return;
  const targetTime = session.playhead * session.trackSeconds;
  if (Math.abs(audioEl.currentTime - targetTime) > 0.5) {
    audioEl.currentTime = targetTime;
  }
});

function onAudioEnded() { session.playing = false; }

// smooth playhead + filter ducking inside masked regions
let rafHandle = 0;
function tick() {
  if (!audioEl) { rafHandle = 0; return; }
  if (session.trackSeconds > 0) {
    session.playhead = audioEl.currentTime / session.trackSeconds;
  }
  if (audioCtx && filterNode && gainNode && session.mask.length > 0) {
    const latIdx = Math.floor(audioEl.currentTime * session.sampleRate / session.downsampleRatio);
    const inMask = latIdx >= 0 && latIdx < session.mask.length && session.mask[latIdx] === 1;
    if (inMask !== _maskedNow) {
      _maskedNow = inMask;
      filterNode.frequency.value = inMask ? LP_CUTOFF_MASKED : LP_CUTOFF_BYPASS;
      gainNode.gain.value = session.volume * currentMaskedGain();
    }
  }
  // tick while the user intends playback, regardless of brief audioEl.paused during src swaps
  if (session.playing) {
    rafHandle = requestAnimationFrame(tick);
  } else {
    rafHandle = 0;
  }
}
$effect(() => {
  if (session.playing && !rafHandle) {
    rafHandle = requestAnimationFrame(tick);
  }
});

// elapsed-time counter under the generating spinner
let genElapsed = $state(0);
let genStart = 0;
let genTimer = 0;
$effect(() => {
  if (session.generating) {
    genStart = performance.now();
    genElapsed = 0;
    if (genTimer) cancelAnimationFrame(genTimer);
    const tickGen = () => {
      if (!session.generating) { genTimer = 0; return; }
      genElapsed = (performance.now() - genStart) / 1000;
      genTimer = requestAnimationFrame(tickGen);
    };
    genTimer = requestAnimationFrame(tickGen);
  }
});

async function pollStats() {
  try {
    const r = await fetch("/api/stats");
    if (!r.ok) { session.modelLoaded = false; return; }
    const j = await r.json();
    session.stats = {
      cpu: j.cpu,
      ram: Math.round(j.ram_used / j.ram_total * 100),
      ramUsedGb: j.ram_used,
      ramTotalGb: j.ram_total,
      mpsAllocGb: j.mps_alloc,
    };
    session.modelLoaded = j.model_loaded;
  } catch (e) {
    session.modelLoaded = false;
  }
}

let statsInterval = 0;

onMount(() => {
  window.addEventListener("keydown", onKeyDown);
  apiState();
  pollStats();
  statsInterval = setInterval(pollStats, 2000);
  return () => {
    window.removeEventListener("keydown", onKeyDown);
    if (rafHandle) cancelAnimationFrame(rafHandle);
    if (statsInterval) clearInterval(statsInterval);
  };
});
</script>

<div
  class="app"
  class:drag-over={isDragOver}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  ondrop={onDrop}
>
  <TopBar />
  <main class="editor">
    <section class="canvas-column">
      <OverviewWave />
      <div class="spacer"></div>
      <TimeAxis />
      <MainCanvas />
      <div class="spacer"></div>
    </section>
    <RightRail />
  </main>
  <BottomBar />

  {#if session.generating}
    <div class="generating-overlay">
      <div class="spinner"></div>
      <div>generating… <span class="gen-elapsed">{genElapsed.toFixed(1)}s</span></div>
    </div>
  {/if}

  {#if isDragOver}
    <div class="drop-overlay">drop audio file to load</div>
  {/if}
</div>

<audio
  bind:this={audioEl}
  onended={onAudioEnded}
  preload="auto"
></audio>

<style>
.app {
  display: grid;
  grid-template-rows: var(--topbar-h) 1fr var(--bottombar-h);
  height: 100vh;
  position: relative;
}
.editor {
  display: grid;
  grid-template-columns: 1fr var(--rail-w);
  column-gap: var(--gap-1);    /* tiny breathing room before the sidebar */
  overflow: hidden;
  min-height: 0;
}
.canvas-column {
  display: grid;
  grid-template-rows: 32px var(--main-margin) 44px 1fr var(--main-margin);
  background: var(--bg-dark);
  overflow: visible;       /* let the playhead-time label flow into the bottom spacer */
  min-height: 0;
  min-width: 0;
}
.spacer { background: var(--bg-dark); }
/* every row gets left+right indents, EXCEPT the time-axis (3rd row) which is
   edge-to-edge so the grey playbar strip extends to both screen edges */
.canvas-column > :global(*) {
  padding-left: var(--gap-2);
  padding-right: var(--gap-2);
}
.canvas-column > :global(*:nth-child(3)) {
  padding-left: 0;
  padding-right: 0;
}

.generating-overlay {
  position: absolute;
  inset: var(--topbar-h) var(--rail-w) var(--bottombar-h) 0;
  background: rgba(0,0,0,0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--gap-3);
  color: var(--text-primary);
  z-index: 100;
  font-size: 13px;
  pointer-events: none;   /* still let pinch/scroll through to the canvas */
}
.spinner {
  width: 32px;
  height: 32px;
  border: 2px solid var(--border-color);
  border-top-color: var(--accent-blue);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.gen-elapsed { color: var(--text-secondary); font-variant-numeric: tabular-nums; }

.drop-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 120, 202, 0.15);
  border: 2px dashed var(--accent-blue);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: var(--text-primary);
  z-index: 99;
  pointer-events: none;
}
</style>
