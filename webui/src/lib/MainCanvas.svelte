<script>
import { session } from "./session.svelte.js";
import WaveformCanvas from "./WaveformCanvas.svelte";

let canvasBody = $state(null);
let isShiftDown = $state(false);

// preview state — only commits on pointerup
let paintActive = $state(false);
let paintMode = $state("regen");
let paintAnchor = $state(-1);
let paintCursor = $state(-1);
let paintDownX = 0;
let paintMovedFar = $state(false);
const PAINT_DEAD_ZONE_PX = 5;

// preview range in latent indices (committed only on release)
let previewRange = $derived.by(() => {
  if (!paintActive) return null;
  const lo = Math.min(paintAnchor, paintCursor);
  const hi = Math.max(paintAnchor, paintCursor) + 1;
  return [lo, hi];
});

// committed regions to render
let ghostRegions = $derived.by(() => {
  const N = session.latentCount;
  if (N === 0) return [];
  const lStart = session.zoomStart * N;
  const lEnd = session.zoomEnd * N;
  const lSpan = Math.max(1, lEnd - lStart);
  const out = [];
  for (const [s, e] of session.ghostRanges) {
    const visStart = Math.max(s, lStart);
    const visEnd = Math.min(e, lEnd);
    if (visEnd <= visStart) continue;
    out.push({ left: (visStart - lStart) / lSpan * 100, width: (visEnd - visStart) / lSpan * 100 });
  }
  return out;
});

let regions = $derived.by(() => {
  const N = session.latentCount;
  const lStart = session.zoomStart * N;
  const lEnd = session.zoomEnd * N;
  const lSpan = Math.max(1, lEnd - lStart);
  const out = [];
  for (const [s, e] of session.paintedRanges) {
    const visStart = Math.max(s, lStart);
    const visEnd = Math.min(e, lEnd);
    if (visEnd <= visStart) continue;
    out.push({
      left: (visStart - lStart) / lSpan * 100,
      width: (visEnd - visStart) / lSpan * 100,
    });
  }
  return out;
});

let previewBox = $derived.by(() => {
  if (!previewRange) return null;
  const N = session.latentCount;
  const lStart = session.zoomStart * N;
  const lEnd = session.zoomEnd * N;
  const lSpan = Math.max(1, lEnd - lStart);
  const visStart = Math.max(previewRange[0], lStart);
  const visEnd = Math.min(previewRange[1], lEnd);
  if (visEnd <= visStart) return null;
  return {
    left: (visStart - lStart) / lSpan * 100,
    width: (visEnd - visStart) / lSpan * 100,
  };
});

let playheadPct = $derived.by(() => {
  const span = session.zoomEnd - session.zoomStart;
  if (span <= 0) return -1;
  const p = (session.playhead - session.zoomStart) / span;
  return (p >= 0 && p <= 1) ? p * 100 : -1;
});

let playheadTimeLabel = $derived.by(() => {
  const t = session.playhead * session.trackSeconds;
  if (!isFinite(t) || t < 0) return "00:00.000";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
});

// transform on the displayed image so zoom pan/scale follows zoom rect
let imgTransform = $derived.by(() => {
  const span = Math.max(0.001, session.zoomEnd - session.zoomStart);
  const scale = 1 / span;
  const tx = -session.zoomStart * 100;
  return `transform: scaleX(${scale}) translateX(${tx}%); transform-origin: 0 0;`;
});


function xToLatent(clientX) {
  const rect = canvasBody.getBoundingClientRect();
  const norm = (clientX - rect.left) / rect.width;
  const N = session.latentCount;
  const lStart = session.zoomStart * N;
  const lEnd = session.zoomEnd * N;
  return Math.round(lStart + norm * (lEnd - lStart));
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  if (!session.hasAudio || session.generating) return;
  paintActive = true;
  paintMode = e.shiftKey ? "preserve" : "regen";
  paintAnchor = xToLatent(e.clientX);
  paintCursor = paintAnchor;
  paintDownX = e.clientX;
  paintMovedFar = false;
  canvasBody.setPointerCapture?.(e.pointerId);
}

function onPointerMove(e) {
  if (!paintActive) return;
  paintCursor = xToLatent(e.clientX);
  if (!paintMovedFar && Math.abs(e.clientX - paintDownX) >= PAINT_DEAD_ZONE_PX) {
    paintMovedFar = true;
  }
}

function onWheel(e) {
  if (!session.hasAudio) return;
  // scroll → zoom (anchored at cursor); shift+scroll OR horizontal swipe → pan
  e.preventDefault();
  const rect = canvasBody.getBoundingClientRect();
  const cursorNorm = (e.clientX - rect.left) / rect.width;
  const span = session.zoomEnd - session.zoomStart;
  const anchor = session.zoomStart + cursorNorm * span;

  // pinch (ctrlKey) always zooms; shift forces pan; otherwise pick based on dominant axis.
  // each wheel event is independent, so hybrid gestures (pinch → pan without lifting fingers)
  // flow naturally — each event is routed by its own ctrlKey/delta.
  let wantPan;
  if (e.ctrlKey) wantPan = false;
  else if (e.shiftKey) wantPan = true;
  else wantPan = Math.abs(e.deltaX) > Math.abs(e.deltaY);

  if (wantPan) {
    const delta = (e.deltaX || e.deltaY) / rect.width * span;
    let newStart = session.zoomStart + delta;
    let newEnd   = session.zoomEnd + delta;
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (newEnd > 1)   { newStart -= (newEnd - 1); newEnd = 1; }
    session.zoomStart = newStart;
    session.zoomEnd = newEnd;
  } else {
    const k = e.ctrlKey ? 0.02 : 0.0025;
    const factor = Math.exp(e.deltaY * k);
    let newSpan = Math.max(0.002, Math.min(1, span * factor));
    let newStart = anchor - cursorNorm * newSpan;
    let newEnd   = anchor + (1 - cursorNorm) * newSpan;
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (newEnd > 1)   { newStart -= (newEnd - 1); newEnd = 1; newStart = Math.max(0, newStart); }
    session.zoomStart = newStart;
    session.zoomEnd = newEnd;
  }
}

function onPointerUp(e) {
  if (!paintActive) return;
  if (paintMovedFar) {
    // committed drag → write the painted range to the mask
    const r = previewRange;
    if (r) session.paint(r[0], r[1], paintMode);
  } else {
    // click without dragging → seek playhead instead of paint
    const rect = canvasBody.getBoundingClientRect();
    const norm = (e.clientX - rect.left) / rect.width;
    const span = session.zoomEnd - session.zoomStart;
    session.playhead = Math.max(0, Math.min(1, session.zoomStart + norm * span));
  }
  paintActive = false;
  paintAnchor = -1;
  paintCursor = -1;
  paintMovedFar = false;
  canvasBody?.releasePointerCapture?.(e.pointerId);
}

// global shift tracking for cursor display
function onKeyDown(e) { if (e.key === "Shift") isShiftDown = true; }
function onKeyUp(e)   { if (e.key === "Shift") isShiftDown = false; }

$effect(() => {
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", () => isShiftDown = false);
  return () => {
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
  };
});
</script>

<div class="canvas-row main">
  <div
    class="row-body main-canvas"
    class:erasing={isShiftDown || paintMode === "preserve" && paintActive}
    bind:this={canvasBody}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
    onwheel={onWheel}
  >
    <div class="pane spec-pane" class:with-audio={session.hasAudio}>
      {#if session.hasAudio}
        <img src={`/api/spec.png?v=${session.version}`} alt="" draggable="false" style={imgTransform} />
        <!-- noise preview overlay — only visible while user is actively dragging the A2A slider -->
        {#if session.scrubbingNoise && !session.hasMask && session.noise > 0}
          <img src="/api/noise_spec.png" alt="" draggable="false"
               class="noise-overlay" style="{imgTransform}; opacity: {session.noise}" />
        {/if}
      {/if}
    </div>
    <div class="pane wave-pane">
      <WaveformCanvas />
    </div>
    {#if !session.hasAudio && !session.generating}
      <div class="empty-hint">load a sample or type a prompt to start</div>
    {/if}
    <div class="region-layer">
      {#each ghostRegions as r}
        <div class="region ghost" style="left: {r.left}%; width: {r.width}%"></div>
      {/each}
      {#each regions as r}
        <div class="region" style="left: {r.left}%; width: {r.width}%"></div>
      {/each}
      {#if previewBox}
        <div class="region preview" class:erase={paintMode === "preserve"}
             style="left: {previewBox.left}%; width: {previewBox.width}%"></div>
      {/if}
    </div>
    {#if playheadPct >= 0 && session.hasAudio && !session.generating}
      <div class="playhead" style="left: {playheadPct}%"></div>
      <div class="playhead-time" style="left: {playheadPct}%">{playheadTimeLabel}</div>
    {/if}
  </div>
</div>

<style>
.canvas-row.main {
  display: block;
  min-height: 0;
  min-width: 0;
  overflow: visible;        /* let the playhead time label escape below */
  height: 100%;
  position: relative;
}
.row-label.freq-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: var(--gap-2) var(--gap-3);
  align-items: flex-end;
  color: var(--text-muted);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  background: var(--bg-dark);
  z-index: 2;
}
.row-body { position: relative; background: var(--bg-dark); min-height: 0; min-width: 0; height: 100%; }
.main-canvas {
  overflow: visible;
  cursor: crosshair;
  touch-action: none;
  min-height: 0; min-width: 0;
  display: grid;
  grid-template-rows: 65% 35%;
  height: 100%;
}
.main-canvas.erasing { cursor: not-allowed; }

.pane {
  position: relative;
  overflow: hidden;
  min-height: 0;
}
.spec-pane.with-audio { border-bottom: 1px solid var(--border-color); }
.wave-pane { padding: 6px 0; }   /* breathing room top + bottom of waveform */

img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: fill;
  pointer-events: none;
  user-select: none;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}
img.noise-overlay { mix-blend-mode: screen; }
.region-layer { position: absolute; inset: 0; pointer-events: none; }
.region {
  position: absolute;
  top: 0; bottom: 0;
  /* greyscale-desaturate the spec+wave behind this region.
     stack: backdrop kills color, then a dark overlay on top pulls blacks
     back to true black (semi-transparent grey alone would *brighten* dark bg). */
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: saturate(0);
  -webkit-backdrop-filter: saturate(0);
  border-left: 1px dashed rgba(255, 255, 255, 0.85);
  border-right: 1px dashed rgba(255, 255, 255, 0.85);
}
.region.preview {
  background: rgba(0, 120, 202, 0.32);
  border-left-color: rgba(255, 255, 255, 1);
  border-right-color: rgba(255, 255, 255, 1);
}
.region.ghost {
  background: rgba(0, 120, 202, 0.16);
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  border: 0;
  border-left: 1px solid rgba(0, 120, 202, 0.85);
  border-right: 1px solid rgba(0, 120, 202, 0.85);
}
.region.preview.erase {
  background: rgba(241, 76, 76, 0.28);
  border-left-color: rgba(241, 76, 76, 1);
  border-right-color: rgba(241, 76, 76, 1);
}
.playhead {
  position: absolute;
  top: 0; bottom: 0;
  width: 1px;
  background: #ffffff;
  box-shadow: 2px 0 0 rgba(0,0,0,0.85), -2px 0 0 rgba(0,0,0,0.85);
  pointer-events: none;
}
.playhead-time {
  position: absolute;
  bottom: -22px;        /* below the waveform pane, in the bottom spacer area */
  transform: translateX(-50%);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 11px;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  pointer-events: none;
  z-index: 4;
  white-space: nowrap;
}
.empty-hint {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  font-size: 13px;
  pointer-events: none;
}
</style>
