<script>
import { onMount } from "svelte";
import { session } from "./session.svelte.js";
import { dprResize, cssVar } from "./util.js";

let canvas = $state(null);
let body = $state(null);
let img = null;
let imgLoaded = $state(false);

// drag state
let dragMode = null;          // 'pan' | 'resizeL' | 'resizeR' | null
let dragStartX = 0;
let dragStartZoom = null;
const EDGE_PX = 6;
const INSET_PX = 10;     // visual inset so the zoom-rect handles don't sit at the screen edge

function draw() {
  if (!canvas) return;
  const ctx = dprResize(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  // no audio loaded → don't show the demo placeholder wave
  if (!session.hasAudio) return;

  const innerL = INSET_PX;
  const innerW = Math.max(1, w - 2 * INSET_PX);

  // pre-greyed overview waveform asset (already amplitude bars on black). Squish vertically.
  if (imgLoaded && img) {
    const padY = h * 0.10;
    ctx.drawImage(img, innerL, padY, innerW, h - padY * 2);
  }

  const zStart = innerL + session.zoomStart * innerW;
  const zEnd   = innerL + session.zoomEnd   * innerW;

  ctx.fillStyle = "rgba(0, 120, 202, 0.18)";
  ctx.fillRect(zStart, 0, zEnd - zStart, h);

  ctx.fillStyle = cssVar("--text-primary");
  const handleW = 3;
  ctx.fillRect(zStart - handleW / 2, 0, handleW, h);
  ctx.fillRect(zEnd   - handleW / 2, 0, handleW, h);

  // mini-playhead — tracks GLOBAL playhead position across the entire track
  const phX = innerL + session.playhead * innerW;
  // dark surround for contrast
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  ctx.fillRect(phX - 1.5, 0, 3, h);
  // white center line
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(phX - 0.5, 0, 1, h);
}

$effect(() => {
  session.zoomStart; session.zoomEnd; session.playhead; session.hasAudio; imgLoaded;
  draw();
});

function hitTest(clientX) {
  if (!body) return null;
  const rect = body.getBoundingClientRect();
  const x = clientX - rect.left;
  const innerL = INSET_PX;
  const innerW = Math.max(1, rect.width - 2 * INSET_PX);
  const zStart = innerL + session.zoomStart * innerW;
  const zEnd   = innerL + session.zoomEnd   * innerW;
  if (Math.abs(x - zStart) <= EDGE_PX) return "resizeL";
  if (Math.abs(x - zEnd)   <= EDGE_PX) return "resizeR";
  if (x >= zStart && x <= zEnd) return "pan";
  return null;
}

function cursorFor(mode) {
  if (mode === "resizeL" || mode === "resizeR") return "ew-resize";
  if (mode === "pan") return "grab";
  return "default";
}

let hoverCursor = $state("default");

function onPointerMove(e) {
  if (dragMode) return; // cursor stays in drag mode
  hoverCursor = cursorFor(hitTest(e.clientX));
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  if (!session.hasAudio) return;
  const mode = hitTest(e.clientX);
  if (!mode) return;
  dragMode = mode;
  dragStartX = e.clientX;
  dragStartZoom = { start: session.zoomStart, end: session.zoomEnd };
  hoverCursor = mode === "pan" ? "grabbing" : "ew-resize";
  body.setPointerCapture?.(e.pointerId);
  e.preventDefault();
}

function onPointerDrag(e) {
  if (!dragMode) return;
  const rect = body.getBoundingClientRect();
  const innerW = Math.max(1, rect.width - 2 * INSET_PX);
  const dxNorm = (e.clientX - dragStartX) / innerW;
  let { start, end } = dragStartZoom;
  const minSpan = 0.005;
  if (dragMode === "pan") {
    let delta = dxNorm;
    // clamp so we don't go out of bounds
    delta = Math.max(-start, Math.min(1 - end, delta));
    session.zoomStart = start + delta;
    session.zoomEnd = end + delta;
  } else if (dragMode === "resizeL") {
    let v = Math.max(0, Math.min(end - minSpan, start + dxNorm));
    session.zoomStart = v;
  } else if (dragMode === "resizeR") {
    let v = Math.min(1, Math.max(start + minSpan, end + dxNorm));
    session.zoomEnd = v;
  }
}

function onPointerUp(e) {
  if (!dragMode) return;
  dragMode = null;
  body.releasePointerCapture?.(e.pointerId);
  hoverCursor = cursorFor(hitTest(e.clientX));
}

function setSrc() {
  if (!session.hasAudio) { imgLoaded = false; return; }
  img = new Image();
  img.onload = () => { imgLoaded = true; draw(); };
  img.src = `/api/overview.png?v=${session.version}`;
}

$effect(() => { session.version; session.hasAudio; setSrc(); });

onMount(() => {
  const ro = new ResizeObserver(draw);
  ro.observe(canvas);
  return () => ro.disconnect();
});
</script>

<div
  class="row-body"
  bind:this={body}
  style="cursor: {hoverCursor}"
  onpointermove={(e) => { onPointerMove(e); onPointerDrag(e); }}
  onpointerdown={onPointerDown}
  onpointerup={onPointerUp}
  onpointercancel={onPointerUp}
>
  <canvas bind:this={canvas}></canvas>
</div>

<style>
.row-body {
  position: relative;
  background: var(--bg-dark);
  touch-action: none;
  height: 100%;
  width: 100%;
  overflow: hidden;
}
canvas { display: block; width: 100%; height: 100%; pointer-events: none; }
</style>
