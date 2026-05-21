<script>
import { onMount } from "svelte";
import { session } from "./session.svelte.js";
import { dprResize, cssVar } from "./util.js";

let canvas = $state(null);
let body = $state(null);
let isShiftDown = $state(false);

// preview state — commits on release
let paintActive = false;
let paintMode = "regen";
let paintAnchor = -1;
let paintCursor = -1;

function draw() {
  if (!canvas) return;
  const ctx = dprResize(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const N = session.latentCount;
  const colW = w / N;
  const barW = Math.max(0.5, colW * 0.6);

  // preserve background (subtle grey)
  ctx.fillStyle = "#1f1f1f";
  for (let i = 0; i < N; i++) {
    const x = i * colW + (colW - barW) / 2;
    ctx.fillRect(x, h * 0.2, barW, h * 0.6);
  }

  // committed mask painted (accent blue)
  const accent = cssVar("--accent-blue");
  ctx.fillStyle = accent;
  for (let i = 0; i < N; i++) {
    if (session.mask[i]) {
      const x = i * colW + (colW - barW) / 2;
      ctx.fillRect(x, h * 0.1, barW, h * 0.8);
    }
  }

  // preview overlay during drag
  if (paintActive) {
    const lo = Math.min(paintAnchor, paintCursor);
    const hi = Math.max(paintAnchor, paintCursor);
    ctx.fillStyle = paintMode === "regen"
      ? "rgba(0, 120, 202, 0.45)"
      : "rgba(241, 76, 76, 0.45)";
    ctx.fillRect(lo * colW, 0, (hi - lo + 1) * colW, h);
  }
}

$effect(() => {
  // re-draw on mask / latent-count / paint state changes
  session.mask;
  session.latentCount;
  paintActive; paintAnchor; paintCursor;
  draw();
});

function xToLatent(clientX) {
  const rect = body.getBoundingClientRect();
  const norm = (clientX - rect.left) / rect.width;
  return Math.max(0, Math.min(session.latentCount - 1, Math.round(norm * session.latentCount)));
}

function onPointerDown(e) {
  if (e.button !== 0) return;
  paintActive = true;
  paintMode = e.shiftKey ? "preserve" : "regen";
  paintAnchor = xToLatent(e.clientX);
  paintCursor = paintAnchor;
  body.setPointerCapture?.(e.pointerId);
  draw();
}
function onPointerMove(e) {
  if (!paintActive) return;
  paintCursor = xToLatent(e.clientX);
  draw();
}
function onPointerUp(e) {
  if (!paintActive) return;
  const lo = Math.min(paintAnchor, paintCursor);
  const hi = Math.max(paintAnchor, paintCursor) + 1;
  session.paint(lo, hi, paintMode);
  paintActive = false;
  paintAnchor = -1; paintCursor = -1;
  body?.releasePointerCapture?.(e.pointerId);
}

function onKeyDown(e) { if (e.key === "Shift") isShiftDown = true; }
function onKeyUp(e)   { if (e.key === "Shift") isShiftDown = false; }

onMount(() => {
  draw();
  const ro = new ResizeObserver(draw);
  ro.observe(canvas);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  return () => {
    ro.disconnect();
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
  };
});
</script>

<div class="canvas-row">
  <span class="row-label">Latents</span>
  <div
    class="row-body" class:erasing={isShiftDown}
    bind:this={body}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointercancel={onPointerUp}
  >
    <canvas bind:this={canvas}></canvas>
  </div>
</div>

<style>
.canvas-row {
  display: grid;
  grid-template-columns: var(--row-label-w) 1fr;
  border-bottom: 1px solid var(--border-color);
  min-height: 0;
  min-width: 0;
  overflow: hidden;
}
.row-label {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 0 var(--gap-3);
  color: var(--text-muted);
  font-size: 10px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  background: var(--bg-dark);
  z-index: 2;
}
.row-body {
  position: relative;
  background: var(--bg-dark);
  cursor: crosshair;
  touch-action: none;
}
.row-body.erasing { cursor: not-allowed; }
canvas { display: block; width: 100%; height: 100%; }
</style>
