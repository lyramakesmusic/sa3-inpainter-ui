<script>
import { onMount } from "svelte";
import { session } from "./session.svelte.js";
import { dprResize } from "./util.js";

let canvas = $state(null);
let envelope = $state(null);   // { sr, downsample, count, data: [[peak, r, g, b], ...] }

async function loadEnvelope() {
  if (!session.hasAudio) { envelope = null; return; }
  const r = await fetch(`/api/envelope.json?v=${session.version}`);
  const j = await r.json();
  envelope = j.count > 0 ? j : null;
}

$effect(() => { session.version; loadEnvelope(); });

function draw() {
  if (!canvas) return;
  const ctx = dprResize(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!envelope || !session.hasAudio) return;

  const N = envelope.count;
  const lStart = session.zoomStart * N;
  const lEnd   = session.zoomEnd   * N;
  const lSpan = Math.max(1, lEnd - lStart);
  const cw = w / lSpan;

  const barW = cw;
  const midY = h / 2;
  const data = envelope.data;
  const ghost = session.ghostMask;
  const hasGhost = ghost && ghost.length === N;

  // compute the "original" max peak (excluding ghost = newly-inpainted latents).
  // newly-inpainted latents have their visual peak capped to 1.05x of that, so a
  // hot inpaint doesn't compress the original visually.
  let originalMax = 0;
  if (hasGhost) {
    for (let i = 0; i < N; i++) {
      if (!ghost[i] && data[i][0] > originalMax) originalMax = data[i][0];
    }
  }
  const cap = originalMax > 0 ? originalMax * 1.05 : Infinity;

  const iStart = Math.max(0, Math.floor(lStart));
  const iEnd   = Math.min(N, Math.ceil(lEnd));
  for (let i = iStart; i < iEnd; i++) {
    const [peak, r, g, b] = data[i];
    const x = (i - lStart) * cw;
    const isGhost = hasGhost && ghost[i] === 1;
    const visPeak = isGhost ? Math.min(peak, cap) : peak;
    const half = visPeak * (h * 0.5);
    ctx.fillStyle = `rgb(${Math.round(r*255)}, ${Math.round(g*255)}, ${Math.round(b*255)})`;
    ctx.fillRect(x, midY - half, barW + 0.5, half * 2);
  }
}

$effect(() => {
  envelope; session.zoomStart; session.zoomEnd; session.hasAudio; session.version; session.ghostMask;
  draw();
});

onMount(() => {
  loadEnvelope();
  const ro = new ResizeObserver(draw);
  ro.observe(canvas);
  return () => ro.disconnect();
});
</script>

<canvas bind:this={canvas}></canvas>

<style>
canvas { display: block; width: 100%; height: 100%; pointer-events: none; }
</style>
