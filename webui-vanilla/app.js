// Audio Inpainter — v0 static mock
// No backend yet. Hardcoded "session" state for layout validation.

const session = {
  trackSeconds: 180,
  sampleRate: 44100,
  downsampleRatio: 4096,
  // demo painted regions (latent index ranges)
  paintedRanges: [
    [270, 312],
    [710, 770],
    [1100, 1198],
  ],
  // zoom window over the full track, normalized 0..1
  zoom: { start: 0.18, end: 0.78 },
  // play cursor, normalized 0..1
  playhead: 0.07,
};

session.latentCount = Math.ceil(session.trackSeconds * session.sampleRate / session.downsampleRatio);

// ─── utility ─────────────────────────────────────────────────────────
function dprResize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ─── latent strip ────────────────────────────────────────────────────
function drawLatentStrip() {
  const canvas = document.getElementById("latentStrip");
  const ctx = dprResize(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const N = session.latentCount;
  const colW = w / N;

  // background: faint grey "preserve" bars across the whole track
  ctx.fillStyle = "#1f1f1f";
  const barW = Math.max(0.5, colW * 0.6);
  for (let i = 0; i < N; i++) {
    const x = i * colW + (colW - barW) / 2;
    ctx.fillRect(x, h * 0.2, barW, h * 0.6);
  }

  // painted: accent-blue solid bars, full height
  ctx.fillStyle = cssVar("--accent-blue");
  for (const [s, e] of session.paintedRanges) {
    for (let i = s; i < e; i++) {
      const x = i * colW + (colW - barW) / 2;
      ctx.fillRect(x, h * 0.1, barW, h * 0.8);
    }
    // bounding outline
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 1;
    ctx.strokeRect(s * colW - 0.5, h * 0.08, (e - s) * colW + 1, h * 0.84);
  }
}

// ─── overview waveform with zoom rect ───────────────────────────────
const overviewImg = new Image();
overviewImg.src = "assets/demo_wave.png";
let overviewReady = false;
overviewImg.onload = () => { overviewReady = true; drawOverview(); };

function drawOverview() {
  const canvas = document.getElementById("overview");
  const ctx = dprResize(canvas);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  if (overviewReady) {
    ctx.globalAlpha = 0.55;
    ctx.drawImage(overviewImg, 0, 0, w, h);
    ctx.globalAlpha = 1;
  }

  // zoom rect (accent-blue border, slight tint)
  const zStart = session.zoom.start * w;
  const zEnd = session.zoom.end * w;
  ctx.fillStyle = "rgba(0, 120, 202, 0.10)";
  ctx.fillRect(zStart, 0, zEnd - zStart, h);
  ctx.strokeStyle = cssVar("--accent-blue");
  ctx.lineWidth = 1;
  ctx.strokeRect(zStart + 0.5, 0.5, zEnd - zStart - 1, h - 1);

  // tiny edge handles
  ctx.fillStyle = cssVar("--text-primary");
  ctx.fillRect(zStart - 1, h / 2 - 6, 2, 12);
  ctx.fillRect(zEnd - 1, h / 2 - 6, 2, 12);
}

// ─── region overlays on the spectrogram ─────────────────────────────
function positionRegions() {
  const layer = document.getElementById("regionLayer");
  layer.innerHTML = "";
  const w = layer.clientWidth;
  const N = session.latentCount;
  // zoom window in latent index space
  const lStart = session.zoom.start * N;
  const lEnd = session.zoom.end * N;
  const lSpan = lEnd - lStart;

  for (const [s, e] of session.paintedRanges) {
    // only render if intersects zoom window
    const visStart = Math.max(s, lStart);
    const visEnd = Math.min(e, lEnd);
    if (visEnd <= visStart) continue;
    const leftPct = (visStart - lStart) / lSpan * 100;
    const widthPct = (visEnd - visStart) / lSpan * 100;
    const div = document.createElement("div");
    div.className = "region";
    div.style.left = leftPct + "%";
    div.style.width = widthPct + "%";
    layer.appendChild(div);
  }

  // playhead
  const ph = document.getElementById("playhead");
  const phNorm = (session.playhead - session.zoom.start) / (session.zoom.end - session.zoom.start);
  if (phNorm >= 0 && phNorm <= 1) {
    ph.style.left = (phNorm * 100) + "%";
    ph.style.display = "block";
  } else {
    ph.style.display = "none";
  }
}

// ─── slider value-display sync ──────────────────────────────────────
function wireSliders() {
  for (const slider of document.querySelectorAll(".slider")) {
    const valueEl = slider.parentElement.querySelector(".value");
    if (!valueEl) continue;
    const update = () => {
      const v = Number(slider.value);
      // heuristic: if max <=100, render as 0..1 with 2dp; otherwise as int
      if (slider.max == "100" && slider.min == "0") {
        valueEl.textContent = (v / 100).toFixed(2);
      } else {
        valueEl.textContent = v;
      }
    };
    slider.addEventListener("input", update);
    update();
  }
}

// ─── render & resize ────────────────────────────────────────────────
function render() {
  drawLatentStrip();
  drawOverview();
  positionRegions();
}

window.addEventListener("resize", render);
window.addEventListener("DOMContentLoaded", () => {
  wireSliders();
  render();
});
