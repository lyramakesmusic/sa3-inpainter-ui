# SA3 Inpainter — Design Spec

Working name. **A studio for a single audio track**, AI-native. SA3's arbitrary inpainting and audio-to-audio capabilities should feel like an instrument, not a form. Heavy interaction, every parameter visualized, every drag updates the canvas.

- One track at a time. No arrangement view, no multi-clip, no MIDI.
- If source audio is loaded → editing / inpainting / audio-to-audio mode.
- If nothing is loaded → from-scratch text-to-audio. Same UI, latent strip is just empty until you generate.
- **Inpainting happens in place.** No separate "Result" panel — the displayed audio updates. A variant navigator lets you flip through alternate generations without losing history.

This is a **design doc**, not an implementation plan. Open questions are marked `?`.

---

## 0. Interaction principles (apply everywhere)

Set by the painting model (§2). Hold the rest of the tool to the same standard.

- **Modeless.** No tools, no mode buttons. Where you click defines what happens.
- **One gesture, one modifier.** Each surface has a primary drag behavior plus an optional `shift` (or contextual) inverse. No third modifier on the same surface.
- **Cursor announces the gesture.** Before clicking, the cursor's shape (and any small glyph appended) tells you what's about to happen. Holding a modifier updates the cursor immediately.
- **Direct manipulation, no dialogs.** Paint, drag, scrub. No confirmation popups. No "Are you sure?". Undo is the universal safety net.
- **No vestigial chrome.** If a control isn't applicable in the current state, it's hidden, not greyed out.
- **Snap to model reality.** Drags on the canvas snap to the underlying latent grid; sliders that map to a discrete set snap to those discrete values. Don't pretend a control is continuous when the model bins it.
- **State is the source of truth, view follows.** Every visible element is a render of explicit state. Slider moves → state updates → all dependent visualizations update in lockstep (the cascading interpolation pattern in §10).

---

## 1. Core mental model

SA3's inpaint mask is **`(B, T_lat)`** — one bit per latent timestep, no frequency dimension. So:

- Editing happens in time only. No vertical resizing, no spectrogram rectangles, no frequency-cropping.
- Each latent (~46ms of audio at 44.1kHz / 4096 downsample) can independently be **preserved** or **regenerated**.
- The full edit state is a binary array of length T_lat. Regions and prompts are views on top.

---

## 2. Interaction model — gestural, toolless

No tool palette. Drawing is one gesture with one modifier:

| Gesture | Effect |
|---|---|
| **drag** on main canvas | Paint "regenerate" onto the covered latents (snapped to latent grid) |
| **shift+drag** on main canvas | Erase / unpaint (mark latents back to "preserve") |
| **click** (no drag) on main canvas | Move play cursor |
| **click** region card (below canvas) | Highlight that region; scroll into view if needed |
| **drag** on latent strip column(s) | Same paint behavior — strip is just another surface for the same gesture |
| **drag** on overview zoom-selector body | Pan view |
| **drag** on overview zoom-selector edges | Resize zoom window |
| **scroll** on main canvas | Zoom time axis |
| **cmd/space + drag** anywhere | Pan |
| **backspace** with regions highlighted via cards | Delete (clears those latents back to preserve) |

Cursors:
- Default: paint cursor (slim brush icon)
- Shift held + over main canvas: paint cursor **with small `x`** to signal erase
- Over overview body: grab cursor
- Over overview edges: horizontal resize cursor
- Over region card: pointer

That's the whole interaction model. No regions to "move," no edge-handles to drag — if you want a region somewhere else, just paint it there and shift-erase the old one. Cheap and direct.

---

## 3. Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  Logo · Audio Inpainter                              [New]   [Clear]   │
├──────────────────────────────────────────────────┬─────────────────────┤
│  Latent / Paintouts  (one vertical bar/latent)   │                     │
│                                                  │                     │
│  Overview  (full track grey wave + zoom rect)    │   Right rail:       │
│                                                  │     · Prompt        │
│  ┌─ Main canvas ───────────────────────────┐     │     · Generation    │
│  │  Spectrogram (default) — log-Hz, magma  │     │     · LoRAs         │
│  │     [region rect] [region rect] [...]   │     │     · Inpaint CTA   │
│  │     (regions = full-height blue strips  │     │       + Reroll      │
│  │      with L/R I-handles, no corner dots)│     │       + Variant nav │
│  │                                         │     │                     │
│  │  Time axis  (00:00, 00:05, …)           │     │                     │
│  │  Region cards  ( [1] 00:12.4-00:14.1 )  │     │                     │
│  └─────────────────────────────────────────┘     │                     │
├──────────────────────────────────────────────────┴─────────────────────┤
│  ▶ ⏸ ⏹ │ 🔊 │  00:12.428 / 01:05.237  │ ⌽ Spec/Wave │ ● Model loaded │
│                                          │ CPU 18% · CUDA 32% · …   │
└────────────────────────────────────────────────────────────────────────┘
```

- Right rail: ~320px, sections collapsible.
- Main canvas: everything else, resizes fluidly.
- Top bar: ~48px. Logo, app name, New, Clear. No file menu.
- Bottom bar: ~40px. See §9.

---

## 4. The latent strip — primary edit surface

A row of small vertical bars at the top of the canvas. **One bar per latent** at typical track length (~2000 bars for a 3-min track), full width of the canvas.

### Visual encoding
- **Preserve** latent: bar tinted with its PCA-3 RGB (computed from the source latents via `latent[t] → PCA → rgb`). You see the latent texture of the source audio.
- **Regenerate** latent: bar outlined in blue, slightly elevated. Marks the "hole."
- A contiguous run of regenerate-bars renders as a single boxed group with the region's number above.
- During diffusion streaming: regenerate bars animate as their latent denoises (seed jitter → final PCA color).

### Behavior
- Width locks 1:1 to the spectrogram time axis below.
- Hover bar → tooltip: latent index · audio time · PCA values if preserved.
- Click bar → toggle that single latent.
- Drag across bars → bulk toggle to whichever state the drag started flipping toward.
- Painting a contiguous run promotes it to a "region" (see §6).

---

## 5. The overview — zoom navigator (Ableton-style)

Below the latent strip, a thin row: the full track waveform rendered tiny in grey, with a **blue rectangular selector** marking the current zoom window into the main spectrogram.

- Drag selector body → pan
- Drag selector edges → resize (zoom)
- Click outside selector → jump zoom window to cursor
- Always shows full track — global structure visible at a glance

No frequency content visualized here; it's pure time-navigation.

---

## 6. Main canvas — spectrogram (default) / waveform (toggleable)

Single visualization in the middle, toggled via the bottom-bar control. Spectrogram is the default for inpainting work; waveform mode is for amplitude-focused views.

### Spectrogram mode
- Log-Hz vertical axis on left (30Hz → 20kHz; tick marks: 30, 60, 120, 250, 500, 1k, 2k, 5k, 10k, 20k)
- Magma colormap (locked, matches `viz_demo.py`)
- Inpaint regions: full-height translucent **blue** rectangles, dashed white border. **No handles** — regions aren't moved or resized, only painted/erased.
- Play cursor: vertical hairline extending through everything

### Waveform mode
- Frequency-balance RGB (`viz_demo.py` algorithm)
- Same region overlays
- Same play cursor

### Below the canvas
- Time axis (00:00, 00:05, …)
- That's it. No region cards, no chip list. Regions exist only as overlays on the spectrogram and as painted runs on the latent strip; there's no separate UI list to manage them.

### Interactions (same in both modes)
- Click on empty time + drag → create region
- Click on region body + drag → move
- Click on region edge handle + drag → resize
- Scroll → zoom time
- Cmd/space + drag → pan

---

## 7. Regions — purely visual

A region = any **contiguous run of regenerate-flagged latents**. Not a separate UI element, not a data structure, not a list. Just paint on the latent strip + an overlay on the spectrogram that follows the paint.

- Edit by painting. Split = preserve-flag a latent inside; merge = regenerate-flag the gap.
- No region identity, no region list, no card chips. The mask is the truth; the visual rendering follows.
- Per-region prompts: deferred to v2 (see §12.6 roadmap; will likely replace the global prompt panel with hover-select-to-edit interaction).

---

## 8. Right rail — controls

Collapsible sections, top to bottom. Header style matches the mock (chevron right-aligned, section title left, optional info `ⓘ`).

### Prompt
- Big textarea (global)
- Char counter (`31 / 500`) bottom-left; `Clear` link bottom-right
- Negative prompt collapsed by default

### Generation Settings (real SA3 params only)
- **Model**: dropdown (Medium [ARC, default] · Medium-base [RF])
- **Steps**: slider 1–32, default **8** (ARC-distilled medium runs cleanly here; medium-base wants more)
- **CFG scale (Guidance)**: slider 1.0–10.0, default 1.0 — **interpolates**
- **Seed**: numeric input + dice icon to randomize
- **Noise level** (only visible when source loaded — audio-to-audio): slider 0.0–1.0, default 1.0 — maps to `init_noise_level`. **Interpolates** (the killer slider).
- **Duration**: slider in seconds. Defaults to source duration when source loaded; user-set when generating from scratch.

### Source mode (implicit)
- If **no source loaded**: top of rail shows "Load audio…" drop zone + "Or generate from scratch" hint. Same Inpaint CTA → behaves as text-to-audio.
- If **source loaded**: source filename + duration + sample rate shown above Prompt; replace/remove options.

### LoRAs
- Header: `LoRAs ⓘ` left; `+` and `−` icons right
- `+` opens a dropdown of `.safetensors` files in the configured LoRA directory; select to add to active stack
- `−` shortcut to remove the most-recent LoRA (also `X` on each card)
- Each active LoRA card:
  - LoRA name top-left, `X` top-right
  - Strength slider 0.0–1.0 (default 0.7-ish) with current value displayed — **interpolates**
- Multi-stack supported; SA3 handles per-`lora_index`. Hot-reload on file drop.

### Generate / Inpaint CTA
The button has one job — produce a result — and its label/behavior depends on current state:

| Source loaded? | Mask painted? | Noise > 0? | Button label | Action |
|---|---|---|---|---|
| No | — | — | **Generate** | Text-to-audio from scratch |
| Yes | Yes | — | **Inpaint** | Regenerate painted latents using current prompt + source elsewhere |
| Yes | No | > 0 | **Vary** | Audio-to-audio with current noise level |
| Yes | No | = 0 | (button hidden) | Would produce identity — show hint "paint latents, or raise noise" |

- Small circular **Reroll** button next to the CTA (same settings, new seed)
- Below: **Variant navigator** `< N / M >` — flips through generations in this session. Each Inpaint/Vary/Generate/Reroll adds a slot.
- **Variant flip restores the full state at gen time** — mask, prompt, settings, source. Variants are snapshots of the moment they were created, not just alternate audio.

---

## 9. Bottom bar

Left → right:

- Transport: loop · prev-marker · play/pause · stop · next-marker
- Volume slider (small)
- Time display: `00:12.428 / 01:05.237` (mono, fixed-width numeric font)
- Source metadata small grey: `44.1 kHz · 24-bit · Stereo`
- Vis toggle: pill-button with two states (Spectrogram | Waveform), single visible control showing the *currently active* mode
- Status badge: `● Model loaded` (green dot) or `◌ Loading…` (spinner)
- System resources: `CPU 18% · CUDA 32% · VRAM 2.1/8 GB · RAM 6.3/32 GB` with mini bars under each. CUDA only shows if a CUDA device is present (will be present on the desktop, hidden on Mac).

---

## 10. Live-interpolation viz (the killer pattern)

Continuous controls pre-compute N intermediate states and the slider scrubs through them with per-pixel blending.

### Mechanism
1. On session freeze (track loaded, regions committed, prompt set), background-decode AE at slider positions `[0.0, 0.1, …, 1.0]` for each interpolating slider.
2. Frontend keeps the 11 frames as canvas textures.
3. Slider drag → linear interpolate between adjacent frames per pixel.
4. Slider release (optional) → precise re-decode at exact value replaces lerped preview.

### Sliders that get it
- **Variation Strength** (audio-to-audio noise level) — clean source → uniform noise palette
- **Guidance** — low → high CFG preview
- **LoRA strength** (per LoRA) — base → trained style
- **Steps** — preview intermediate diffusion states

### Cascading
The slider changes propagate to **every visualization in lockstep**: spectrogram, waveform mode, latent strip. Not "one knob → one number"; "the knob warms the whole UI."

### Cost budget
- Pre-compute: 11 frames × ~5s AE decode = ~55s per interpolating slider per session, background.
- Cache by `(source_hash, mask_hash, prompt_hash, slider_id)` so unchanged edits skip decode.
- ? Limit pre-compute concurrency so loading multiple LoRAs doesn't queue 4× decodes.

---

## 11. Keyboard

| Key | Action |
|---|---|
| Space | Play/pause |
| Cmd+G | Generate / Inpaint (CTA) |
| Cmd+Z / Cmd+Shift+Z | Undo / Redo |
| R | Reroll (same as CTA's reroll button) |
| Backspace | Clear selection (latent → preserve; region → delete) |
| Alt+drag | Invert paint (preserve ↔ regenerate) |
| Shift+click / drag | Add to selection |
| Cmd+drag / Space+drag | Pan |
| Scroll | Zoom time |
| Cmd+A | Select all regions |
| Cmd+D | Deselect |
| ← / → | Previous / next variant |
| Cmd+N | New (reset session) |
| Cmd+Shift+K | Clear current state (keep source) |

---

## 12. Design tokens

### Color palette (locked)
Pure black background, slight elevation for panels, visible borders. Maximum contrast with accents and content.
```css
--bg-dark:           #000000;   /* canvas, app background — true black */
--bg-lighter:        #0a0a0a;   /* panel surfaces, right rail */
--border-color:      #1c1c1c;   /* panel borders, dividers */
--code-highlight:    #1f1f1f;   /* hover surface */
--code-block:        #050505;   /* nested surfaces */
--scrollbar-active:  #333333;
--scrollbar-inactive:#2a2a2a;
--text-primary:      #ffffff;   /* pure white for headings, values, active state */
--text-secondary:    #888888;
--text-muted:        #555555;
--accent-blue:       #0078ca;   /* CTAs, sliders, region overlays, focus */
--accent-blue-dim:   #005a9e;   /* hover-down state of accent */
--error-red:         #f14c4c;
--success-green:     #4ec9b0;   /* "Model loaded" badge */
--warning-yellow:    #cca700;
```

### Visualizations (locked)
- Spectrogram colormap: **magma**
- Waveform RGB: **3-band, bass=R / mid=G / treble=B**
- Latent strip color: **PCA-3 of source latents → RGB** (preserve), **accent-blue outline** (regenerate)

### Typography
- System font stack for UI text
- Mono numerics for time displays and slider values (system mono is fine)
- Sizes: 11 (status/meta), 13 (body/labels), 15 (headers), 18 (large header)

### Spacing scale
- 4 / 8 / 12 / 16 / 24 / 32 px

### Component conventions
- Slider track: thin (~2px) on `--border-color`, fill in `--accent-blue`, white circle handle ~10px
- Button (primary CTA): `--accent-blue` background, white text, hover → `--accent-blue-dim`
- Region overlay on spectrogram: `--accent-blue` at ~15% alpha + 1px dashed white border
- Region paint on latent strip: bars outlined in `--accent-blue`
- Focus ring: 1px `--accent-blue` outline, no glow
- No glow, no gradient backgrounds, no rounded-rect-with-shadow chrome. Subtle wins.

---

## 12.5 Persistence — explicit save only

- In-memory by default. Closing the tab discards state.
- `Cmd+S` opens a directory picker; writes `source.wav`, `mask.json`, `prompt.txt`, `settings.json`, and `variants/N.wav` (one per slot).
- Reopening that directory via `New → Open session` restores everything including the variant navigator.
- No autosave, no automatic session manager listing.

## 12.6 Roadmap (pro features deferred)

- **Async per-region prompts** — bigger v2 thing. Replaces the global prompt panel entirely. Interaction: select a painted region (hover-based), type into a contextual input that floats near it, and the region async-inpaints as you type. No "Generate" button per region — the typing is the trigger. Removes the right-rail prompt section. Probably cleaner overall.
- **Inpaint with frequency masking** (requires model-side support that doesn't exist yet)
- **Multi-track / arrangement view**

## 12.7 Generation lifecycle

- Clicking the CTA → whole editor enters a "generating" state:
  - A spinner overlays the main canvas (darkened underneath)
  - Transport, paint, sliders all disabled
  - Right rail stays visible but inert
  - The streaming preview (if enabled, §15) updates the spectrogram and latent strip *under* the spinner — so the user sees the result form before the spinner clears
- Generation completes → spinner fades, controls re-enable, new variant slot is created and selected
- Clicking CTA again while generating: blocked (button looks inert). Generation is fast (a few seconds); no real need for cancellation.

## 13. Out of scope for v1

- Multi-track arrangement view
- MIDI in/out
- Real-time mic input
- Plugins / VSTs
- Cloud / collaboration
- Mobile / touch
- Frequency-bounded selection (SA3 doesn't support natively)

---

## 14. Stack

### Frontend — vanilla, no framework
- One HTML file, small CSS file, a handful of JS modules
- Bootstrap icons (icon library only, not the framework) via downloaded sprite
- Canvas 2D for all the visualizations (latent strip, overview, spectrogram, waveform)
- System fonts; mono numerics for time/values
- Dark theme, actually dark — not slate-grey

### Backend — minimal Python
- FastAPI + WebSocket on a single port; serves the static frontend and exposes the generation/preview API
- MLX AE for decode (already built — `mlx_sa3/`)
- DIT on torch+MPS on Mac, torch+CUDA on desktop. Full MLX-DIT port deferred.
- Per-session state: source audio, latent mask, regions (derived), global prompt, settings, slider-strip cache, variant history
- Generation queue: one active job per session, cancel-on-new

## 15. Inpaint animation

Inpainting is a 4-second visual moment, not just "click → wait → done." Two streams of feedback:

### Cheap (always on)
- **Indeterminate progress** at the top of the canvas (a thin blue line, no spinner — subtler)
- **Regenerate latents desaturate** to seed noise the moment generation starts; the spectrogram regions go translucent grey

### Streaming (if rendering fast enough)
- Hook into `sample_diffusion`'s step loop and emit current latents per step over WebSocket
- Decode each step's latents through MLX AE (cheap — 5s for full 3min track)
- Animate:
  - Latent strip regenerate bars: morph from seed-noise PCA color toward final PCA color
  - Spectrogram: regenerate regions sharpen as denoise progresses (blur radius → 0, opacity 1.0)
  - Waveform: regenerate regions go from low-amplitude grey to final amplitude + color
- Easing: linear or ease-out; per-frame transition between adjacent steps
- 8 steps × ~0.5s/step = 4s of animation feels right

### Costs
- AE decode per step on Mac MLX is ~5s for 3min, ~0.2s for 4s. For a typical inpaint region of a few seconds, decoding per-step is real-time.
- If decoding lags the DIT, emit every other step instead of every step.
- For large tracks, decode only the regenerate-region latents per step, not the full track.

---

## Open questions

1. **"Context Length"** — what model parameter does this map to? Inpaint context window? duration_padding_sec? A new abstraction we want to expose?
2. **Steps default** — mock shows 40, ARC-distilled medium runs fine at 8. Default 8 with a note, or 16, or what?
3. **Per-region prompts** — in v1 or deferred?
4. **Region identity on split/merge** — clone prompt to both halves on split? Keep longer side's prompt on merge?
5. **Beat grid / BPM-aware snap** — required, optional, skip?
6. **LoRA directory** — configurable per-session, or one global dir?
7. **Variant history depth** — unlimited / last N / configurable?
