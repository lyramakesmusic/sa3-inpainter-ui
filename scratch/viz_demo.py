"""Generate 10s of random-latent audio through the MLX SA3 AE, then emit two
chromeless visualizations:
  1. minimeters-style waveform with frequency-balance RGB coloring  (20" x 2")
  2. log-Hz spectrogram, magma colormap                              (20" x 4")

No axes, no legends, no margins — just the picture.
"""
import sys
sys.path.insert(0, "/Users/lyra/Projects/sa3-mlx")

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import stft

CKPT = "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors"
SR = 44100
DPI = 100


def decode_random(duration: float = 10.0, seed: int = 0):
    from mlx_sa3.ae import SA3MediumAE
    from mlx_sa3.weights import load_ae_weights
    ae = SA3MediumAE()
    load_ae_weights(ae, CKPT)
    T_lat = int(duration * SR / 4096) + 1
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((1, 256, T_lat)).astype(np.float32) * 0.3
    wav = ae.decode(mx.array(lat))
    mx.eval(wav)
    arr = np.array(wav)[0]               # (2, T)
    return arr[:, : int(duration * SR)]  # truncate to exact duration


def waveform_freq_colored(audio: np.ndarray, out_path: str,
                          width_in: float = 20.0, height_in: float = 2.0,
                          bands=((0, 250), (250, 2500), (2500, 22050))):
    """audio: (2, T) stereo or (T,) mono. Renders mono mixdown vertically with
    per-frame RGB color = normalized energy in (low, mid, high) bands.

    The waveform is drawn as a vertical stem at each column pixel from -amp..+amp.
    """
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    n_samples = mono.shape[0]
    width_px = int(width_in * DPI)
    height_px = int(height_in * DPI)

    # STFT for color
    f, t, Z = stft(mono, fs=SR, nperseg=2048, noverlap=2048 - 512, boundary=None, padded=False)
    P = np.abs(Z) ** 2  # (n_freqs, n_frames)
    # 3-band energy
    bins = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        bins.append(P[mask].sum(axis=0))
    band_E = np.stack(bins, axis=0)              # (3, n_frames) low/mid/high
    band_E = band_E / (band_E.sum(axis=0, keepdims=True) + 1e-12)
    # boost saturation: small gamma so dominant band shines
    rgb = (band_E.T ** 0.6)                       # (n_frames, 3)
    rgb = rgb / (rgb.max(axis=-1, keepdims=True) + 1e-12)
    # map color from STFT frame indices to columns
    frame_centers = (t * SR).astype(int).clip(0, n_samples - 1)

    # Build the column-level color and amplitude arrays
    col_to_sample = np.linspace(0, n_samples - 1, width_px).astype(int)
    # for each column, sample = mono[col_to_sample[i]] (single point) — but better to take
    # peak/envelope of the samples in this column's window:
    bins_per_col = max(1, n_samples // width_px)
    # peak per column
    peaks = np.zeros(width_px, dtype=np.float32)
    for i in range(width_px):
        s = i * bins_per_col
        e = min(n_samples, s + bins_per_col)
        if e > s:
            peaks[i] = np.max(np.abs(mono[s:e]))
    peaks /= peaks.max() + 1e-12

    # color per column: pick nearest STFT frame
    col_to_frame = np.searchsorted(frame_centers, col_to_sample).clip(0, len(frame_centers) - 1)
    col_rgb = rgb[col_to_frame]  # (width_px, 3)

    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])  # no margins, no axes chrome
    ax.set_facecolor("black")
    ax.set_xlim(0, width_px)
    ax.set_ylim(-1.05, 1.05)
    ax.set_axis_off()

    # draw vertical lines (vectorized via vlines)
    x = np.arange(width_px)
    ymin = -peaks
    ymax = peaks
    # plot in chunks because matplotlib LineCollection is faster for many segments
    from matplotlib.collections import LineCollection
    segs = np.zeros((width_px, 2, 2))
    segs[:, 0, 0] = x; segs[:, 1, 0] = x
    segs[:, 0, 1] = ymin; segs[:, 1, 1] = ymax
    lc = LineCollection(segs, colors=col_rgb, linewidths=1.0, antialiased=True)
    ax.add_collection(lc)
    fig.savefig(out_path, dpi=DPI, facecolor="black")
    plt.close(fig)
    print(f"wrote {out_path}  ({width_px}x{height_px})")


def spectrogram_logHz(audio: np.ndarray, out_path: str,
                      width_in: float = 40.0, height_in: float = 8.0,
                      f_min: float = 30.0, f_max: float = 16000.0,
                      dpi: int = 100,
                      noise_floor_db: float = -55.0,
                      gamma: float = 0.55,
                      cmap: str = "magma"):
    """Sharper, more contrasty spectrogram for UI use.

    `noise_floor_db` clips the low-dynamic-range muck (lower = more shows through).
    `gamma` < 1 punches contrast (smaller = more contrast).
    Bigger figsize → finer detail when downscaled by the browser.
    """
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    # bigger FFT = finer frequency resolution → crisper harmonic stacks
    n_fft = 8192
    hop = 256
    f, t, Z = stft(mono, fs=SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Z) ** 2

    # convert to dB, clip the noise floor, normalize 0..1, gamma-curve for contrast punch
    P_db = 10.0 * np.log10(P + 1e-12)
    P_db = np.clip(P_db, noise_floor_db, P_db.max())
    P_db -= P_db.min()
    P_db /= P_db.max() + 1e-12
    P_db = P_db ** gamma

    out_height_px = int(height_in * dpi)
    log_f = np.geomspace(f_min, f_max, out_height_px)
    spec_log = np.zeros((out_height_px, P_db.shape[1]), dtype=np.float32)
    for j in range(P_db.shape[1]):
        spec_log[:, j] = np.interp(log_f, f, P_db[:, j])

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(
        spec_log[::-1],
        aspect="auto",
        origin="upper",
        cmap=cmap,
        interpolation="nearest",       # sharp; no browser-side blur either
        extent=(0, 1, 0, 1),
    )
    fig.savefig(out_path, dpi=dpi, facecolor="black")
    plt.close(fig)
    print(f"wrote {out_path}  ({int(width_in*dpi)}x{int(height_in*dpi)})")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wf", default="/tmp/sa3_viz_waveform.png")
    p.add_argument("--sg", default="/tmp/sa3_viz_spectrogram.png")
    args = p.parse_args()

    print(f"decoding {args.duration}s of random latents...")
    audio = decode_random(duration=args.duration, seed=args.seed)
    print(f"audio shape: {audio.shape}, std={audio.std():.3f} peak={np.abs(audio).max():.3f}")

    waveform_freq_colored(audio, args.wf)
    spectrogram_logHz(audio, args.sg)


if __name__ == "__main__":
    main()
