# Goal

Run **Stable Audio 3 Medium** end-to-end on Apple Silicon, with **working sliding-window attention** in the SAME-L autoencoder decoder. Drop the flash-attn dependency entirely.

## Why this is the right shape

- The medium DIT (diffusion transformer) uses *full* self-attention. It runs fine on torch+MPS using `F.scaled_dot_product_attention` once flash-attn is unavailable. No SWA there.
- The medium AE (SAME-L / TAAE_v2 decoder) uses **sliding-window attention** with window `[1,1]` × `(stride+1=17)` = effective `[17, 17]`, on a packed sequence of length ~69632 (latent_length × 17).
- The torch SDPA fallbacks for SWA (flex_attention, chunked-halo SDPA, full-masked SDPA) *produce static noise* on Mac. Confirmed by Lyra; matches the "Output audio is a static glitch sound" troubleshooting note in the official sa3 README.
- Internal tip: an MLX runner is being pushed by EOD; "it was easy to get running." So the architecture maps cleanly onto MLX with `mx.fast.scaled_dot_product_attention` (which always uses fp32 softmax and accepts arbitrary additive masks).

## Minimum deliverable

A function in this project that takes a torch tensor of medium latents on MPS and returns the decoded waveform tensor, with the AE decode running on MLX using an additive band-mask SDPA. The wrapper handles dtype/device conversion at the torch ↔ MLX boundary.

Stretch: full MLX inference path (DIT + AE + t5gemma + RF sampler), wired into `stable_audio_3.StableAudioModel`-shaped API.

## Done means

- No static. Generated audio is musically coherent on a sanity prompt.
- Matches torch reference (where torch reference is the official sa3 + flash-attn pipeline on Windows/Linux) within reasonable numerical tolerance.
- Runs on Lyra's M5 Air 16GB unified.
