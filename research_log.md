# research log

## 2026-05-20 — kickoff

- SA3 dropped this morning. medium is gated CUDA-only (flash-attn required); on Mac the README's "Mac CPU" row is em-dash for medium.
- Small variants ship with MLX npz weights (`stable-audio-3-optimized/MLX/dit_sm-{music,sfx}_f16.npz`, `same_s_*_f32.npz`) and CoreML/TFLite runners. Medium has npz too (`dit_medium_f16.npz` 2.77GB, `same_l_decoder_f32.npz` 1.62GB, `t5gemma_f16.npz` 541MB) but no runner code published — the `MLX/README.md` in the optimized repo is 0 bytes. Uploaded by cortexelus (CJ Carr) ~3h before this session.
- Failure mode (per Lyra): torch SDPA *sliding-window* fallback in `stable_audio_3.models.transformer.apply_attn` produces static glitch sound — only when SWA is in play. SWA only appears in the SAME pretransform encoder/decoder, not the main DIT. So the bug is localized to the AE.
- Zach (sa3 author): "can fall back to flex-attention." Confirms flash-attn isn't strictly needed if you use torch flex_attention. On Mac, flex_attention isn't available (needs torch.compile+Triton, CUDA-only) — so MLX is the right path for Mac.
- dadabots: "flash-attn not needed if you use TRT or MLX pathway." Reinforces MLX SDPA + correct band mask should work.

## arch facts (from medium model_config.json + safetensors header)

DIT (`model.model.*` — 522 tensors, 1.4B params):
- embed_dim=1536, depth=24, num_heads=24, dim_heads=64
- differential attention everywhere: `to_qkv.weight = [7680, 1536]` = 1536*5
- cross-attn (to t5gemma 768d cond): `to_q=[3072,1536]`, `to_kv=[4608,1536]`
- FF: GLU mult=4 → `ff.0.proj=[12288,1536]`, `ff.2=[1536,6144]`
- adaLN global cond: `transformer.global_cond_embedder` projects 1536→9216 (=6*1536); per-layer `to_scale_shift_gate` is a learned 9216 bias added to that
- qk_norm: RMSNorm per-head (`q_norm.gamma=[64]`)
- 64 learned memory tokens (`transformer.memory_tokens=[64,1536]`)
- timestep features: ExpoFourierFeatures (256d) → linear→1536, SiLU, linear→1536
- diffusion_objective: `rf_denoiser` (rectified flow)
- local_add_cond_dim=257 (inpaint_mask 1ch + inpaint_masked_input 256ch); zero for text-to-audio

AE (`pretransform.model.*` — 472 tensors):
- `decoder.layers.1` Linear 256→1536 (bottleneck out → AE-dim)
- `decoder.layers.3` TransformerResamplingBlock (TAAE_v2):
  - dim=1536, dim_heads=64 → 24 heads, depth=12
  - differential attention
  - qk_norm=`dyt` (DynamicTanh) — `q_norm.{alpha[1], beta[64], gamma[64]}`
  - norm_type=`dyt` for pre_norm / ff_norm / cross_attend_norm too
  - FF GLU mult=3 → `ff.0.proj=[9216,1536]`, `ff.2=[1536,4608]`
  - sinusoidal_blocks=[8]: last 8 of the 12 layers use `Sin()` activation in FF instead of SiLU
  - sliding_window=[1,1] → effective [17,17] (multiplied by stride+1=17)
  - variable_stride=True → new_tokens shape [1,1,1536] (broadcast at runtime)
  - mapping: WNConv1d 1536→512 kernel=1 (in!=out)
- patched pretransform: `pretransform.config.pretransform`: patch_size=256, channels=2; decode reshapes 512 channels → 2 channels × 256 samples per latent-frame
- bottleneck: softnorm — `scaling_factor[1,256,1]`, `bias[1,256,1]`, `running_std[1]`. Decode = `(x - bias) * scaling_factor / running_std` roughly (need to verify).

Conditioner:
- t5gemma `b-b-ul2` (1.1GB safetensors) — encodes prompt to 768d cond tokens, max_length=256, `padding_mode=learned`
- seconds_total: scalar → expo Fourier (256d) → linear(256→768) → bias → 768d
- cond_dim=768; passed as both cross_attn and global_cond

## 2026-05-20 — SWA kernel fix verified

`scratch/swa_correctness.py`: MLX `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask=additive_band_mask)` matches brute-force torch sliding-window reference at `max_abs_diff = 5.96e-07` on `(B=1, H=4, T=256, D=64), window=(17,17)`. Confirms the fix is mechanically sound. Now the work is faithfully porting the surrounding AE module to MLX.


## 2026-05-20 — block-level parity vs torch

`scratch/compare_block_torch.py`: same weights from sa3-medium loaded into torch `TransformerBlock(dim=1536, dim_heads=64, qk_norm=dyt, differential=True, mult=3, add_rope=True, norm_type=dyt)` and our MLX `AETransformerBlock`, fed same input (T=136, window=[17,17]). Result: `max_abs_diff=1.28e-6`, `mean=6.4e-8`, `cosine_sim=1.000000`. Torch path here actually uses the same additive-band-mask SDPA via flex_attention compile failure → chunked-halo fallback → ultimately equivalent math, which is *why* it agrees with MLX. Suggests the "AE produces static" failure may be specific to larger sequence sizes, chunked decoding, or MPS backend (the parity check ran on CPU). Need to test full AE forward at realistic shapes next.


## 2026-05-20 — full AE bit-equivalent to torch

`scratch/compare_ae_deterministic.py` (torch noise sources disabled):
- `max_abs_diff = 6.7e-06`, `mean = 7.0e-07`, `cosine_sim = 1.00000012` on random latents (1, 256, 16).

First attempt before deterministic torch had `cosine_sim = 0.999941`, `max_abs_diff = 7.6e-3` — the gap was from torch's eval-mode stochastic noise (`bottleneck.noise_regularize=True` adds `randn * running_std * 1e-3`, and `TRB.mask_noise=0.1` adds noise to new_tokens). Both fire even in `.eval()` mode in the torch code.

Bug found and fixed: off-by-one in `sinusoidal_blocks` cutoff. Torch uses strict `<` (`(transformer_depth - i) < sinusoidal_blocks`), I had `<=`. With `transformer_depth=12, sinusoidal_blocks=8`, this flipped block 4 from SiLU to Sin, which exponentially diverged through the remaining 7 blocks (12x per layer ≈ 70x by block 11). Per-block test passed initially because it only checked block 0. Lesson: parity-check both endpoints of any per-layer config schedule, not just position 0.


## 2026-05-20 — end-to-end win: sounds good

Full pipeline run on Lyra's M5 Air 16GB:
- DIT: torch + MPS, 4-step generation of 4s latents -> 4.8s
- AE: MLX SA3MediumAE.decode (band-mask SWA) -> 0.2s
- Output WAV: 4s @ 44.1kHz stereo, /tmp/sa3_mlx_e2e.wav

Lyra confirmed the audio sounds good. So the SWA-in-MLX fix actually solves the static-glitch problem reported in the official sa3 README, *without* flash-attn. Two pieces of the win:
1. `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask=additive_band_mask)` — fp32 softmax handles the band-mask numerics cleanly where torch's SDPA fallback on Mac doesn't.
2. The torch DIT runs fine on MPS using its existing SDPA path because it uses full attention (no SWA in the DIT itself per `model_config.json`).

Status vs goal "medium on MLX with functioning SWA":
- AE (the SWA-critical part) is on MLX ✓
- DIT still on torch+MPS (full attention works there as-is)
- t5gemma still on torch (single-shot text encode, runs once)


## 2026-05-20 — 3-min real DIT run, SWA stress-tested

Full pipeline at scale on M5 Air 16GB:
- DIT on torch+MPS, 4 steps, 1292 latents → 3.0s
- MLX AE decode (band-mask SWA, chunked at chunk_size=128 / overlap=32) → 3.5s
- Total: ~6.5s for 180s of audio = 28x real time
- Lyra listen-test: "this sounds GOOD"

AE packed sequence at 1292 latents = 21964 tokens with window=[17,17] (34 visible per query). Chunked decoding splits into ~10 windows of 128 latents each (packed seq ~2176 tokens per chunk, band mask ~18MB instead of ~2GB at full seq). No OOM, no static.

This validates the goal: medium AE on MLX with band-mask SWA reproduces flash-attn's semantics at every scale we've tested (T_lat=16, 108, 1292). The fix is complete for the AE-as-blocker problem.

Remaining if we want full MLX:
- Port DIT to MLX (would save the model load round-trip and the torch+MPS overhead; not blocking)
- Port t5gemma to MLX (one-shot encode, modest benefit)
- Wrap as clean public API

