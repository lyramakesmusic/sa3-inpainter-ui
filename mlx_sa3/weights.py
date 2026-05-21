"""Load SA3-medium AE weights from the safetensors checkpoint into our MLX modules.

Maps every `pretransform.*` key in the official safetensors to the MLX attribute path
in SA3MediumAE. Handles:
  - WNConv1d weight_g/weight_v fold (mapping layers)
  - kernel-1 conv -> Linear (drop the trailing kernel dim)
  - list-indexed transformer blocks
"""
from __future__ import annotations

import math
import os
from typing import Dict

import mlx.core as mx
import numpy as np
from safetensors import safe_open


SAFETENSORS_PATH = os.environ.get(
    "SA3_MEDIUM_SAFETENSORS",
    "/Users/lyra/Projects/stable-audio-3/models/stable-audio-3-medium/model.safetensors",
)


def _wnconv1d_fold(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Fold weight-norm parameters into a single conv weight.

    weight_g: (out, 1, 1)  -- per-output-channel scalar
    weight_v: (out, in, k)
    Returns: (out, in, k) ready to drop into a Conv1d / (out, in) for kernel=1 Linear.
    """
    norm = np.sqrt((weight_v ** 2).sum(axis=(1, 2), keepdims=True))  # (out, 1, 1)
    return weight_v * (weight_g / np.clip(norm, 1e-12, None))


def _conv1d_to_linear(weight_oik: np.ndarray) -> np.ndarray:
    """k=1 conv weight (out, in, 1) -> Linear weight (out, in)."""
    assert weight_oik.shape[-1] == 1, f"expected k=1, got shape {weight_oik.shape}"
    return weight_oik[..., 0]


def load_ae_weights(model, safetensors_path: str = SAFETENSORS_PATH) -> Dict[str, mx.array]:
    """Load weights from sa3-medium safetensors into the given SA3MediumAE.

    Modifies `model` in place via `model.update(parameter_tree)` and returns the parameter tree.
    """
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("pretransform.")]
        raw: Dict[str, np.ndarray] = {}
        for k in keys:
            t = f.get_tensor(k)
            raw[k] = t.detach().cpu().numpy()

    params: dict = {
        "bottleneck": {},
        "encoder": {"layers_0": {"mapping": {}, "transformers": [{} for _ in range(12)]}, "layers_2": {}},
        "decoder": {"layers_1": {}, "layers_3": {"mapping": {}, "transformers": [{} for _ in range(12)]}},
        "pretransform": {},
    }

    # ---------- bottleneck ----------
    bn_root = "pretransform.model.bottleneck"
    params["bottleneck"]["scaling_factor"] = mx.array(raw[f"{bn_root}.scaling_factor"])
    params["bottleneck"]["bias"] = mx.array(raw[f"{bn_root}.bias"])
    params["bottleneck"]["noise_scaling_factor"] = mx.array(raw[f"{bn_root}.noise_scaling_factor"])
    params["bottleneck"]["running_std"] = mx.array(raw[f"{bn_root}.running_std"])

    # ---------- decoder.layers.1 (Linear) ----------
    params["decoder"]["layers_1"]["weight"] = mx.array(raw["pretransform.model.decoder.layers.1.weight"])
    params["decoder"]["layers_1"]["bias"] = mx.array(raw["pretransform.model.decoder.layers.1.bias"])

    # ---------- decoder.layers.3 (TRB decoder) ----------
    d3_root = "pretransform.model.decoder.layers.3"
    # mapping: WNConv1d (1536 -> 512, k=1) -> Linear
    map_w = _wnconv1d_fold(raw[f"{d3_root}.mapping.weight_g"], raw[f"{d3_root}.mapping.weight_v"])
    params["decoder"]["layers_3"]["mapping"]["weight"] = mx.array(_conv1d_to_linear(map_w))
    params["decoder"]["layers_3"]["mapping"]["bias"] = mx.array(raw[f"{d3_root}.mapping.bias"])
    # new_tokens
    params["decoder"]["layers_3"]["new_tokens"] = mx.array(raw[f"{d3_root}.new_tokens"])
    # 12 transformer blocks
    for i in range(12):
        t_root = f"{d3_root}.transformers.{i}"
        tb = params["decoder"]["layers_3"]["transformers"][i]
        # DyT norms (pre_norm, ff_norm, self_attn.{q,k}_norm)
        for norm_key in ("pre_norm", "ff_norm"):
            tb[norm_key] = {
                "alpha": mx.array(raw[f"{t_root}.{norm_key}.alpha"]),
                "beta": mx.array(raw[f"{t_root}.{norm_key}.beta"]),
                "gamma": mx.array(raw[f"{t_root}.{norm_key}.gamma"]),
            }
        sa = {}
        for norm_key in ("q_norm", "k_norm"):
            sa[norm_key] = {
                "alpha": mx.array(raw[f"{t_root}.self_attn.{norm_key}.alpha"]),
                "beta": mx.array(raw[f"{t_root}.self_attn.{norm_key}.beta"]),
                "gamma": mx.array(raw[f"{t_root}.self_attn.{norm_key}.gamma"]),
            }
        sa["to_qkv"] = {"weight": mx.array(raw[f"{t_root}.self_attn.to_qkv.weight"])}
        sa["to_out"] = {"weight": mx.array(raw[f"{t_root}.self_attn.to_out.weight"])}
        tb["self_attn"] = sa
        tb["ff"] = {
            "ff_0": {
                "proj": {
                    "weight": mx.array(raw[f"{t_root}.ff.ff.0.proj.weight"]),
                    "bias": mx.array(raw[f"{t_root}.ff.ff.0.proj.bias"]),
                }
            },
            "ff_2": {
                "weight": mx.array(raw[f"{t_root}.ff.ff.2.weight"]),
                "bias": mx.array(raw[f"{t_root}.ff.ff.2.bias"]),
            },
        }
        tb["rope"] = {"inv_freq": mx.array(raw[f"{t_root}.rope.inv_freq"])}

    # ---------- encoder.layers.0 (TRB encoder) ----------
    e0_root = "pretransform.model.encoder.layers.0"
    if f"{e0_root}.mapping.weight_v" in raw:
        emap_w = _wnconv1d_fold(raw[f"{e0_root}.mapping.weight_g"], raw[f"{e0_root}.mapping.weight_v"])
        params["encoder"]["layers_0"]["mapping"]["weight"] = mx.array(_conv1d_to_linear(emap_w))
        params["encoder"]["layers_0"]["mapping"]["bias"] = mx.array(raw[f"{e0_root}.mapping.bias"])
    params["encoder"]["layers_0"]["new_tokens"] = mx.array(raw[f"{e0_root}.new_tokens"])
    for i in range(12):
        t_root = f"{e0_root}.transformers.{i}"
        tb = params["encoder"]["layers_0"]["transformers"][i]
        for norm_key in ("pre_norm", "ff_norm"):
            tb[norm_key] = {
                "alpha": mx.array(raw[f"{t_root}.{norm_key}.alpha"]),
                "beta": mx.array(raw[f"{t_root}.{norm_key}.beta"]),
                "gamma": mx.array(raw[f"{t_root}.{norm_key}.gamma"]),
            }
        sa = {}
        for norm_key in ("q_norm", "k_norm"):
            sa[norm_key] = {
                "alpha": mx.array(raw[f"{t_root}.self_attn.{norm_key}.alpha"]),
                "beta": mx.array(raw[f"{t_root}.self_attn.{norm_key}.beta"]),
                "gamma": mx.array(raw[f"{t_root}.self_attn.{norm_key}.gamma"]),
            }
        sa["to_qkv"] = {"weight": mx.array(raw[f"{t_root}.self_attn.to_qkv.weight"])}
        sa["to_out"] = {"weight": mx.array(raw[f"{t_root}.self_attn.to_out.weight"])}
        tb["self_attn"] = sa
        tb["ff"] = {
            "ff_0": {
                "proj": {
                    "weight": mx.array(raw[f"{t_root}.ff.ff.0.proj.weight"]),
                    "bias": mx.array(raw[f"{t_root}.ff.ff.0.proj.bias"]),
                }
            },
            "ff_2": {
                "weight": mx.array(raw[f"{t_root}.ff.ff.2.weight"]),
                "bias": mx.array(raw[f"{t_root}.ff.ff.2.bias"]),
            },
        }
        tb["rope"] = {"inv_freq": mx.array(raw[f"{t_root}.rope.inv_freq"])}

    # encoder.layers.2 (Linear out)
    e2_root = "pretransform.model.encoder.layers.2"
    if f"{e2_root}.weight" in raw:
        params["encoder"]["layers_2"]["weight"] = mx.array(raw[f"{e2_root}.weight"])
        params["encoder"]["layers_2"]["bias"] = mx.array(raw[f"{e2_root}.bias"])

    model.update(params)
    return params
