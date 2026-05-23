"""MLX LoRA/DoRA/BoRA adapter for SA3 DiT.

Provides adapter wrappers, model injection, and weight save/load
compatible with PyTorch SA3 LoRA safetensors format.

Supported adapter types:
  - lora:       W + scaling * B @ A
  - dora-rows:  magnitude_r * normalize_rows(W + scaling * B @ A)
  - dora-cols:  magnitude_c * normalize_cols(W + scaling * B @ A)
  - bora:       magnitude_c * normalize_cols(magnitude_r * normalize_rows(W + scaling * B @ A))
"""
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# ── Adapter modules ──


class LoRALinear(nn.Module):
    """nn.Linear with low-rank adapter (LoRA)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        fan_out, fan_in = base.weight.shape
        self.weight = base.weight
        self.bias = base.bias if hasattr(base, "bias") and base.bias is not None else None
        self.lora_A = mx.random.normal((rank, fan_in)) * (1.0 / math.sqrt(fan_in))
        self.lora_B = mx.zeros((fan_out, rank))
        self.scale = alpha / rank

    def __call__(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return out


class DoRALinear(nn.Module):
    """nn.Linear with DoRA adapter (weight-decomposed LoRA).

    dora-rows: magnitude per output neuron (row), norm_dim=1
    dora-cols: magnitude per input feature (column), norm_dim=0
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0,
                 norm_dim: int = 1):
        super().__init__()
        fan_out, fan_in = base.weight.shape
        self.weight = base.weight
        self.bias = base.bias if hasattr(base, "bias") and base.bias is not None else None
        self.lora_A = mx.random.normal((rank, fan_in)) * (1.0 / math.sqrt(fan_in))
        self.lora_B = mx.zeros((fan_out, rank))
        self.scale = alpha / rank
        self.norm_dim = norm_dim
        w = base.weight.astype(mx.float32)
        mag = mx.sqrt(mx.sum(w * w, axis=norm_dim))
        self.magnitude = mag

    def __call__(self, x):
        w = self.weight.astype(self.lora_A.dtype)
        delta = self.lora_B @ self.lora_A
        V = w + self.scale * delta
        V_norm = mx.sqrt(mx.sum(V * V, axis=self.norm_dim, keepdims=True) + 1e-12)
        V_hat = V / V_norm
        if self.norm_dim == 1:
            W_out = V_hat * self.magnitude[:, None]
        else:
            W_out = V_hat * self.magnitude[None, :]
        out = x @ W_out.astype(x.dtype).T
        if self.bias is not None:
            out = out + self.bias
        return out


class BoRALinear(nn.Module):
    """nn.Linear with BoRA adapter (bidirectional DoRA — row + column normalization)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        fan_out, fan_in = base.weight.shape
        self.weight = base.weight
        self.bias = base.bias if hasattr(base, "bias") and base.bias is not None else None
        self.lora_A = mx.random.normal((rank, fan_in)) * (1.0 / math.sqrt(fan_in))
        self.lora_B = mx.zeros((fan_out, rank))
        self.scale = alpha / rank
        w = base.weight.astype(mx.float32)
        self.magnitude_r = mx.sqrt(mx.sum(w * w, axis=1))
        self.magnitude_c = mx.sqrt(mx.sum(w * w, axis=0))

    def __call__(self, x):
        w = self.weight.astype(self.lora_A.dtype)
        delta = self.lora_B @ self.lora_A
        V = w + self.scale * delta
        V_r = V / (mx.sqrt(mx.sum(V * V, axis=1, keepdims=True)) + 1e-12)
        intermediate = self.magnitude_r[:, None] * V_r
        H_c = intermediate / (mx.sqrt(mx.sum(intermediate * intermediate, axis=0, keepdims=True)) + 1e-12)
        W_out = H_c * self.magnitude_c[None, :]
        out = x @ W_out.astype(x.dtype).T
        if self.bias is not None:
            out = out + self.bias
        return out


class LoRAConv1d(nn.Module):
    """nn.Conv1d (kernel_size=1) with low-rank adapter."""

    def __init__(self, base: nn.Conv1d, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        out_ch, k, in_ch = base.weight.shape
        assert k == 1, "LoRA Conv1d only supports kernel_size=1"
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)
        self.padding = base.padding
        self.stride = base.stride
        self.lora_A = mx.random.normal((rank, in_ch)) * (1.0 / math.sqrt(in_ch))
        self.lora_B = mx.zeros((out_ch, rank))
        self.scale = alpha / rank

    def __call__(self, x):
        w = self.weight[:, 0, :]  # [out, in]
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        out = out + (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return out


class DoRAConv1d(nn.Module):
    """nn.Conv1d (kernel_size=1) with DoRA adapter."""

    def __init__(self, base: nn.Conv1d, rank: int = 8, alpha: float = 8.0,
                 norm_dim: int = 1):
        super().__init__()
        out_ch, k, in_ch = base.weight.shape
        assert k == 1, "DoRA Conv1d only supports kernel_size=1"
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)
        self.padding = base.padding
        self.stride = base.stride
        self.lora_A = mx.random.normal((rank, in_ch)) * (1.0 / math.sqrt(in_ch))
        self.lora_B = mx.zeros((out_ch, rank))
        self.scale = alpha / rank
        self.norm_dim = norm_dim
        w = base.weight[:, 0, :].astype(mx.float32)
        self.magnitude = mx.sqrt(mx.sum(w * w, axis=norm_dim))

    def __call__(self, x):
        w = self.weight[:, 0, :].astype(self.lora_A.dtype)
        delta = self.lora_B @ self.lora_A
        V = w + self.scale * delta
        V_norm = mx.sqrt(mx.sum(V * V, axis=self.norm_dim, keepdims=True) + 1e-12)
        V_hat = V / V_norm
        if self.norm_dim == 1:
            W_out = V_hat * self.magnitude[:, None]
        else:
            W_out = V_hat * self.magnitude[None, :]
        out = x @ W_out.astype(x.dtype).T
        if self.bias is not None:
            out = out + self.bias
        return out


class BoRAConv1d(nn.Module):
    """nn.Conv1d (kernel_size=1) with BoRA adapter."""

    def __init__(self, base: nn.Conv1d, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        out_ch, k, in_ch = base.weight.shape
        assert k == 1, "BoRA Conv1d only supports kernel_size=1"
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)
        self.padding = base.padding
        self.stride = base.stride
        self.lora_A = mx.random.normal((rank, in_ch)) * (1.0 / math.sqrt(in_ch))
        self.lora_B = mx.zeros((out_ch, rank))
        self.scale = alpha / rank
        w = base.weight[:, 0, :].astype(mx.float32)
        self.magnitude_r = mx.sqrt(mx.sum(w * w, axis=1))
        self.magnitude_c = mx.sqrt(mx.sum(w * w, axis=0))

    def __call__(self, x):
        w = self.weight[:, 0, :].astype(self.lora_A.dtype)
        delta = self.lora_B @ self.lora_A
        V = w + self.scale * delta
        V_r = V / (mx.sqrt(mx.sum(V * V, axis=1, keepdims=True)) + 1e-12)
        intermediate = self.magnitude_r[:, None] * V_r
        H_c = intermediate / (mx.sqrt(mx.sum(intermediate * intermediate, axis=0, keepdims=True)) + 1e-12)
        W_out = H_c * self.magnitude_c[None, :]
        out = x @ W_out.astype(x.dtype).T
        if self.bias is not None:
            out = out + self.bias
        return out


# ── All adapter module types ──

_ALL_LORA_TYPES = (LoRALinear, DoRALinear, BoRALinear,
                   LoRAConv1d, DoRAConv1d, BoRAConv1d)

_ADAPTER_LINEAR = {
    "lora": LoRALinear,
    "dora-rows": lambda base, rank, alpha: DoRALinear(base, rank, alpha, norm_dim=1),
    "dora": lambda base, rank, alpha: DoRALinear(base, rank, alpha, norm_dim=1),
    "dora-cols": lambda base, rank, alpha: DoRALinear(base, rank, alpha, norm_dim=0),
    "bora": BoRALinear,
}

_ADAPTER_CONV1D = {
    "lora": LoRAConv1d,
    "dora-rows": lambda base, rank, alpha: DoRAConv1d(base, rank, alpha, norm_dim=1),
    "dora": lambda base, rank, alpha: DoRAConv1d(base, rank, alpha, norm_dim=1),
    "dora-cols": lambda base, rank, alpha: DoRAConv1d(base, rank, alpha, norm_dim=0),
    "bora": BoRAConv1d,
}


# ── Injection ──


def _should_inject(path: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    if exclude:
        for pat in exclude:
            if pat in path:
                return False
    if include:
        for pat in include:
            if pat in path:
                return True
        return False
    return True


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    adapter_type: str = "lora",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> int:
    """Replace nn.Linear and nn.Conv1d modules with adapter variants.

    adapter_type: "lora", "dora-rows", "dora-cols", "bora"
    Returns the number of injected adapters.
    """
    make_linear = _ADAPTER_LINEAR.get(adapter_type)
    make_conv1d = _ADAPTER_CONV1D.get(adapter_type)
    if make_linear is None:
        raise ValueError(f"Unknown adapter type: {adapter_type}. "
                         f"Supported: {list(_ADAPTER_LINEAR.keys())}")

    count = 0

    def _inject_recursive(parent, parent_path: str):
        nonlocal count
        for name in list(vars(parent).keys()):
            child = getattr(parent, name)
            child_path = f"{parent_path}.{name}" if parent_path else name

            if isinstance(child, nn.Linear) and _should_inject(child_path, include, exclude):
                setattr(parent, name, make_linear(child, rank, alpha))
                count += 1
            elif isinstance(child, nn.Conv1d) and _should_inject(child_path, include, exclude):
                try:
                    setattr(parent, name, make_conv1d(child, rank, alpha))
                    count += 1
                except AssertionError:
                    pass
            elif isinstance(child, nn.Module):
                _inject_recursive(child, child_path)
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    item_path = f"{child_path}.{i}"
                    if isinstance(item, nn.Linear) and _should_inject(item_path, include, exclude):
                        child[i] = make_linear(item, rank, alpha)
                        count += 1
                    elif isinstance(item, nn.Conv1d) and _should_inject(item_path, include, exclude):
                        try:
                            child[i] = make_conv1d(item, rank, alpha)
                            count += 1
                        except AssertionError:
                            pass
                    elif isinstance(item, nn.Module):
                        _inject_recursive(item, item_path)

    _inject_recursive(model, "")
    return count


def freeze_base_unfreeze_lora(model: nn.Module):
    """Freeze all parameters, then unfreeze adapter weights."""
    model.freeze()
    lora_keys = ["lora_A", "lora_B", "magnitude", "magnitude_r", "magnitude_c"]
    model.apply_to_modules(
        lambda path, m: m.unfreeze(keys=lora_keys, strict=False)
        if isinstance(m, _ALL_LORA_TYPES) else None
    )


# ── State dict extraction ──


def get_lora_state_dict(model: nn.Module, prefix: str = "") -> dict[str, mx.array]:
    """Collect all adapter weights from the model tree."""
    result = {}

    def _collect_from_module(mod, path):
        if isinstance(mod, (LoRALinear, LoRAConv1d)):
            result[f"{path}.lora_A"] = mod.lora_A
            result[f"{path}.lora_B"] = mod.lora_B
        elif isinstance(mod, (DoRALinear, DoRAConv1d)):
            result[f"{path}.lora_A"] = mod.lora_A
            result[f"{path}.lora_B"] = mod.lora_B
            result[f"{path}.magnitude"] = mod.magnitude
        elif isinstance(mod, (BoRALinear, BoRAConv1d)):
            result[f"{path}.lora_A"] = mod.lora_A
            result[f"{path}.lora_B"] = mod.lora_B
            result[f"{path}.magnitude_r"] = mod.magnitude_r
            result[f"{path}.magnitude_c"] = mod.magnitude_c

    def _collect(parent, parent_path):
        for name in list(vars(parent).keys()):
            child = getattr(parent, name)
            child_path = f"{parent_path}.{name}" if parent_path else name

            if isinstance(child, _ALL_LORA_TYPES):
                _collect_from_module(child, child_path)
            elif isinstance(child, nn.Module):
                _collect(child, child_path)
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    item_path = f"{child_path}.{i}"
                    if isinstance(item, _ALL_LORA_TYPES):
                        _collect_from_module(item, item_path)
                    elif isinstance(item, nn.Module):
                        _collect(item, item_path)

    _collect(model, prefix)
    return result


# ── Key format conversion ──


def _mlx_key_to_torch_key(k: str) -> str:
    """MLX → PyTorch parametrize key format."""
    parts = k.rsplit(".", 1)
    path, param = parts[0], parts[1]
    return f"{path}.parametrizations.weight.0.{param}"


def _torch_key_to_mlx_key(k: str) -> str:
    """PyTorch parametrize → MLX key format."""
    return k.replace(".parametrizations.weight.0.", ".")


# ── Save / Load ──


def save_lora_safetensors(
    dit_state: dict[str, mx.array],
    conditioner_state: dict[str, mx.array] | None,
    config: dict,
    path: str,
    torch_compat: bool = True,
):
    """Save adapter weights as safetensors with embedded config.

    If torch_compat=True, keys use PyTorch parametrize format with
    'model.' prefix for DiT and 'conditioners.' prefix for conditioner.
    """
    import numpy as np
    from safetensors.numpy import save_file

    all_weights = {}
    for k, v in dit_state.items():
        key = f"model.{_mlx_key_to_torch_key(k)}" if torch_compat else k
        all_weights[key] = np.array(v.astype(mx.float16))
    if conditioner_state:
        for k, v in conditioner_state.items():
            key = f"conditioners.{_mlx_key_to_torch_key(k)}" if torch_compat else k
            all_weights[key] = np.array(v.astype(mx.float16))

    metadata = {"lora_config": json.dumps(config), "trained_with": "mlx"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(all_weights, str(path), metadata=metadata)
    return len(all_weights)


def load_lora_into_model(
    model: nn.Module,
    lora_path: str,
    prefix: str = "",
):
    """Load adapter weights from safetensors into an already-injected model.

    Handles both MLX-native keys and PyTorch parametrize keys.
    """
    from safetensors import safe_open

    with safe_open(str(lora_path), framework="numpy") as f:
        meta = f.metadata() or {}
        config = json.loads(meta.get("lora_config", "{}"))

        weights = {}
        for k in f.keys():
            clean = k
            if clean.startswith("model."):
                clean = clean[len("model."):]
            elif clean.startswith("conditioners."):
                clean = clean[len("conditioners."):]
            clean = _torch_key_to_mlx_key(clean)
            if prefix:
                clean = f"{prefix}.{clean}"
            weights[clean] = mx.array(f.get_tensor(k))

    _LOADABLE_KEYS = {"lora_A", "lora_B", "magnitude", "magnitude_r", "magnitude_c"}

    def _load_recursive(parent, parent_path):
        for name in list(vars(parent).keys()):
            child = getattr(parent, name)
            child_path = f"{parent_path}.{name}" if parent_path else name

            if isinstance(child, _ALL_LORA_TYPES):
                for attr in _LOADABLE_KEYS:
                    wk = f"{child_path}.{attr}"
                    if wk in weights and hasattr(child, attr):
                        setattr(child, attr, weights[wk].astype(getattr(child, attr).dtype))
            elif isinstance(child, nn.Module):
                _load_recursive(child, child_path)
            elif isinstance(child, list):
                for i, item in enumerate(child):
                    item_path = f"{child_path}.{i}"
                    if isinstance(item, _ALL_LORA_TYPES):
                        for attr in _LOADABLE_KEYS:
                            wk = f"{item_path}.{attr}"
                            if wk in weights and hasattr(item, attr):
                                setattr(item, attr, weights[wk].astype(getattr(item, attr).dtype))
                    elif isinstance(item, nn.Module):
                        _load_recursive(item, item_path)

    _load_recursive(model, prefix)
    return config
