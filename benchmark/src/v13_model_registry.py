from __future__ import annotations

import sys
import types
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from v13_contracts import strict_flat_summaries
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
from v13_guard import REPO_ROOT


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_NAMES = (
    "MABOOST",
    "Hybrid-plain",
    "Mamba-z-only",
    "Flat-head",
    "XGBOOST-FLAT",
    "XGBOOST-SUMMARY+STATIC",
    "GRU-D",
    "LSTM",
    "Transformer",
    "TCN",
    "SAnD",
    "STraTS",
    "InterpNet",
    "DuETT",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    trainer: str
    temporal_summaries: int | None
    static_covariates: int | None
    observedness_rates: int | None
    source_validation_selection: bool
    mortality_task: bool = True
    los_task: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "MABOOST": ModelSpec(
        "MABOOST", "primary", "mamba_validation_selected_fusion",
        200, 12, 40, True,
    ),
    "Hybrid-plain": ModelSpec(
        "Hybrid-plain", "primary", "mamba_z_plus_raw_summary_static",
        200, 12, 0, True,
    ),
    "Mamba-z-only": ModelSpec(
        "Mamba-z-only", "primary", "mamba_z_only", None, 0, 0, True,
    ),
    "Flat-head": ModelSpec(
        "Flat-head", "primary", "xgboost_flat_head", 200, 12, 40, True,
    ),
    "XGBOOST-FLAT": ModelSpec(
        "XGBOOST-FLAT", "primary", "xgboost_strict_flat", 200, 0, 0, True,
    ),
    "XGBOOST-SUMMARY+STATIC": ModelSpec(
        "XGBOOST-SUMMARY+STATIC", "primary", "xgboost_summary_static",
        200, 12, 0, True,
    ),
    "GRU-D": ModelSpec("GRU-D", "deep_baseline", "torch_grud", None, 12, None, True),
    "LSTM": ModelSpec("LSTM", "deep_baseline", "torch_lstm", None, 12, None, True),
    "Transformer": ModelSpec(
        "Transformer", "deep_baseline", "torch_transformer", None, 12, None, True
    ),
    "TCN": ModelSpec("TCN", "deep_baseline", "torch_tcn", None, 12, None, True),
    "SAnD": ModelSpec("SAnD", "deep_baseline", "torch_sand", None, 12, None, True),
    "STraTS": ModelSpec("STraTS", "deep_baseline", "torch_strats", None, 12, None, True),
    "InterpNet": ModelSpec(
        "InterpNet", "deep_baseline", "torch_interpnet", None, 12, None, True
    ),
    "DuETT": ModelSpec("DuETT", "deep_baseline", "torch_duett", None, 12, None, True),
}


def registry_rows() -> list[dict[str, Any]]:
    return [asdict(MODEL_REGISTRY[name]) for name in MODEL_NAMES]


def validate_model_registry(expected_names: list[str] | tuple[str, ...]) -> None:
    if tuple(expected_names) != MODEL_NAMES:
        raise RuntimeError(
            f"model registry order mismatch: expected={list(expected_names)}, "
            f"observed={list(MODEL_NAMES)}"
        )
    if set(MODEL_REGISTRY) != set(MODEL_NAMES):
        raise RuntimeError("model registry contains missing or extra models")
    primary = [spec for spec in MODEL_REGISTRY.values() if spec.family == "primary"]
    deep = [spec for spec in MODEL_REGISTRY.values() if spec.family == "deep_baseline"]
    if len(primary) != 6 or len(deep) != 8:
        raise RuntimeError("v13 must contain exactly six primary and eight deep models")


def observedness_rates(mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if mask.ndim != 3 or mask.shape[2] != len(DYNAMIC_FEATURES):
        raise ValueError("mask must use the [N,T,40] dynamic feature contract")
    rates = (mask > 0).mean(axis=1, dtype=np.float32).astype(np.float32)
    names = [f"observed_rate__{name}" for name in DYNAMIC_FEATURES]
    return rates, names


def build_flat_design(
    model_name: str,
    values: np.ndarray,
    mask: np.ndarray,
    static: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    summaries, summary_names = strict_flat_summaries(
        values, mask, list(DYNAMIC_FEATURES)
    )
    if static.ndim != 2 or static.shape != (values.shape[0], len(STATIC_FEATURES)):
        raise ValueError("static must use the aligned [N,12] landmark contract")
    if model_name == "XGBOOST-FLAT":
        matrix, names = summaries, summary_names
    elif model_name in {"XGBOOST-SUMMARY+STATIC", "Hybrid-plain"}:
        matrix = np.concatenate([summaries, static], axis=1).astype(np.float32)
        names = [*summary_names, *STATIC_FEATURES]
    elif model_name in {"Flat-head", "MABOOST"}:
        rates, rate_names = observedness_rates(mask)
        matrix = np.concatenate([summaries, static, rates], axis=1).astype(np.float32)
        names = [*summary_names, *STATIC_FEATURES, *rate_names]
    else:
        raise ValueError(f"{model_name} does not have a registered flat design")
    expected = {
        "XGBOOST-FLAT": 200,
        "XGBOOST-SUMMARY+STATIC": 212,
        "Hybrid-plain": 212,
        "Flat-head": 252,
        "MABOOST": 252,
    }[model_name]
    if matrix.shape != (values.shape[0], expected) or len(names) != expected:
        raise AssertionError(f"{model_name} flat design violates its locked contract")
    return matrix, names


def instantiate_deep_model(
    model_name: str,
    *,
    pos_rate: float,
    hidden_dim: int = 64,
    sequence_length: int = 64,
) -> torch.nn.Module:
    from src.models.baselines import (
        DuETTModel,
        GRUDModel,
        InterpNetModel,
        LSTMModel,
        SAND,
        Strats,
        TCN_TS,
        TransformerModel,
    )
    from src.training.baseline_trainer import _make_args

    classes = {
        "GRU-D": GRUDModel,
        "LSTM": LSTMModel,
        "Transformer": TransformerModel,
        "TCN": TCN_TS,
        "SAnD": SAND,
        "STraTS": Strats,
        "InterpNet": InterpNetModel,
        "DuETT": DuETTModel,
    }
    if model_name not in classes:
        raise ValueError(f"not a registered deep baseline: {model_name}")
    args = _make_args(
        d_input=len(DYNAMIC_FEATURES),
        d_static=len(STATIC_FEATURES),
        pos_rate=pos_rate,
        model_type=model_name,
        hid_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        T=sequence_length,
        M=32,
        ref_points=32,
        max_obs=512,
    )
    args.d_input = len(DYNAMIC_FEATURES)
    args.d_model = hidden_dim
    model = classes[model_name](args)
    if model_name == "SAnD":
        # The inherited attention keeps functional dropout active in eval mode
        # and contracts each attention row with v at the query index instead of
        # the key/value index.  Patch only this isolated v13 instance; the
        # registered architecture, parameters, shapes, and training dropout are
        # unchanged.
        for layer in model.transformer:
            layer.mha.forward = types.MethodType(
                _landmark_sand_attention_forward,
                layer.mha,
            )
        model.landmark_attention_semantics_correction = True
    if model_name == "STraTS":
        # Correction-02 fixed the inherited query/value contraction.  The
        # correction-04 qualification then exposed the remaining paper-semantic
        # omissions: sqrt(d/h) score scaling and LayerNorm after both residual
        # sublayers.  Patch only this isolated v13 instance; parameter names,
        # shapes, layer count, capacity, and all hyperparameters remain fixed.
        model.transformer.forward = types.MethodType(
            _landmark_strats_transformer_forward,
            model.transformer,
        )
        model.landmark_attention_axis_correction = True
    return model


def _landmark_sand_attention_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention with deterministic SAnD evaluation."""
    batch_size, sequence_length, hidden_dimension = x.size()
    query = torch.matmul(x, module.Wq).view(
        batch_size,
        sequence_length,
        module.num_heads,
        module.dk,
    ) / np.sqrt(module.dk)
    key = torch.matmul(x, module.Wk).view(
        batch_size,
        sequence_length,
        module.num_heads,
        module.dk,
    )
    value = torch.matmul(x, module.Wv).view(
        batch_size,
        sequence_length,
        module.num_heads,
        module.dk,
    )
    attention = torch.einsum("bthd,blhd->bhtl", query, key) + mask
    attention = F.dropout(
        F.softmax(attention, dim=-1),
        p=module.dropout,
        training=module.training,
    )
    attended = torch.einsum("bhtl,blhd->bthd", attention, value)
    return module.Wo(attended.reshape(
        batch_size,
        sequence_length,
        hidden_dimension,
    ))


def _landmark_strats_transformer_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Paper-defined scaled STraTS attention with normalized residual blocks."""
    batch_size, max_length, _ = x.size()
    valid = mask > 0
    valid_query = valid[:, :, None].to(x.dtype)
    key_penalty = (~valid)[:, None, None, :].to(x.dtype) * torch.finfo(x.dtype).min
    for index in range(module.N):
        query = torch.einsum("bld,hde->bhle", x, module.Wq[index])
        key = torch.einsum("bld,hde->bhle", x, module.Wk[index])
        value = torch.einsum("bld,hde->bhle", x, module.Wv[index])
        score = torch.einsum("bhle,bhke->bhlk", query, key) / np.sqrt(
            module.dk
        )
        attention = torch.softmax(score + key_penalty, dim=-1)
        attention = F.dropout(
            attention,
            p=float(module.attention_dropout),
            training=module.training,
        )
        context = torch.einsum("bhlk,bhke->bhle", attention, value)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, max_length, -1)
        x = F.layer_norm((
            F.dropout(
                torch.matmul(context, module.Wo[index]),
                float(module.dropout),
                module.training,
            )
            + x
        ) / 2, (module.d,)) * valid_query
        feed_forward = torch.matmul(x, module.W1[index]) + module.b1[index]
        feed_forward = F.gelu(feed_forward)
        feed_forward = torch.matmul(feed_forward, module.W2[index]) + module.b2[index]
        x = F.layer_norm((
            F.dropout(feed_forward, float(module.dropout), module.training) + x
        ) / 2, (module.d,)) * valid_query
    return x


def instantiate_maboost(
    *,
    hidden_dim: int = 64,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    event_max_tokens: int = 96,
) -> torch.nn.Module:
    from src.models.mamba_encoder import DualHeadMamba

    return DualHeadMamba(
        d_input=len(DYNAMIC_FEATURES),
        d_model=hidden_dim,
        d_static=len(STATIC_FEATURES),
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        event_max_tokens=event_max_tokens,
    )
