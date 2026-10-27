from __future__ import annotations

import copy
import gc
import hashlib
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback

from auxiliary_los_contract import (
    AggregateNumericalDiagnostics,
    CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    LOG_LOS_BOUND,
    bounded_log_los,
)
from v13_analysis import canonical_metric_row, temporal_stress_view
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
from v13_guard import (
    REPO_ROOT,
    assert_safe_output_path,
    safe_write_text,
    sha256_file,
)
from v13_model_registry import build_flat_design, instantiate_deep_model, instantiate_maboost
from v13_selection import (
    IsotonicCalibration,
    fit_source_isotonic_calibration,
    select_source_threshold,
    select_three_component_fusion,
)
from v13_temporal_protocol import (
    deterministic_event_anchor_indices,
    last_observed_elapsed_days,
    retain_through_anchor,
)
from v13_resource_guardian import resource_safe_point


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FIXED_LANDMARK_DAYS = 6.0 / 24.0


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_normalization_chunked(
    arrays: dict[str, np.ndarray],
    train_rows: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> dict[str, np.ndarray]:
    dynamic_sum = np.zeros(len(DYNAMIC_FEATURES), dtype=np.float64)
    dynamic_sum_sq = np.zeros(len(DYNAMIC_FEATURES), dtype=np.float64)
    dynamic_count = np.zeros(len(DYNAMIC_FEATURES), dtype=np.int64)
    static_sum = np.zeros(len(STATIC_FEATURES), dtype=np.float64)
    static_sum_sq = np.zeros(len(STATIC_FEATURES), dtype=np.float64)
    static_count = 0
    for start in range(0, len(train_rows), chunk_size):
        rows = train_rows[start:start + chunk_size]
        values = np.asarray(arrays["values"][rows], dtype=np.float32)
        mask = np.asarray(arrays["mask"][rows], dtype=bool)
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        dynamic_sum += (clean * mask).sum(axis=(0, 1))
        dynamic_sum_sq += ((clean * clean) * mask).sum(axis=(0, 1))
        dynamic_count += mask.sum(axis=(0, 1))
        static = np.nan_to_num(
            np.asarray(arrays["static"][rows], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float64)
        static_sum += static.sum(axis=0)
        static_sum_sq += (static * static).sum(axis=0)
        static_count += len(static)
    safe_dynamic_count = np.maximum(dynamic_count, 1)
    dynamic_mean = dynamic_sum / safe_dynamic_count
    dynamic_var = dynamic_sum_sq / safe_dynamic_count - dynamic_mean * dynamic_mean
    dynamic_std = np.sqrt(np.maximum(dynamic_var, 0.0))
    dynamic_mean[dynamic_count == 0] = 0.0
    dynamic_std[dynamic_count == 0] = 1.0
    dynamic_std = np.maximum(dynamic_std, 1e-6)
    static_mean = static_sum / max(static_count, 1)
    static_var = static_sum_sq / max(static_count, 1) - static_mean * static_mean
    static_std = np.maximum(np.sqrt(np.maximum(static_var, 0.0)), 1e-6)
    return {
        "dynamic_mean": dynamic_mean.astype(np.float32),
        "dynamic_std": dynamic_std.astype(np.float32),
        "dynamic_observed_count": dynamic_count.astype(np.int64),
        "static_mean": static_mean.astype(np.float32),
        "static_std": static_std.astype(np.float32),
        "source_train_rows": np.asarray([len(train_rows)], dtype=np.int64),
    }


class PrivateArrayDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        rows: np.ndarray,
        normalization: dict[str, np.ndarray],
        *,
        longitudinal: bool = False,
        minimum_elapsed_hours: float = 2.0,
        maximum_anchors: int = 4,
    ) -> None:
        self.arrays = arrays
        self.rows = np.asarray(rows, dtype=np.int64)
        self.normalization = normalization
        self.samples: list[tuple[int, int | None]] = []
        for row in self.rows:
            row_int = int(row)
            if longitudinal:
                anchors = deterministic_event_anchor_indices(
                    np.asarray(arrays["mask"][row_int], dtype=np.uint8),
                    np.asarray(arrays["elapsed"][row_int], dtype=np.float32),
                    minimum_elapsed_hours=minimum_elapsed_hours,
                    maximum_anchors=maximum_anchors,
                )
                self.samples.extend((row_int, anchor) for anchor in anchors)
            else:
                self.samples.append((row_int, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row, anchor_index = self.samples[index]
        raw_mask = np.asarray(self.arrays["mask"][row], dtype=np.uint8)
        raw_values = np.asarray(self.arrays["values"][row], dtype=np.float32)
        if anchor_index is None:
            mask = raw_mask.astype(np.float32, copy=True)
            retained_values = raw_values.copy()
        else:
            retained_values, retained_mask = retain_through_anchor(
                raw_values, raw_mask, anchor_index
            )
            mask = retained_mask.astype(np.float32)
        values = np.nan_to_num(
            retained_values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        values = np.where(
            mask > 0,
            (values - self.normalization["dynamic_mean"])
            / self.normalization["dynamic_std"],
            0.0,
        ).clip(-5.0, 5.0).astype(np.float32)
        tau = np.nan_to_num(
            np.asarray(self.arrays["tau"][row], dtype=np.float32).copy(),
            nan=60.0,
            posinf=3600.0,
            neginf=1.0,
        ).clip(1.0, 86400.0)
        static = np.nan_to_num(
            np.asarray(self.arrays["static"][row], dtype=np.float32).copy(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        static = (
            (static - self.normalization["static_mean"])
            / self.normalization["static_std"]
        ).clip(-10.0, 10.0).astype(np.float32)
        # Every full and stressed view is a prediction made at the fixed 6 h
        # landmark.  Last observation time is therefore never the LOS origin.
        remaining_los = np.asarray(
            [self.arrays["los"][row]], dtype=np.float32
        )
        return (
            torch.from_numpy(values),
            torch.from_numpy(tau),
            torch.from_numpy(mask),
            torch.from_numpy(static),
            torch.tensor(int(self.arrays["mortality"][row]), dtype=torch.long),
            torch.tensor(float(remaining_los[0]), dtype=torch.float32),
        )


def make_loaders(
    arrays: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
    *,
    batch_size: int,
    seed: int,
    longitudinal_train: bool = False,
    minimum_elapsed_hours: float = 2.0,
    maximum_anchors: int = 4,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            PrivateArrayDataset(
                arrays,
                rows,
                normalization,
                longitudinal=bool(longitudinal_train and split == "train"),
                minimum_elapsed_hours=minimum_elapsed_hours,
                maximum_anchors=maximum_anchors,
            ),
            batch_size=batch_size,
            shuffle=shuffle_train and split == "train",
            num_workers=0,
            pin_memory=True,
            generator=generator if shuffle_train and split == "train" else None,
        )
        for split, rows in indices.items()
    }


def remaining_los_for_rows(
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    elapsed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(rows, dtype=np.int64)
    selected_mask = (
        np.asarray(arrays["mask"][selected], dtype=np.uint8)
        if mask is None else np.asarray(mask, dtype=np.uint8)
    )
    selected_elapsed = (
        np.asarray(arrays["elapsed"][selected], dtype=np.float32)
        if elapsed is None else np.asarray(elapsed, dtype=np.float32)
    )
    # Keep the observed anchor only as an audit variable. The supervised LOS
    # label is always discharge minus the fixed 6 h landmark, even for a
    # prefix/missingness view with an earlier last observation.
    observed_anchor = last_observed_elapsed_days(selected_mask, selected_elapsed)
    labels = np.asarray(arrays["los"][selected], dtype=np.float32)
    fixed_anchor = np.full(labels.shape, FIXED_LANDMARK_DAYS, dtype=np.float32)
    if (observed_anchor > fixed_anchor + 1e-6).any():
        raise RuntimeError("predictor observation exceeds the fixed 6 h landmark")
    return labels, fixed_anchor


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_state(path: Path, model: nn.Module) -> None:
    target = assert_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)


def _save_state_with_hash(path: Path, model: nn.Module) -> str:
    _save_state(path, model)
    digest = sha256_file(path)
    safe_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.relative_to(REPO_ROOT)}\n",
    )
    return digest


@torch.no_grad()
def _predict_mamba(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    include_embeddings: bool,
    ignore_static: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    los_predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    los_labels: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for values, tau, mask, static, mortality, los in loader:
        values_device = values.to(device, non_blocking=True)
        tau_device = tau.to(device, non_blocking=True)
        mask_device = mask.to(device, non_blocking=True)
        static_device = static.to(device, non_blocking=True)
        logits, pred_log_los = model(
            values_device,
            tau_device,
            mask_device,
            None if ignore_static else static_device,
        )
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        los_predictions.append(torch.expm1(pred_log_los.float()).clamp(min=0).cpu().numpy())
        labels.append(mortality.numpy())
        los_labels.append(los.numpy())
        if include_embeddings:
            z_multi, _ = model.encoder.extract_features(
                values_device, tau_device, mask_device
            )
            embeddings.append(z_multi.float().cpu().numpy())
    result = {
        "probability": np.concatenate(probabilities).astype(np.float32),
        "los_prediction": np.concatenate(los_predictions).astype(np.float32),
        "mortality": np.concatenate(labels).astype(np.int8),
        "los": np.concatenate(los_labels).astype(np.float32),
    }
    if include_embeddings:
        result["embedding"] = np.concatenate(embeddings).astype(np.float32)
    return result


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, targets, reduction="none")
    probability = torch.exp(-cross_entropy)
    return (
        alpha[targets] * (1.0 - probability).pow(gamma) * cross_entropy
    ).mean()


def _log_cosh_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = prediction - target
    return torch.mean(difference + F.softplus(-2.0 * difference) - math.log(2.0))


def _warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float = 0.01,
) -> float:
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return float(
        minimum_ratio
        + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def _mixup_positive_negative(
    values: torch.Tensor,
    tau: torch.Tensor,
    mask: torch.Tensor,
    mortality: torch.Tensor,
    los: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = torch.nonzero(mortality == 1, as_tuple=False).squeeze(-1)
    negative = torch.nonzero(mortality == 0, as_tuple=False).squeeze(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        return values, tau, mask, mortality, los
    count = positive.numel()
    weight = torch.distributions.Beta(alpha, alpha).sample((count,)).to(
        values.device
    ).clamp(0.3, 0.7)
    partner = negative[
        torch.randint(negative.numel(), (count,), device=values.device)
    ]
    weight_3d = weight.view(-1, 1, 1)
    weight_2d = weight.view(-1, 1)
    return (
        torch.cat([
            values,
            weight_3d * values[positive] + (1.0 - weight_3d) * values[partner],
        ]),
        torch.cat([
            tau,
            weight_2d * tau[positive] + (1.0 - weight_2d) * tau[partner],
        ]),
        torch.cat([
            mask,
            weight_3d * mask[positive] + (1.0 - weight_3d) * mask[partner],
        ]),
        torch.cat([
            mortality,
            torch.ones(count, dtype=torch.long, device=values.device),
        ]),
        torch.cat([
            los,
            weight * los[positive] + (1.0 - weight) * los[partner],
        ]),
    )


class _ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key] = (
                    self.shadow[key] * self.decay
                    + value.detach().float() * (1.0 - self.decay)
                )

    def state_for(self, device: str) -> dict[str, torch.Tensor]:
        return {key: value.to(device) for key, value in self.shadow.items()}


def train_mamba_encoder(
    arrays: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: str,
    training_config: dict[str, Any] | None = None,
    bounded: bool = False,
) -> tuple[nn.Module, dict[str, Any], dict[str, DataLoader]]:
    recipe = training_config or {}
    batch_size = int(recipe.get("batch_size", 32))
    # Stage-2 joins these predictions/embeddings to index-ordered labels and
    # tabular designs, so every inference split keeps index order.
    inference_loaders = make_loaders(
        arrays,
        indices,
        normalization,
        batch_size=batch_size,
        seed=seed,
        shuffle_train=False,
    )
    training_loaders = make_loaders(
        arrays,
        indices,
        normalization,
        batch_size=batch_size,
        seed=seed,
        longitudinal_train=bool(recipe.get("longitudinal_training", True)),
        minimum_elapsed_hours=float(recipe.get("minimum_anchor_hours", 2.0)),
        maximum_anchors=int(recipe.get("maximum_anchors_per_stay", 4)),
    )
    model = instantiate_maboost(
        hidden_dim=128,
        d_state=32,
        d_conv=4,
        expand=2,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        event_max_tokens=96,
    ).to(device)
    learning_rate = float(recipe.get("learning_rate", 2e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(recipe.get("weight_decay", 1e-4)),
        betas=(0.9, 0.98),
    )
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    effective_epochs = 1 if bounded else max_epochs
    total_steps = effective_epochs * max(len(training_loaders["train"]), 1)
    warmup_steps = int(total_steps * float(recipe.get("warmup_fraction", 0.10)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_multiplier(
            step, total_steps, warmup_steps
        ),
    )
    train_mortality = np.asarray(
        arrays["mortality"][indices["train"]], dtype=np.int8
    )
    negatives = max(int((train_mortality == 0).sum()), 1)
    positives = max(int((train_mortality == 1).sum()), 1)
    inverse_negative = 1.0 / negatives
    inverse_positive = 1.0 / positives
    alpha = torch.tensor(
        [
            inverse_negative / (inverse_negative + inverse_positive),
            inverse_positive / (inverse_negative + inverse_positive),
        ],
        dtype=torch.float32,
        device=device,
    )
    ema = _ModelEMA(model, float(recipe.get("ema_decay", 0.999)))
    use_mixup = bool(recipe.get("use_mixup", True)) and not bounded
    for epoch in range(1, effective_epochs + 1):
        model.train()
        losses = []
        for batch_index, (values, tau, mask, _static, mortality, los) in enumerate(
            training_loaders["train"]
        ):
            optimizer.zero_grad(set_to_none=True)
            values_device = values.to(device, non_blocking=True)
            tau_device = tau.to(device, non_blocking=True)
            mask_device = mask.to(device, non_blocking=True)
            mortality_device = mortality.to(device, non_blocking=True)
            los_device = los.to(device, non_blocking=True)
            if use_mixup:
                (
                    values_device,
                    tau_device,
                    mask_device,
                    mortality_device,
                    los_device,
                ) = _mixup_positive_negative(
                    values_device,
                    tau_device,
                    mask_device,
                    mortality_device,
                    los_device,
                    alpha=float(recipe.get("mixup_alpha", 0.4)),
                )
            logits, pred_log_los = model(
                values_device,
                tau_device,
                mask_device,
                None,
            )
            logits = logits.clamp(-20.0, 20.0)
            pred_log_los = pred_log_los.clamp(-8.0, 8.0)
            mortality_loss = _focal_loss(
                logits,
                mortality_device,
                alpha,
                float(recipe.get("focal_gamma", 2.0)),
            )
            los_loss = _log_cosh_loss(pred_log_los, torch.log1p(los_device))
            loss = mortality_loss + float(recipe.get("los_loss_weight", 0.5)) * los_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite MABOOST training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    torch.nan_to_num_(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(recipe.get("clip_norm", 1.0))
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("non-finite MABOOST gradient norm")
            optimizer.step()
            scheduler.step()
            ema.update(model)
            resource_safe_point(f"stage1_epoch{epoch}_batch{batch_index}")
            losses.append(float(loss.detach().cpu()))
            if bounded and batch_index >= 1:
                break
        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.state_for(device))
        validation = _predict_mamba(
            model,
            inference_loaders["validation"],
            device,
            include_embeddings=False,
            ignore_static=True,
        )
        score = float(average_precision_score(
            validation["mortality"], validation["probability"]
        ))
        history.append({
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "validation_auprc": score,
        })
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                model.load_state_dict(raw_state)
                break
        model.load_state_dict(raw_state)
        del validation, raw_state
        resource_safe_point(f"stage1_epoch{epoch}_complete", force_cleanup=True)
    if best_state is None:
        raise RuntimeError("MABOOST training did not produce a finite validation checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    return model, {
        "epochs_completed": len(history),
        "best_validation_auprc": float(best_score),
        "history": history,
        "wall_seconds": float(time.perf_counter() - started),
        "training_rows_base": int(len(indices["train"])),
        "training_anchor_rows": int(len(training_loaders["train"].dataset)),
        "remaining_los_target": True,
        "legacy_recipe": {
            "longitudinal_training": bool(recipe.get("longitudinal_training", True)),
            "focal_gamma": float(recipe.get("focal_gamma", 2.0)),
            "use_mixup": use_mixup,
            "mixup_alpha": float(recipe.get("mixup_alpha", 0.4)),
            "warmup_fraction": float(recipe.get("warmup_fraction", 0.10)),
            "ema_decay": float(recipe.get("ema_decay", 0.999)),
            "ignore_static_inputs": True,
        },
    }, inference_loaders


def _flat_design_split(
    model_name: str,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, list[str]]:
    chunks: list[np.ndarray] = []
    names: list[str] | None = None
    for start in range(0, len(rows), chunk_size):
        selected = rows[start:start + chunk_size]
        matrix, current_names = build_flat_design(
            model_name,
            np.asarray(arrays["values"][selected], dtype=np.float32),
            np.asarray(arrays["mask"][selected], dtype=np.uint8),
            np.asarray(arrays["static"][selected], dtype=np.float32),
        )
        chunks.append(matrix)
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("flat design feature order changed between chunks")
    return np.concatenate(chunks), names or []


@torch.no_grad()
def _view_embeddings(
    model: nn.Module,
    values: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    normalization: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch_mask = (np.asarray(mask[start:stop]) > 0).astype(np.float32)
        clean = np.nan_to_num(
            np.asarray(values[start:stop], dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized = np.where(
            batch_mask > 0,
            (clean - normalization["dynamic_mean"])
            / normalization["dynamic_std"],
            0.0,
        ).clip(-5.0, 5.0).astype(np.float32)
        clean_tau = np.nan_to_num(
            np.asarray(tau[start:stop], dtype=np.float32),
            nan=60.0,
            posinf=3600.0,
            neginf=1.0,
        ).clip(1.0, 86400.0)
        z_multi, _ = model.encoder.extract_features(
            torch.from_numpy(normalized).to(device),
            torch.from_numpy(clean_tau).to(device),
            torch.from_numpy(batch_mask).to(device),
        )
        parts.append(z_multi.float().cpu().numpy().astype(np.float32))
    return np.concatenate(parts) if parts else np.empty((0, 512), dtype=np.float32)


def _stress_weight_class(scenario: str) -> str:
    if scenario.startswith("early_"):
        return "early"
    if "missing" in scenario or "drop" in scenario:
        return "missing"
    return "irregular"


def matched_augmentation_rows(
    rows: np.ndarray,
    *,
    seed: int,
    scenario: str,
    maximum_rows: int,
) -> np.ndarray:
    """Return architecture-independent source rows for a matched stress view."""
    selected = np.asarray(rows, dtype=np.int64)
    if maximum_rows <= 0:
        raise ValueError("matched augmentation maximum_rows must be positive")
    if len(selected) <= maximum_rows:
        return selected.copy()
    digest = hashlib.sha256(
        f"{seed}|matched_augmentation_rows_v1|{scenario}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(selected, size=maximum_rows, replace=False))


def _build_temporal_robust_bundle(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    normalization: dict[str, np.ndarray],
    *,
    scenarios: list[str],
    seed: int,
    device: str,
    max_rows_per_view: int,
    class_weights: dict[str, float],
    cache_root: Path,
    base_bundle: dict[str, np.ndarray] | None = None,
    base_mass: float = 0.0,
    max_total_rows: int | None = None,
    minimum_free_cache_bytes: int = 0,
) -> dict[str, np.ndarray]:
    cache = assert_safe_output_path(cache_root)
    cache.mkdir(parents=True, exist_ok=False)
    piece_paths: dict[str, list[Path]] = {
        "hybrid": [],
        "strict_flat": [],
        "flat_head": [],
        "mortality": [],
        "los": [],
        "sample_weight": [],
    }
    scenario_rows: list[int] = []
    scenario_audit: list[dict[str, Any]] = []
    class_counts = {
        group: max(sum(_stress_weight_class(item) == group for item in scenarios), 1)
        for group in ("early", "irregular", "missing")
    }
    for scenario in scenarios:
        selected_rows = matched_augmentation_rows(
            rows,
            seed=seed,
            scenario=scenario,
            maximum_rows=max_rows_per_view,
        )
        raw_values = np.asarray(arrays["values"][selected_rows], dtype=np.float32)
        raw_mask = np.asarray(arrays["mask"][selected_rows], dtype=np.uint8)
        raw_tau = np.asarray(arrays["tau"][selected_rows], dtype=np.float32)
        raw_elapsed = np.asarray(arrays["elapsed"][selected_rows], dtype=np.float32)
        view, _ = temporal_stress_view(
            raw_values,
            raw_mask,
            raw_tau,
            raw_elapsed,
            scenario=scenario,
            seed=seed,
            source_row_ids=selected_rows,
        )
        local = np.asarray(view["source_row_indices"], dtype=np.int64)
        source_rows = selected_rows[local]
        if not len(source_rows):
            continue
        static = np.asarray(arrays["static"][source_rows], dtype=np.float32)
        embedding = _view_embeddings(
            model,
            view["values"],
            view["tau"],
            view["mask"],
            normalization,
            device=device,
        )
        summary_static, _ = build_flat_design(
            "XGBOOST-SUMMARY+STATIC", view["values"], view["mask"], static
        )
        strict_flat, _ = build_flat_design(
            "XGBOOST-FLAT", view["values"], view["mask"], static
        )
        flat_head, _ = build_flat_design(
            "Flat-head", view["values"], view["mask"], static
        )
        remaining = np.asarray(arrays["los"][source_rows], dtype=np.float32)
        weight_class = _stress_weight_class(scenario)
        scenario_mass = float(class_weights[weight_class]) / class_counts[weight_class]
        scenario_audit.append({
            "scenario": scenario,
            "rows": int(len(source_rows)),
            "mass": scenario_mass,
            "source_row_sha256": hashlib.sha256(
                np.asarray(source_rows, dtype=np.int64).tobytes()
            ).hexdigest(),
        })
        pieces = {
            "hybrid": np.concatenate([embedding, summary_static], axis=1),
            "strict_flat": strict_flat,
            "flat_head": flat_head,
            "mortality": np.asarray(
                arrays["mortality"][source_rows], dtype=np.int8
            ),
            "los": remaining,
            "sample_weight": np.full(
                len(source_rows), scenario_mass / len(source_rows), dtype=np.float64
            ),
        }
        scenario_rows.append(len(source_rows))
        for key, matrix in pieces.items():
            path = assert_safe_output_path(cache / f"piece_{len(scenario_rows):02d}_{key}.npy")
            with path.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            piece_paths[key].append(path)
        del pieces, embedding, summary_static, strict_flat, flat_head, view
        resource_safe_point(f"stage2_cache_{cache.name}_{scenario}", force_cleanup=True)
    if not piece_paths["mortality"]:
        raise RuntimeError("temporal-robust Stage 2 produced no source rows")
    base_count = 0
    if base_bundle is not None:
        required = {"hybrid", "strict_flat", "flat_head", "mortality", "los"}
        if set(base_bundle) != required:
            raise ValueError("temporal-robust base bundle has an invalid schema")
        base_count = len(base_bundle["mortality"])
        if base_count <= 0 or base_mass <= 0.0:
            raise ValueError("temporal-robust full-data base requires positive rows and mass")
        if any(len(base_bundle[key]) != base_count for key in required):
            raise ValueError("temporal-robust base bundle is not row-aligned")
        scenario_audit.insert(0, {
            "scenario": "full_6h",
            "rows": int(base_count),
            "mass": float(base_mass),
            "source_row_sha256": hashlib.sha256(
                np.asarray(rows, dtype=np.int64).tobytes()
            ).hexdigest(),
        })
    total_rows = base_count + sum(scenario_rows)
    if max_total_rows is not None and total_rows > int(max_total_rows):
        raise MemoryError(
            "temporal-robust Stage 2 row budget exceeded: "
            f"observed={total_rows}, limit={int(max_total_rows)}"
        )
    estimated_combined_bytes = 0
    for paths in piece_paths.values():
        sample = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        bytes_per_row = int(sample.dtype.itemsize * np.prod(sample.shape[1:], dtype=np.int64))
        estimated_combined_bytes += total_rows * bytes_per_row
        del sample
    free_cache_bytes = int(shutil.disk_usage(cache).free)
    required_cache_bytes = estimated_combined_bytes + int(minimum_free_cache_bytes)
    if free_cache_bytes < required_cache_bytes:
        raise MemoryError(
            "temporal-robust Stage 2 disk budget exceeded: "
            f"free={free_cache_bytes}, required={required_cache_bytes}"
        )
    result: dict[str, np.ndarray] = {}
    for key, paths in piece_paths.items():
        first = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        if key == "sample_weight" and base_bundle is not None:
            base_value = np.full(
                base_count, base_mass / base_count, dtype=np.float64
            )
        elif base_bundle is not None:
            base_value = np.asarray(base_bundle[key])
        else:
            base_value = None
        tail_shape = tuple(first.shape[1:])
        combined_path = assert_safe_output_path(cache / f"combined_{key}.npy")
        combined = np.lib.format.open_memmap(
            combined_path,
            mode="w+",
            dtype=first.dtype if base_value is None else base_value.dtype,
            shape=(total_rows, *tail_shape),
        )
        position = 0
        if base_value is not None:
            combined[:base_count] = base_value
            position = base_count
        for path in paths:
            piece = np.load(path, mmap_mode="r", allow_pickle=False)
            stop = position + len(piece)
            combined[position:stop] = piece
            position = stop
            del piece
            path.unlink()
        if position != total_rows:
            raise RuntimeError("temporal-robust disk cache row count mismatch")
        if key == "sample_weight":
            combined *= total_rows / max(float(combined.sum()), 1e-12)
        combined.flush()
        result[key] = combined
        del first
        resource_safe_point(
            f"stage2_cache_{cache.name}_combined_{key}", force_cleanup=True
        )
    result["audit"] = scenario_audit  # type: ignore[assignment]
    return result


def _fit_xgb_pair(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_mortality: np.ndarray,
    validation_mortality: np.ndarray,
    train_los: np.ndarray,
    validation_los: np.ndarray,
    *,
    seed: int,
    n_estimators: int,
    device: str,
    early_stopping_rounds: int = 50,
    train_sample_weight: np.ndarray | None = None,
    validation_sample_weight: np.ndarray | None = None,
) -> tuple[XGBClassifier, XGBRegressor]:
    class _GuardianCallback(TrainingCallback):
        def __init__(self, role: str) -> None:
            self.role = role

        def after_iteration(
            self,
            _model: Any,
            epoch: int,
            _evals_log: dict[str, Any],
        ) -> bool:
            resource_safe_point(f"xgboost_{self.role}_round{epoch}")
            return False

    common = {
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": device,
        "random_state": seed,
        "n_jobs": 6,
        "early_stopping_rounds": int(early_stopping_rounds),
    }
    mortality_model = XGBClassifier(
        **common,
        callbacks=[_GuardianCallback("mortality")],
        eval_metric="aucpr",
        scale_pos_weight=float(
            (train_mortality == 0).sum() / max((train_mortality == 1).sum(), 1)
        ),
    )
    mortality_model.fit(
        train_features,
        train_mortality,
        sample_weight=train_sample_weight,
        eval_set=[(validation_features, validation_mortality)],
        sample_weight_eval_set=(
            [validation_sample_weight]
            if validation_sample_weight is not None else None
        ),
        verbose=False,
    )
    los_model = XGBRegressor(
        **common,
        callbacks=[_GuardianCallback("los")],
        objective="reg:pseudohubererror",
        eval_metric="rmse",
    )
    los_model.fit(
        train_features,
        np.log1p(train_los),
        sample_weight=train_sample_weight,
        eval_set=[(validation_features, np.log1p(validation_los))],
        sample_weight_eval_set=(
            [validation_sample_weight]
            if validation_sample_weight is not None else None
        ),
        verbose=False,
    )
    return mortality_model, los_model


def _cleanup_stage2_cache(
    job_root: Path,
    *bundles: dict[str, np.ndarray] | None,
) -> None:
    for bundle in bundles:
        if bundle is None:
            continue
        for matrix in bundle.values():
            if isinstance(matrix, np.memmap) and matrix._mmap is not None:
                matrix.flush()
                matrix._mmap.close()
        bundle.clear()
    cache_root = assert_safe_output_path(job_root / "stage2_cache")
    if cache_root.is_dir():
        shutil.rmtree(cache_root)
    resource_safe_point("stage2_disk_cache_removed", force_cleanup=True)


def _build_flat_temporal_robust_bundle(
    model_name: str,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    scenarios: list[str],
    seed: int,
    max_rows_per_view: int,
    weights: dict[str, float],
    cache_root: Path,
    max_total_rows: int,
    minimum_free_cache_bytes: int,
) -> dict[str, np.ndarray]:
    """Create the matched, disk-backed XGBoost augmentation bundle.

    Row selection is a pure function of source row id, seed, and scenario, so
    all registered architectures receive exactly the same source views.  The
    full landmark view is always retained in full; only augmented views are
    capped to control disk/RAM demand.
    """
    if model_name not in {"XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC"}:
        raise ValueError(f"matched flat augmentation is not registered for {model_name}")
    if "full_6h" in scenarios:
        raise ValueError("full_6h is supplied as the explicit base, not an augmented view")
    cache = assert_safe_output_path(cache_root)
    cache.mkdir(parents=True, exist_ok=False)
    piece_paths: dict[str, list[Path]] = {
        "design": [], "mortality": [], "los": [], "sample_weight": []
    }
    class_counts = {
        group: max(sum(_stress_weight_class(item) == group for item in scenarios), 1)
        for group in ("early", "irregular", "missing")
    }
    view_specs = [("full_6h", np.asarray(rows, dtype=np.int64))]
    for scenario in scenarios:
        selected = matched_augmentation_rows(
            rows,
            seed=seed,
            scenario=scenario,
            maximum_rows=max_rows_per_view,
        )
        view_specs.append((scenario, selected))
    scenario_audit: list[dict[str, Any]] = []
    for view_index, (scenario, selected_rows) in enumerate(view_specs):
        if scenario == "full_6h":
            source_rows = selected_rows
            view_values = np.asarray(arrays["values"][source_rows], dtype=np.float32)
            view_mask = np.asarray(arrays["mask"][source_rows], dtype=np.uint8)
            scenario_mass = float(weights["full"])
        else:
            view, _ = temporal_stress_view(
                np.asarray(arrays["values"][selected_rows], dtype=np.float32),
                np.asarray(arrays["mask"][selected_rows], dtype=np.uint8),
                np.asarray(arrays["tau"][selected_rows], dtype=np.float32),
                np.asarray(arrays["elapsed"][selected_rows], dtype=np.float32),
                scenario=scenario,
                seed=seed,
                source_row_ids=selected_rows,
            )
            local = np.asarray(view["source_row_indices"], dtype=np.int64)
            source_rows = selected_rows[local]
            view_values = np.asarray(view["values"], dtype=np.float32)
            view_mask = np.asarray(view["mask"], dtype=np.uint8)
            weight_class = _stress_weight_class(scenario)
            scenario_mass = float(weights[weight_class]) / class_counts[weight_class]
        if not len(source_rows):
            raise RuntimeError(f"matched augmentation view is empty: {scenario}")
        design, names = build_flat_design(
            model_name,
            view_values,
            view_mask,
            np.asarray(arrays["static"][source_rows], dtype=np.float32),
        )
        pieces = {
            "design": design,
            "mortality": np.asarray(arrays["mortality"][source_rows], dtype=np.int8),
            "los": np.asarray(arrays["los"][source_rows], dtype=np.float32),
            "sample_weight": np.full(
                len(source_rows), scenario_mass / len(source_rows), dtype=np.float64
            ),
        }
        scenario_audit.append({
            "scenario": scenario,
            "rows": int(len(source_rows)),
            "mass": scenario_mass,
            "source_row_sha256": hashlib.sha256(
                np.asarray(source_rows, dtype=np.int64).tobytes()
            ).hexdigest(),
            "design_width": len(names),
        })
        for key, matrix in pieces.items():
            path = assert_safe_output_path(cache / f"piece_{view_index:02d}_{key}.npy")
            with path.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            piece_paths[key].append(path)
        del design, pieces, view_values, view_mask
        resource_safe_point(f"matched_flat_cache_{model_name}_{scenario}", force_cleanup=True)
    total_rows = sum(
        int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
        for path in piece_paths["mortality"]
    )
    if total_rows > int(max_total_rows):
        raise MemoryError(
            f"matched augmentation row budget exceeded: {total_rows}>{max_total_rows}"
        )
    estimated_bytes = 0
    for paths in piece_paths.values():
        sample = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        estimated_bytes += int(
            total_rows * sample.dtype.itemsize
            * np.prod(sample.shape[1:], dtype=np.int64)
        )
        del sample
    free_bytes = int(shutil.disk_usage(cache).free)
    if free_bytes < estimated_bytes + int(minimum_free_cache_bytes):
        raise MemoryError(
            "matched augmentation disk budget exceeded: "
            f"free={free_bytes}, required={estimated_bytes + int(minimum_free_cache_bytes)}"
        )
    result: dict[str, np.ndarray] = {}
    for key, paths in piece_paths.items():
        first = np.load(paths[0], mmap_mode="r", allow_pickle=False)
        combined_path = assert_safe_output_path(cache / f"combined_{key}.npy")
        combined = np.lib.format.open_memmap(
            combined_path,
            mode="w+",
            dtype=first.dtype,
            shape=(total_rows, *first.shape[1:]),
        )
        position = 0
        for path in paths:
            piece = np.load(path, mmap_mode="r", allow_pickle=False)
            stop = position + len(piece)
            combined[position:stop] = piece
            position = stop
            del piece
            path.unlink()
        combined.flush()
        if key == "sample_weight":
            combined *= total_rows / max(float(combined.sum()), 1e-12)
            combined.flush()
        result[key] = combined
        del first
    result["audit"] = scenario_audit  # type: ignore[assignment]
    return result


def _xgb_predictions(
    mortality_model: XGBClassifier,
    los_model: XGBRegressor,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = mortality_model.predict_proba(features)[:, 1].astype(np.float32)
    los = np.expm1(los_model.predict(features)).clip(min=0).astype(np.float32)
    return probability, los


def _select_los_fusion(
    component_predictions: np.ndarray,
    labels: np.ndarray,
    *,
    simplex_step: float = 0.05,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    predictions = np.asarray(component_predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("LOS tri-fusion requires aligned [N,3] predictions")
    resolution = int(round(1.0 / simplex_step))
    if not np.isclose(resolution * simplex_step, 1.0):
        raise ValueError("LOS fusion simplex_step must evenly divide one")
    best_weights = (0.0, 0.0, 1.0)
    best_mae = np.inf
    candidates = 0
    for first in range(resolution + 1):
        for second in range(resolution - first + 1):
            third = resolution - first - second
            weights = np.asarray([first, second, third], dtype=np.float64) / resolution
            prediction = predictions @ weights
            losses = np.abs(labels - prediction)
            mae = float(
                np.average(losses, weights=sample_weight)
                if sample_weight is not None
                else np.mean(losses)
            )
            candidates += 1
            if mae < best_mae:
                best_mae = mae
                best_weights = tuple(float(value) for value in weights)
    return {
        "selection_split": "validation",
        "selection_metric": "MAE",
        "components": ["Hybrid-plain", "XGBOOST-FLAT", "Flat-head"],
        "weights": list(best_weights),
        "validation_mae": best_mae,
        "simplex_step": float(simplex_step),
        "candidate_count": int(candidates),
        "sample_weighted": sample_weight is not None,
        "target_information_used": False,
    }


def train_xgb_or_mamba_primary(
    model_name: str,
    arrays: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    n_estimators: int,
    device: str,
    job_root: Path,
    training_config: dict[str, Any] | None = None,
    bounded: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[Path]]:
    labels = {
        split: np.asarray(arrays["mortality"][rows], dtype=np.int8)
        for split, rows in indices.items()
    }
    los_targets = {
        split: remaining_los_for_rows(arrays, rows)
        for split, rows in indices.items()
    }
    los_labels = {split: values[0] for split, values in los_targets.items()}
    los_anchors = {split: values[1] for split, values in los_targets.items()}
    xgb_config = (training_config or {}).get("xgboost", {})
    mamba_config = (training_config or {}).get("mamba", {})
    early_stopping_rounds = int(xgb_config.get("mortality_early_stopping_rounds", 50))
    effective_estimators = min(n_estimators, 2) if bounded else n_estimators
    artifacts: list[Path] = []
    training_audit: dict[str, Any] = {}
    selection_probability: np.ndarray | None = None
    selection_labels: np.ndarray | None = None
    selection_sample_weight: np.ndarray | None = None

    if model_name in {"XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC", "Flat-head"}:
        designs = {
            split: _flat_design_split(model_name, arrays, rows)[0]
            for split, rows in indices.items()
        }
        robust_config = (training_config or {}).get("temporal_robust_stage2", {})
        robust_train_flat: dict[str, np.ndarray] | None = None
        robust_validation_flat: dict[str, np.ndarray] | None = None
        if (
            model_name in {"XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC"}
            and bool(robust_config.get("enabled", False))
            and not bounded
        ):
            scenarios = [
                str(value) for value in robust_config["scenarios"]
                if str(value) != "full_6h"
            ]
            robust_train_flat = _build_flat_temporal_robust_bundle(
                model_name,
                arrays,
                indices["train"],
                scenarios=scenarios,
                seed=seed,
                max_rows_per_view=int(robust_config["max_train_rows_per_view"]),
                weights={
                    key: float(value)
                    for key, value in robust_config["weights"].items()
                },
                cache_root=job_root / "stage2_cache" / "train",
                max_total_rows=int(robust_config["max_total_rows"]),
                minimum_free_cache_bytes=int(
                    robust_config.get("minimum_free_cache_bytes", 0)
                ),
            )
            robust_validation_flat = _build_flat_temporal_robust_bundle(
                model_name,
                arrays,
                indices["validation"],
                scenarios=scenarios,
                seed=seed + 1_000_003,
                max_rows_per_view=int(robust_config["max_validation_rows_per_view"]),
                weights={
                    key: float(value)
                    for key, value in robust_config["weights"].items()
                },
                cache_root=job_root / "stage2_cache" / "validation",
                max_total_rows=int(robust_config["max_total_rows"]),
                minimum_free_cache_bytes=int(
                    robust_config.get("minimum_free_cache_bytes", 0)
                ),
            )
        mortality_model, los_model = _fit_xgb_pair(
            (
                robust_train_flat["design"]
                if robust_train_flat is not None else designs["train"]
            ),
            (
                robust_validation_flat["design"]
                if robust_validation_flat is not None else designs["validation"]
            ),
            (
                robust_train_flat["mortality"]
                if robust_train_flat is not None else labels["train"]
            ),
            (
                robust_validation_flat["mortality"]
                if robust_validation_flat is not None else labels["validation"]
            ),
            (
                robust_train_flat["los"]
                if robust_train_flat is not None else los_labels["train"]
            ),
            (
                robust_validation_flat["los"]
                if robust_validation_flat is not None else los_labels["validation"]
            ),
            seed=seed, n_estimators=effective_estimators, device=device,
            early_stopping_rounds=early_stopping_rounds,
            train_sample_weight=(
                robust_train_flat["sample_weight"]
                if robust_train_flat is not None else None
            ),
            validation_sample_weight=(
                robust_validation_flat["sample_weight"]
                if robust_validation_flat is not None else None
            ),
        )
        resource_safe_point(f"{model_name}_xgboost_fit_complete", force_cleanup=True)
        validation_probability, validation_los_prediction = _xgb_predictions(
            mortality_model, los_model, designs["validation"]
        )
        test_probability, test_los_prediction = _xgb_predictions(
            mortality_model, los_model, designs["test"]
        )
        mortality_path = job_root / "mortality_model.ubj"
        los_path = job_root / "los_model.ubj"
        mortality_model.save_model(mortality_path)
        los_model.save_model(los_path)
        artifacts.extend([mortality_path, los_path])
        training_audit["design_width"] = int(designs["train"].shape[1])
        if robust_validation_flat is not None and robust_train_flat is not None:
            selection_probability, _ = _xgb_predictions(
                mortality_model,
                los_model,
                robust_validation_flat["design"],
            )
            selection_labels = np.asarray(
                robust_validation_flat["mortality"], dtype=np.int8
            ).copy()
            selection_sample_weight = np.asarray(
                robust_validation_flat["sample_weight"], dtype=np.float64
            ).copy()
            training_audit["temporal_robust_stage2"] = {
                "enabled": True,
                "training_regime": "stress_augmented",
                "train_views": robust_train_flat["audit"],
                "validation_views": robust_validation_flat["audit"],
                "same_view_contract_across_architectures": True,
                "target_information_used": False,
            }
            _cleanup_stage2_cache(
                job_root, robust_train_flat, robust_validation_flat
            )
    else:
        mamba, mamba_audit, loaders = train_mamba_encoder(
            arrays,
            indices,
            normalization,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            training_config=mamba_config,
            bounded=bounded,
        )
        encoder_path = job_root / "mamba_encoder.pt"
        _save_state(encoder_path, mamba)
        artifacts.append(encoder_path)
        training_audit["mamba"] = mamba_audit
        extracted = {
            split: _predict_mamba(
                mamba,
                loaders[split],
                device,
                include_embeddings=True,
                ignore_static=True,
            )
            for split in ("train", "validation", "test")
        }
        for split, result in extracted.items():
            # Stage-2 joins these rows positionally; refuse any misalignment.
            if not np.array_equal(result["mortality"], labels[split]):
                raise RuntimeError(f"stage-2 {split} embeddings are not row-aligned with labels")
        if model_name == "Mamba-z-only":
            designs = {split: result["embedding"] for split, result in extracted.items()}
        else:
            flat = {
                split: _flat_design_split("XGBOOST-SUMMARY+STATIC", arrays, rows)[0]
                for split, rows in indices.items()
            }
            designs = {
                split: np.concatenate([extracted[split]["embedding"], flat[split]], axis=1)
                for split in extracted
            }
        robust_config = (training_config or {}).get("temporal_robust_stage2", {})
        robust_train: dict[str, np.ndarray] | None = None
        robust_validation: dict[str, np.ndarray] | None = None
        if (
            model_name in {"MABOOST", "Hybrid-plain"}
            and bool(robust_config.get("enabled", False))
            and not bounded
        ):
            scenarios = [str(value) for value in robust_config["scenarios"]]
            class_weights = {
                key: float(value) for key, value in robust_config["weights"].items()
                if key != "full"
            }
            full_strict_train = _flat_design_split(
                "XGBOOST-FLAT", arrays, indices["train"]
            )[0]
            full_flat_head_train = _flat_design_split(
                "Flat-head", arrays, indices["train"]
            )[0]
            robust_train = _build_temporal_robust_bundle(
                mamba,
                arrays,
                indices["train"],
                normalization,
                scenarios=scenarios,
                seed=seed,
                device=device,
                max_rows_per_view=int(robust_config["max_train_rows_per_view"]),
                class_weights=class_weights,
                cache_root=job_root / "stage2_cache" / "train",
                base_bundle={
                    "hybrid": designs["train"],
                    "strict_flat": full_strict_train,
                    "flat_head": full_flat_head_train,
                    "mortality": labels["train"],
                    "los": los_labels["train"],
                },
                base_mass=float(robust_config["weights"]["full"]),
                max_total_rows=int(robust_config["max_total_rows"]),
                minimum_free_cache_bytes=int(
                    robust_config.get("minimum_free_cache_bytes", 0)
                ),
            )
            del full_strict_train, full_flat_head_train
            resource_safe_point("stage2_train_cache_complete", force_cleanup=True)
            robust_validation = _build_temporal_robust_bundle(
                mamba,
                arrays,
                indices["validation"],
                normalization,
                scenarios=scenarios,
                seed=seed + 1_000_003,
                device=device,
                max_rows_per_view=int(robust_config["max_validation_rows_per_view"]),
                class_weights=class_weights,
                cache_root=job_root / "stage2_cache" / "validation",
                base_bundle={
                    "hybrid": designs["validation"],
                    "strict_flat": _flat_design_split(
                        "XGBOOST-FLAT", arrays, indices["validation"]
                    )[0],
                    "flat_head": _flat_design_split(
                        "Flat-head", arrays, indices["validation"]
                    )[0],
                    "mortality": labels["validation"],
                    "los": los_labels["validation"],
                },
                base_mass=float(robust_config["weights"]["full"]),
                max_total_rows=int(robust_config["max_total_rows"]),
                minimum_free_cache_bytes=int(
                    robust_config.get("minimum_free_cache_bytes", 0)
                ),
            )
            full_count = len(indices["train"])
            training_audit["temporal_robust_stage2"] = {
                "enabled": True,
                "scenarios": scenarios,
                "full_train_rows": int(full_count),
                "combined_train_rows": int(len(robust_train["mortality"])),
                "augmented_validation_rows": int(len(robust_validation["mortality"])),
                "configured_mass": {
                    key: float(value)
                    for key, value in robust_config["weights"].items()
                },
                "train_views": robust_train["audit"],
                "validation_views": robust_validation["audit"],
                "same_view_contract_across_architectures": True,
                "disk_backed_cache": True,
                "target_information_used": False,
            }
        hybrid_train_features = (
            robust_train["hybrid"] if robust_train is not None else designs["train"]
        )
        hybrid_train_mortality = (
            robust_train["mortality"] if robust_train is not None else labels["train"]
        )
        hybrid_train_los = robust_train["los"] if robust_train is not None else los_labels["train"]
        hybrid_mortality, hybrid_los = _fit_xgb_pair(
            hybrid_train_features,
            robust_validation["hybrid"] if robust_validation is not None else designs["validation"],
            hybrid_train_mortality,
            robust_validation["mortality"] if robust_validation is not None else labels["validation"],
            hybrid_train_los,
            robust_validation["los"] if robust_validation is not None else los_labels["validation"],
            seed=seed, n_estimators=effective_estimators, device=device,
            early_stopping_rounds=early_stopping_rounds,
            train_sample_weight=(
                robust_train["sample_weight"] if robust_train is not None else None
            ),
            validation_sample_weight=(
                robust_validation["sample_weight"]
                if robust_validation is not None else None
            ),
        )
        resource_safe_point("stage2_hybrid_fit_complete", force_cleanup=True)
        hybrid_val_probability, hybrid_val_los = _xgb_predictions(
            hybrid_mortality, hybrid_los, designs["validation"]
        )
        hybrid_test_probability, hybrid_test_los = _xgb_predictions(
            hybrid_mortality, hybrid_los, designs["test"]
        )
        hybrid_mortality_path = job_root / "hybrid_mortality.ubj"
        hybrid_los_path = job_root / "hybrid_los.ubj"
        hybrid_mortality.save_model(hybrid_mortality_path)
        hybrid_los.save_model(hybrid_los_path)
        artifacts.extend([hybrid_mortality_path, hybrid_los_path])
        training_audit["design_width"] = int(designs["train"].shape[1])

        if model_name != "MABOOST":
            validation_probability = hybrid_val_probability
            validation_los_prediction = hybrid_val_los
            test_probability = hybrid_test_probability
            test_los_prediction = hybrid_test_los
            if robust_validation is not None:
                selection_probability, _ = _xgb_predictions(
                    hybrid_mortality,
                    hybrid_los,
                    robust_validation["hybrid"],
                )
                selection_labels = np.asarray(
                    robust_validation["mortality"], dtype=np.int8
                ).copy()
                selection_sample_weight = np.asarray(
                    robust_validation["sample_weight"], dtype=np.float64
                ).copy()
        else:
            flat_head_splits = (
                ("validation", "test") if robust_train is not None
                else ("train", "validation", "test")
            )
            flat_head = {
                split: _flat_design_split("Flat-head", arrays, indices[split])[0]
                for split in flat_head_splits
            }
            flat_mortality, flat_los = _fit_xgb_pair(
                robust_train["flat_head"] if robust_train is not None else flat_head["train"],
                robust_validation["flat_head"] if robust_validation is not None else flat_head["validation"],
                robust_train["mortality"] if robust_train is not None else labels["train"],
                robust_validation["mortality"] if robust_validation is not None else labels["validation"],
                robust_train["los"] if robust_train is not None else los_labels["train"],
                robust_validation["los"] if robust_validation is not None else los_labels["validation"],
                seed=seed, n_estimators=effective_estimators, device=device,
                early_stopping_rounds=early_stopping_rounds,
                train_sample_weight=(
                    robust_train["sample_weight"] if robust_train is not None else None
                ),
                validation_sample_weight=(
                    robust_validation["sample_weight"]
                    if robust_validation is not None else None
                ),
            )
            resource_safe_point("stage2_flat_head_fit_complete", force_cleanup=True)
            flat_val_probability, flat_val_los = _xgb_predictions(
                flat_mortality, flat_los, flat_head["validation"]
            )
            flat_test_probability, flat_test_los = _xgb_predictions(
                flat_mortality, flat_los, flat_head["test"]
            )
            strict_splits = (
                ("validation", "test") if robust_train is not None
                else ("train", "validation", "test")
            )
            strict_flat = {
                split: _flat_design_split("XGBOOST-FLAT", arrays, indices[split])[0]
                for split in strict_splits
            }
            strict_mortality, strict_los = _fit_xgb_pair(
                robust_train["strict_flat"] if robust_train is not None else strict_flat["train"],
                robust_validation["strict_flat"] if robust_validation is not None else strict_flat["validation"],
                robust_train["mortality"] if robust_train is not None else labels["train"],
                robust_validation["mortality"] if robust_validation is not None else labels["validation"],
                robust_train["los"] if robust_train is not None else los_labels["train"],
                robust_validation["los"] if robust_validation is not None else los_labels["validation"],
                seed=seed,
                n_estimators=effective_estimators,
                device=device,
                early_stopping_rounds=early_stopping_rounds,
                train_sample_weight=(
                    robust_train["sample_weight"] if robust_train is not None else None
                ),
                validation_sample_weight=(
                    robust_validation["sample_weight"]
                    if robust_validation is not None else None
                ),
            )
            resource_safe_point("stage2_strict_flat_fit_complete", force_cleanup=True)
            strict_val_probability, strict_val_los = _xgb_predictions(
                strict_mortality, strict_los, strict_flat["validation"]
            )
            strict_test_probability, strict_test_los = _xgb_predictions(
                strict_mortality, strict_los, strict_flat["test"]
            )
            simplex_step = float(
                (training_config or {}).get("selection_and_calibration", {}).get(
                    "fusion_simplex_step", 0.05
                )
            )
            if robust_validation is not None:
                fusion_hybrid_probability, fusion_hybrid_los = _xgb_predictions(
                    hybrid_mortality, hybrid_los, robust_validation["hybrid"]
                )
                fusion_strict_probability, fusion_strict_los = _xgb_predictions(
                    strict_mortality, strict_los, robust_validation["strict_flat"]
                )
                fusion_flat_probability, fusion_flat_los = _xgb_predictions(
                    flat_mortality, flat_los, robust_validation["flat_head"]
                )
                fusion_labels = robust_validation["mortality"]
                fusion_los_labels = robust_validation["los"]
            else:
                fusion_hybrid_probability, fusion_hybrid_los = (
                    hybrid_val_probability, hybrid_val_los
                )
                fusion_strict_probability, fusion_strict_los = (
                    strict_val_probability, strict_val_los
                )
                fusion_flat_probability, fusion_flat_los = (
                    flat_val_probability, flat_val_los
                )
                fusion_labels = labels["validation"]
                fusion_los_labels = los_labels["validation"]
            fusion = select_three_component_fusion(
                np.column_stack([
                    fusion_hybrid_probability,
                    fusion_strict_probability,
                    fusion_flat_probability,
                ]),
                fusion_labels,
                source_dataset="eicu" if "eicu" in str(job_root) else "mimic_iv",
                split="validation",
                simplex_step=simplex_step,
                sample_weight=(
                    robust_validation["sample_weight"]
                    if robust_validation is not None else None
                ),
            )
            los_fusion = _select_los_fusion(
                np.column_stack([
                    fusion_hybrid_los,
                    fusion_strict_los,
                    fusion_flat_los,
                ]),
                fusion_los_labels,
                simplex_step=simplex_step,
                sample_weight=(
                    robust_validation["sample_weight"]
                    if robust_validation is not None else None
                ),
            )
            validation_probability = np.column_stack([
                hybrid_val_probability,
                strict_val_probability,
                flat_val_probability,
            ]) @ np.asarray(fusion["weights"])
            test_probability = np.column_stack([
                hybrid_test_probability,
                strict_test_probability,
                flat_test_probability,
            ]) @ np.asarray(fusion["weights"])
            validation_los_prediction = np.column_stack([
                hybrid_val_los,
                strict_val_los,
                flat_val_los,
            ]) @ np.asarray(los_fusion["weights"])
            test_los_prediction = np.column_stack([
                hybrid_test_los,
                strict_test_los,
                flat_test_los,
            ]) @ np.asarray(los_fusion["weights"])
            flat_mortality_path = job_root / "flat_head_mortality.ubj"
            flat_los_path = job_root / "flat_head_los.ubj"
            flat_mortality.save_model(flat_mortality_path)
            flat_los.save_model(flat_los_path)
            strict_mortality_path = job_root / "xgboost_flat_mortality.ubj"
            strict_los_path = job_root / "xgboost_flat_los.ubj"
            strict_mortality.save_model(strict_mortality_path)
            strict_los.save_model(strict_los_path)
            artifacts.extend([
                flat_mortality_path,
                flat_los_path,
                strict_mortality_path,
                strict_los_path,
            ])
            training_audit["fusion"] = fusion
            training_audit["los_fusion"] = los_fusion
            training_audit["official_stage2_variant"] = "maboost_tri_fusion"
            if robust_validation is not None:
                selection_probability = np.column_stack([
                    fusion_hybrid_probability,
                    fusion_strict_probability,
                    fusion_flat_probability,
                ]) @ np.asarray(fusion["weights"])
                selection_labels = np.asarray(fusion_labels, dtype=np.int8).copy()
                selection_sample_weight = np.asarray(
                    robust_validation["sample_weight"], dtype=np.float64
                ).copy()
        if robust_train is not None:
            del hybrid_train_features, hybrid_train_mortality, hybrid_train_los
            _cleanup_stage2_cache(job_root, robust_train, robust_validation)
        del mamba, loaders, extracted
        torch.cuda.empty_cache()
        gc.collect()

    source_dataset = "eicu" if "eicu" in str(job_root) else "mimic_iv"
    training_audit["los_target"] = "remaining_from_fixed_6h_landmark"
    training_audit["xgboost_early_stopping_rounds"] = early_stopping_rounds
    if selection_probability is None:
        selection_probability = validation_probability
        selection_labels = labels["validation"]
    calibrator = fit_source_isotonic_calibration(
        selection_probability,
        selection_labels,
        source_dataset=source_dataset,
        split="validation",
        sample_weight=selection_sample_weight,
    )
    calibrated_validation = calibrator.apply(validation_probability)
    calibrated_test = calibrator.apply(test_probability)
    threshold = select_source_threshold(
        calibrator.apply(selection_probability),
        selection_labels,
        source_dataset=source_dataset,
        split="validation",
        sample_weight=selection_sample_weight,
    )
    choices = {
        "calibration": calibrator.to_dict(),
        "threshold": threshold,
        "training_audit": training_audit,
        "target_information_used": False,
    }
    predictions = {
        "mortality_probability": calibrated_test.astype(np.float32),
        "los_prediction": test_los_prediction.astype(np.float32),
        "mortality_label": labels["test"].astype(np.int8),
        "los_label": los_labels["test"].astype(np.float32),
        "anchor_elapsed_days": los_anchors["test"].astype(np.float32),
        "stay_key": np.asarray(arrays["stay_keys"])[indices["test"]],
        "group_key": np.asarray(arrays["group_keys"])[indices["test"]],
    }
    metrics = canonical_metric_row(
        dataset=source_dataset,
        direction="source_internal",
        population="locked_source_test",
        model=model_name,
        seed=seed,
        scenario="full_6h",
        mortality_labels=predictions["mortality_label"],
        mortality_probabilities=predictions["mortality_probability"],
        los_labels=predictions["los_label"],
        los_predictions=predictions["los_prediction"],
        calibration_status="source_validation_calibrated",
    )
    return {"metrics": metrics, "choices": choices}, predictions, artifacts


class DeepMultitaskWrapper(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        los_contract_version: str = CONTRACT_VERSION,
    ) -> None:
        super().__init__()
        if los_contract_version not in {
            CONTRACT_VERSION,
            LEGACY_CONTRACT_VERSION,
        }:
            raise ValueError(f"unknown auxiliary LOS contract: {los_contract_version}")
        self.base = base
        self.los_contract_version = los_contract_version
        first = next(
            module
            for module in base.binary_head.modules()
            if isinstance(module, nn.Linear)
        )
        self.los_input_dimension = int(first.in_features)
        # elementwise_affine=False makes normalization target-independent and
        # preserves state-dict compatibility with pre-correction checkpoints.
        self.los_input_normalization = nn.LayerNorm(
            self.los_input_dimension,
            elementwise_affine=False,
        )
        self.los_head = nn.Sequential(
            nn.Linear(self.los_input_dimension, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self._features: torch.Tensor | None = None
        self._normalized_features: torch.Tensor | None = None
        self._pre_link_log_los: torch.Tensor | None = None
        self._hook = base.binary_head.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._features = inputs[0]

    def los_diagnostic_tensors(self) -> dict[str, torch.Tensor]:
        if (
            self._features is None
            or self._normalized_features is None
            or self._pre_link_log_los is None
        ):
            raise RuntimeError("auxiliary LOS diagnostic tensors are unavailable")
        return {
            "representation_pre_normalization": self._features,
            "representation_post_normalization": self._normalized_features,
            "pre_link_raw_log_los": self._pre_link_log_los,
        }

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = self.base(*inputs, labels=None)
        if self._features is None:
            raise RuntimeError("deep baseline feature hook did not execute")
        if self.los_contract_version == CONTRACT_VERSION:
            normalized = self.los_input_normalization(self._features.float())
            pre_link = self.los_head(normalized).squeeze(-1)
            pred_log_los = bounded_log_los(pre_link)
        else:
            normalized = self._features
            pre_link = self.los_head(normalized).squeeze(-1)
            pred_log_los = pre_link
        self._normalized_features = normalized
        self._pre_link_log_los = pre_link
        return probability, pred_log_los


def _adapt_deep_batch(
    model_name: str,
    batch: tuple[torch.Tensor, ...],
    device: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    values, tau, mask, static, mortality, los = batch
    values = values.to(device, non_blocking=True)
    tau = tau.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    static = static.to(device, non_blocking=True)
    if model_name in {"GRU-D", "LSTM", "Transformer"}:
        inputs = (values, tau, mask, static)
    elif model_name in {"TCN", "SAnD", "DuETT"}:
        delta = tau.unsqueeze(-1).expand_as(values) / 3600.0
        inputs = (torch.cat([values, mask, delta], dim=-1), static)
    elif model_name == "STraTS":
        batch_size = values.shape[0]
        max_obs = 512
        sparse_values = torch.zeros(batch_size, max_obs, device=device)
        times = torch.zeros(batch_size, max_obs, device=device)
        variables = torch.zeros(batch_size, max_obs, dtype=torch.long, device=device)
        obs_mask = torch.zeros(batch_size, max_obs, device=device)
        cumulative = tau.cumsum(dim=1) / 3600.0
        for row in range(batch_size):
            positions = torch.nonzero(mask[row] > 0, as_tuple=False)[:max_obs]
            count = len(positions)
            if count:
                time_index = positions[:, 0]
                feature_index = positions[:, 1]
                sparse_values[row, :count] = values[row, time_index, feature_index]
                times[row, :count] = cumulative[row, time_index]
                variables[row, :count] = feature_index
                obs_mask[row, :count] = 1.0
        inputs = (sparse_values, times, variables, obs_mask, static)
    elif model_name == "InterpNet":
        cumulative = tau.cumsum(dim=1) / 3600.0
        inputs = (values, mask, cumulative, torch.zeros_like(mask), static)
    else:
        raise ValueError(model_name)
    return inputs, mortality.to(device), los.to(device)


@torch.no_grad()
def _predict_deep(
    model_name: str,
    model: DeepMultitaskWrapper,
    loader: DataLoader,
    device: str,
    *,
    diagnostic_recorder: AggregateNumericalDiagnostics | None = None,
    diagnostic_stage: str = "inference",
) -> dict[str, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    los_predictions: list[np.ndarray] = []
    mortality_labels: list[np.ndarray] = []
    los_labels: list[np.ndarray] = []
    try:
        for batch_index, batch in enumerate(loader):
            inputs, mortality, los = _adapt_deep_batch(model_name, batch, device)
            probability, pred_log_los = model(*inputs)
            diagnostic_tensors = model.los_diagnostic_tensors()
            raw_log_los = diagnostic_tensors["pre_link_raw_log_los"]
            if diagnostic_recorder is not None:
                for name, tensor in diagnostic_tensors.items():
                    diagnostic_recorder.update(
                        diagnostic_stage,
                        name,
                        tensor,
                    )
                diagnostic_recorder.update(
                    diagnostic_stage,
                    "probability",
                    probability,
                )
                diagnostic_recorder.update(
                    diagnostic_stage,
                    "bounded_log_los",
                    pred_log_los,
                    saturation_bound=(
                        LOG_LOS_BOUND
                        if model.los_contract_version == CONTRACT_VERSION
                        else None
                    ),
                )
            if not torch.isfinite(probability).all():
                raise RuntimeError(
                    f"{model_name} emitted non-finite mortality predictions "
                    f"during {diagnostic_stage} batch {batch_index}"
                )
            if not torch.isfinite(raw_log_los).all():
                raise RuntimeError(
                    f"{model_name} emitted non-finite pre-link raw log-LOS "
                    f"during {diagnostic_stage} batch {batch_index}"
                )
            if not torch.isfinite(pred_log_los).all():
                raise RuntimeError(
                    f"{model_name} emitted non-finite bounded log-LOS "
                    f"during {diagnostic_stage} batch {batch_index}"
                )
            transformed_los = torch.expm1(pred_log_los.float()).clamp(min=0)
            if diagnostic_recorder is not None:
                diagnostic_recorder.update(
                    diagnostic_stage,
                    "los_after_expm1",
                    transformed_los,
                )
            if not torch.isfinite(transformed_los).all():
                raw_max = float(raw_log_los.detach().float().max().cpu())
                bounded_max = float(pred_log_los.detach().float().max().cpu())
                raise RuntimeError(
                    f"{model_name} LOS inverse transform overflow during "
                    f"{diagnostic_stage} batch {batch_index}; "
                    f"maximum_pre_link_raw_log_los={raw_max}; "
                    f"maximum_model_log_los={bounded_max}"
                )
            probabilities.append(probability.float().cpu().numpy())
            los_predictions.append(transformed_los.cpu().numpy())
            mortality_labels.append(mortality.cpu().numpy())
            los_labels.append(los.cpu().numpy())
    except BaseException as error:
        if diagnostic_recorder is not None:
            diagnostic_recorder.persist(
                status="failed",
                failure=f"{type(error).__name__}: {error}",
            )
        raise
    return {
        "probability": np.concatenate(probabilities).astype(np.float32),
        "los_prediction": np.concatenate(los_predictions).astype(np.float32),
        "mortality": np.concatenate(mortality_labels).astype(np.int8),
        "los": np.concatenate(los_labels).astype(np.float32),
    }


def _deep_correction_binding(*, active_correction_sha256, los_correction_sha256=None, predecessor_correction_sha256=None):
    from reproduction_contract import require_run
    registered = require_run()["record_sha256"]
    if active_correction_sha256 != registered:
        raise RuntimeError("Deep model belongs to a different reproduction run")
    return {"reproduction_run_sha256": registered}


def train_deep_baseline(
    model_name: str,
    arrays: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    device: str,
    job_root: Path,
    bounded: bool = False,
    correction_sha256: str = "unbound_nonproduction",
    los_correction_sha256: str | None = None,
    predecessor_correction_sha256: str | None = None,
    source_dataset: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[Path]]:
    resolved_source = source_dataset or (
        "eicu" if "eicu" in str(job_root) else
        "mimic_iv" if "mimic_iv" in str(job_root) else
        "synthetic"
    )
    loaders = make_loaders(
        arrays,
        indices,
        normalization,
        batch_size=16 if model_name == "STraTS" else 32,
        seed=seed,
    )
    pos_rate = float(np.asarray(arrays["mortality"])[indices["train"]].mean())
    base = instantiate_deep_model(
        model_name,
        pos_rate=pos_rate,
        hidden_dim=256,
        sequence_length=int(np.asarray(arrays["values"]).shape[1]),
    )
    model = DeepMultitaskWrapper(
        base,
        los_contract_version=CONTRACT_VERSION,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor((1.0 - pos_rate) / max(pos_rate, 1e-6), device=device)
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    effective_epochs = 1 if bounded else max_epochs
    diagnostics_path = job_root / "numerical_diagnostics_aggregate.json"
    diagnostics = AggregateNumericalDiagnostics(
        diagnostics_path,
        job_id=job_root.name,
        model=model_name,
        seed=seed,
        source_dataset=resolved_source,
        correction_sha256=correction_sha256,
    )
    last_epoch_path = job_root / "deep_multitask_last_epoch_private.pt"
    try:
        for epoch in range(1, effective_epochs + 1):
            model.train()
            losses: list[float] = []
            gradient_norms: list[float] = []
            train_stage = f"epoch_{epoch:03d}_train"
            for batch_index, batch in enumerate(loaders["train"]):
                inputs, mortality, los = _adapt_deep_batch(model_name, batch, device)
                optimizer.zero_grad(set_to_none=True)
                probability, pred_log_los = model(*inputs)
                diagnostic_tensors = model.los_diagnostic_tensors()
                for name, tensor in diagnostic_tensors.items():
                    diagnostics.update(train_stage, name, tensor)
                diagnostics.update(train_stage, "probability", probability)
                diagnostics.update(
                    train_stage,
                    "bounded_log_los",
                    pred_log_los,
                    saturation_bound=LOG_LOS_BOUND,
                )
                transformed_los = torch.expm1(pred_log_los.float()).clamp(min=0)
                diagnostics.update(
                    train_stage,
                    "los_after_expm1",
                    transformed_los,
                )
                if not torch.isfinite(diagnostic_tensors["pre_link_raw_log_los"]).all():
                    raise RuntimeError(
                        f"non-finite {model_name} pre-link raw log-LOS during training"
                    )
                if not torch.isfinite(pred_log_los).all() or not torch.isfinite(
                    transformed_los
                ).all():
                    raise RuntimeError(
                        f"non-finite {model_name} corrected LOS output during training"
                    )
                logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
                loss = F.binary_cross_entropy_with_logits(
                    logits, mortality.float(), pos_weight=pos_weight
                )
                loss = loss + 0.20 * F.smooth_l1_loss(
                    pred_log_los,
                    torch.log1p(los),
                )
                diagnostics.update(train_stage, "training_loss", loss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite {model_name} training loss")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                diagnostics.update(train_stage, "gradient_norm", gradient_norm)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"non-finite {model_name} gradient norm")
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                gradient_value = float(gradient_norm.detach().cpu())
                losses.append(loss_value)
                gradient_norms.append(gradient_value)
                resource_safe_point(
                    f"{model_name}_epoch{epoch}_batch{batch_index}"
                )
                if bounded and batch_index >= 1:
                    break
            diagnostics.persist(status="running")
            validation = _predict_deep(
                model_name,
                model,
                loaders["validation"],
                device,
                diagnostic_recorder=diagnostics,
                diagnostic_stage=f"epoch_{epoch:03d}_validation",
            )
            score = float(average_precision_score(
                validation["mortality"], validation["probability"]
            ))
            train_loss = float(np.mean(losses))
            last_checkpoint_hash = _save_state_with_hash(last_epoch_path, model)
            history.append({
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_auprc": score,
            })
            diagnostics.record_epoch({
                "epoch": int(epoch),
                "train_loss": train_loss,
                "validation_auprc": score,
                "gradient_norm_max": float(max(gradient_norms)),
                "last_epoch_checkpoint_sha256": last_checkpoint_hash,
            })
            diagnostics.persist(status="running")
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            del validation
            resource_safe_point(
                f"{model_name}_epoch{epoch}_complete", force_cleanup=True
            )
            if epochs_without_improvement >= patience:
                break
        if best_state is None:
            raise RuntimeError(
                f"{model_name} did not produce a finite validation checkpoint"
            )
        model.load_state_dict({
            key: value.to(device) for key, value in best_state.items()
        })
        validation = _predict_deep(
            model_name,
            model,
            loaders["validation"],
            device,
            diagnostic_recorder=diagnostics,
            diagnostic_stage="final_validation",
        )
        test = _predict_deep(
            model_name,
            model,
            loaders["test"],
            device,
            diagnostic_recorder=diagnostics,
            diagnostic_stage="final_test",
        )
        calibrator = fit_source_isotonic_calibration(
            validation["probability"], validation["mortality"],
            source_dataset=resolved_source, split="validation",
        )
        validation_probability = calibrator.apply(validation["probability"])
        test_probability = calibrator.apply(test["probability"])
        threshold = select_source_threshold(
            validation_probability, validation["mortality"],
            source_dataset=resolved_source, split="validation",
        )
        model_path = job_root / "deep_multitask.pt"
        _save_state_with_hash(model_path, model)
        predictions = {
            "mortality_probability": test_probability,
            "los_prediction": test["los_prediction"],
            "mortality_label": test["mortality"],
            "los_label": test["los"],
            "anchor_elapsed_days": remaining_los_for_rows(
                arrays, indices["test"]
            )[1].astype(np.float32),
            "stay_key": np.asarray(arrays["stay_keys"])[indices["test"]],
            "group_key": np.asarray(arrays["group_keys"])[indices["test"]],
        }
        metrics = canonical_metric_row(
            dataset=resolved_source,
            direction="source_internal",
            population="locked_source_test",
            model=model_name,
            seed=seed,
            scenario="full_6h",
            mortality_labels=predictions["mortality_label"],
            mortality_probabilities=predictions["mortality_probability"],
            los_labels=predictions["los_label"],
            los_predictions=predictions["los_prediction"],
            calibration_status="source_validation_calibrated",
        )
        diagnostics_hash = diagnostics.persist(status="complete")
        choices = {
            "calibration": calibrator.to_dict(),
            "threshold": threshold,
            "best_validation_auprc": float(best_score),
            "epochs_completed": len(history),
            "history": history,
            "wall_seconds": float(time.perf_counter() - started),
            "los_target": "remaining_from_fixed_6h_landmark",
            "los_contract_version": CONTRACT_VERSION,
            "los_log_bound": LOG_LOS_BOUND,
            **_deep_correction_binding(
                active_correction_sha256=correction_sha256,
                los_correction_sha256=los_correction_sha256,
                predecessor_correction_sha256=predecessor_correction_sha256,
            ),
            "numerical_diagnostics_sha256": diagnostics_hash,
            "target_information_used": False,
        }
        artifacts = [
            model_path,
            model_path.with_suffix(".pt.sha256"),
            last_epoch_path,
            last_epoch_path.with_suffix(".pt.sha256"),
            diagnostics_path,
            diagnostics_path.with_suffix(".json.sha256"),
        ]
        return {"metrics": metrics, "choices": choices}, predictions, artifacts
    except BaseException as error:
        diagnostics.persist(
            status="failed",
            failure=f"{type(error).__name__}: {error}",
        )
        raise


def save_training_payloads(
    job_root: Path,
    normalization: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    normalization_path = job_root / "normalization_source_train_only.npz"
    predictions_path = job_root / "predictions_private.npz"
    _save_npz(normalization_path, **normalization)
    _save_npz(predictions_path, **predictions)
    return normalization_path, predictions_path
