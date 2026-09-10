from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor

from auxiliary_los_contract import CONTRACT_VERSION, DEEP_BASELINE_MODELS
from v13_guard import sha256_file
from v13_model_registry import build_flat_design, instantiate_deep_model, instantiate_maboost
from v13_selection import IsotonicCalibration
from v13_training import DeepMultitaskWrapper, _adapt_deep_batch


def load_normalization(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        normalization = {name: payload[name] for name in payload.files}
    required = {"dynamic_mean", "dynamic_std", "static_mean", "static_std"}
    if not required.issubset(normalization):
        raise RuntimeError(f"source normalization is incomplete: {path}")
    return normalization


def load_source_choices(path: Path) -> dict[str, Any]:
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"source choices or hash is missing: {path}")
    if sha256_file(path) != hash_path.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError(f"source-choice integrity failure: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target_information_used"):
        raise PermissionError("locked source choices contain target information")
    return payload


def _load_xgb_pair(
    mortality_path: Path,
    los_path: Path,
) -> tuple[XGBClassifier, XGBRegressor]:
    mortality = XGBClassifier()
    mortality.load_model(mortality_path)
    los = XGBRegressor()
    los.load_model(los_path)
    return mortality, los


class ModelPredictor:
    """Reload one immutable Phase 07 source model without opening target labels."""

    def __init__(
        self,
        model_name: str,
        job_root: Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.job_root = job_root
        self.device = device
        self.normalization = load_normalization(
            job_root / "normalization_source_train_only.npz"
        )
        self.source_payload = load_source_choices(job_root / "source_choices.json")
        self.source_dataset = str(self.source_payload["source_dataset"])
        self.choices = self.source_payload["choices"]
        if self.model_name in DEEP_BASELINE_MODELS:
            if self.choices.get("los_contract_version") != CONTRACT_VERSION:
                raise RuntimeError(
                    "deep checkpoint is not bound to the correction-03 LOS contract"
                )
            from reproduction_contract import require_run
            if self.choices.get("reproduction_run_sha256") != require_run()["record_sha256"]:
                raise RuntimeError("Deep checkpoint belongs to a different reproduction run")
        calibration = self.choices["calibration"]
        if calibration["source_dataset"] != self.source_dataset:
            raise RuntimeError("calibrator source differs from the locked source model")
        self.calibrator = IsotonicCalibration(
            x_thresholds=tuple(float(value) for value in calibration["x_thresholds"]),
            y_thresholds=tuple(float(value) for value in calibration["y_thresholds"]),
            source_dataset=calibration["source_dataset"],
        )
        self.mamba: torch.nn.Module | None = None
        self.deep: DeepMultitaskWrapper | None = None
        self.mortality_model: XGBClassifier | None = None
        self.los_model: XGBRegressor | None = None
        self.hybrid_mortality: XGBClassifier | None = None
        self.hybrid_los: XGBRegressor | None = None
        self.flat_mortality: XGBClassifier | None = None
        self.flat_los: XGBRegressor | None = None
        self.strict_flat_mortality: XGBClassifier | None = None
        self.strict_flat_los: XGBRegressor | None = None
        self._load()

    def _load(self) -> None:
        if self.model_name in {
            "XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC", "Flat-head"
        }:
            self.mortality_model, self.los_model = _load_xgb_pair(
                self.job_root / "mortality_model.ubj",
                self.job_root / "los_model.ubj",
            )
            return
        if self.model_name in {"MABOOST", "Hybrid-plain", "Mamba-z-only"}:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Mamba checkpoint inference requires visible CUDA")
            self.mamba = instantiate_maboost(
                hidden_dim=128,
                d_state=32,
                d_conv=4,
                expand=2,
                n_layers=4,
                n_heads=4,
                dropout=0.1,
                event_max_tokens=96,
            ).to(self.device)
            self.mamba.load_state_dict(torch.load(
                self.job_root / "mamba_encoder.pt",
                map_location=self.device,
                weights_only=True,
            ))
            self.mamba.eval()
            self.hybrid_mortality, self.hybrid_los = _load_xgb_pair(
                self.job_root / "hybrid_mortality.ubj",
                self.job_root / "hybrid_los.ubj",
            )
            if self.model_name == "MABOOST":
                self.flat_mortality, self.flat_los = _load_xgb_pair(
                    self.job_root / "flat_head_mortality.ubj",
                    self.job_root / "flat_head_los.ubj",
                )
                self.strict_flat_mortality, self.strict_flat_los = _load_xgb_pair(
                    self.job_root / "xgboost_flat_mortality.ubj",
                    self.job_root / "xgboost_flat_los.ubj",
                )
            return
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("deep checkpoint inference requires visible CUDA")
        base = instantiate_deep_model(
            self.model_name,
            pos_rate=0.5,
            hidden_dim=256,
            sequence_length=96,
        )
        self.deep = DeepMultitaskWrapper(base).to(self.device)
        self.deep.load_state_dict(torch.load(
            self.job_root / "deep_multitask.pt",
            map_location=self.device,
            weights_only=True,
        ))
        self.deep.eval()

    def _normalized_tensors(
        self,
        values: np.ndarray,
        tau: np.ndarray,
        mask: np.ndarray,
        static: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_array = (np.asarray(mask) > 0).astype(np.float32)
        clean_values = np.nan_to_num(
            np.asarray(values, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized_values = np.where(
            mask_array > 0,
            (clean_values - self.normalization["dynamic_mean"])
            / self.normalization["dynamic_std"],
            0.0,
        ).clip(-5.0, 5.0).astype(np.float32)
        clean_tau = np.nan_to_num(
            np.asarray(tau, dtype=np.float32),
            nan=60.0,
            posinf=3600.0,
            neginf=1.0,
        ).clip(1.0, 86400.0)
        clean_static = np.nan_to_num(
            np.asarray(static, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized_static = (
            (clean_static - self.normalization["static_mean"])
            / self.normalization["static_std"]
        ).clip(-10.0, 10.0).astype(np.float32)
        return (
            torch.from_numpy(normalized_values),
            torch.from_numpy(clean_tau),
            torch.from_numpy(mask_array),
            torch.from_numpy(normalized_static),
        )

    @torch.no_grad()
    def embeddings(
        self,
        values: np.ndarray,
        tau: np.ndarray,
        mask: np.ndarray,
        static: np.ndarray,
    ) -> np.ndarray:
        if self.mamba is None:
            raise RuntimeError(f"{self.model_name} has no Mamba embedding component")
        values_t, tau_t, mask_t, _ = self._normalized_tensors(
            values, tau, mask, static
        )
        z_multi, _ = self.mamba.encoder.extract_features(
            values_t.to(self.device, non_blocking=True),
            tau_t.to(self.device, non_blocking=True),
            mask_t.to(self.device, non_blocking=True),
        )
        return z_multi.float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _predict_deep(
        self,
        values: np.ndarray,
        tau: np.ndarray,
        mask: np.ndarray,
        static: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.deep is None:
            raise RuntimeError(f"{self.model_name} is not a deep checkpoint")
        values_t, tau_t, mask_t, static_t = self._normalized_tensors(
            values, tau, mask, static
        )
        count = len(values_t)
        batch = (
            values_t,
            tau_t,
            mask_t,
            static_t,
            torch.zeros(count, dtype=torch.long),
            torch.zeros(count, dtype=torch.float32),
        )
        inputs, _, _ = _adapt_deep_batch(self.model_name, batch, self.device)
        probability, pred_log_los = self.deep(*inputs)
        return (
            probability.float().cpu().numpy().astype(np.float32),
            torch.expm1(pred_log_los).clamp(min=0).cpu().numpy().astype(np.float32),
        )

    def explanation_components(
        self,
        values: np.ndarray,
        tau: np.ndarray,
        mask: np.ndarray,
        static: np.ndarray,
    ) -> dict[str, dict[str, Any]]:
        if self.model_name in {"XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC", "Flat-head"}:
            features, names = build_flat_design(
                self.model_name, values, mask, static
            )
            return {
                self.model_name: {
                    "features": features,
                    "feature_names": names,
                    "mortality_model": self.mortality_model,
                    "los_model": self.los_model,
                    "direct_feature_offset": 0,
                }
            }
        if self.model_name not in {"MABOOST", "Hybrid-plain"}:
            raise ValueError(f"SHAP is not registered for {self.model_name}")
        embeddings = self.embeddings(values, tau, mask, static)
        summary_static, clinical_names = build_flat_design(
            "XGBOOST-SUMMARY+STATIC", values, mask, static
        )
        hybrid = np.concatenate([embeddings, summary_static], axis=1)
        hybrid_names = [
            *(f"latent__{index:03d}" for index in range(embeddings.shape[1])),
            *clinical_names,
        ]
        components: dict[str, dict[str, Any]] = {
            "hybrid_component" if self.model_name == "MABOOST" else "Hybrid-plain": {
                "features": hybrid,
                "feature_names": hybrid_names,
                "mortality_model": self.hybrid_mortality,
                "los_model": self.hybrid_los,
                "direct_feature_offset": embeddings.shape[1],
            }
        }
        if self.model_name == "MABOOST":
            strict_flat, strict_names = build_flat_design(
                "XGBOOST-FLAT", values, mask, static
            )
            components["xgboost_flat_component"] = {
                "features": strict_flat,
                "feature_names": strict_names,
                "mortality_model": self.strict_flat_mortality,
                "los_model": self.strict_flat_los,
                "direct_feature_offset": 0,
            }
            flat, flat_names = build_flat_design("Flat-head", values, mask, static)
            components["flat_head_component"] = {
                "features": flat,
                "feature_names": flat_names,
                "mortality_model": self.flat_mortality,
                "los_model": self.flat_los,
                "direct_feature_offset": 0,
            }
        return components

    def predict(
        self,
        values: np.ndarray,
        tau: np.ndarray,
        mask: np.ndarray,
        static: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(values):
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        if self.model_name in {"XGBOOST-FLAT", "XGBOOST-SUMMARY+STATIC", "Flat-head"}:
            features, _ = build_flat_design(self.model_name, values, mask, static)
            if self.mortality_model is None or self.los_model is None:
                raise RuntimeError("XGBoost predictor was not loaded")
            probability = self.mortality_model.predict_proba(features)[:, 1]
            los_prediction = np.expm1(self.los_model.predict(features)).clip(min=0)
        elif self.model_name in {"MABOOST", "Hybrid-plain", "Mamba-z-only"}:
            embedding = self.embeddings(values, tau, mask, static)
            if self.model_name == "Mamba-z-only":
                hybrid_features = embedding
            else:
                summary_static, _ = build_flat_design(
                    "XGBOOST-SUMMARY+STATIC", values, mask, static
                )
                hybrid_features = np.concatenate([embedding, summary_static], axis=1)
            if self.hybrid_mortality is None or self.hybrid_los is None:
                raise RuntimeError("hybrid XGBoost heads were not loaded")
            hybrid_probability = self.hybrid_mortality.predict_proba(
                hybrid_features
            )[:, 1]
            hybrid_los_prediction = np.expm1(
                self.hybrid_los.predict(hybrid_features)
            ).clip(min=0)
            if self.model_name != "MABOOST":
                probability = hybrid_probability
                los_prediction = hybrid_los_prediction
            else:
                flat_features, _ = build_flat_design(
                    "Flat-head", values, mask, static
                )
                if self.flat_mortality is None or self.flat_los is None:
                    raise RuntimeError("MABOOST flat component was not loaded")
                flat_probability = self.flat_mortality.predict_proba(flat_features)[:, 1]
                flat_los_prediction = np.expm1(
                    self.flat_los.predict(flat_features)
                ).clip(min=0)
                strict_features, _ = build_flat_design(
                    "XGBOOST-FLAT", values, mask, static
                )
                if self.strict_flat_mortality is None or self.strict_flat_los is None:
                    raise RuntimeError("MABOOST strict flat component was not loaded")
                strict_probability = self.strict_flat_mortality.predict_proba(
                    strict_features
                )[:, 1]
                strict_los_prediction = np.expm1(
                    self.strict_flat_los.predict(strict_features)
                ).clip(min=0)
                fusion = self.choices["training_audit"]["fusion"]
                los_fusion = self.choices["training_audit"]["los_fusion"]
                probability = np.column_stack([
                    hybrid_probability,
                    strict_probability,
                    flat_probability,
                ]) @ np.asarray(fusion["weights"], dtype=np.float64)
                los_prediction = np.column_stack([
                    hybrid_los_prediction,
                    strict_los_prediction,
                    flat_los_prediction,
                ]) @ np.asarray(los_fusion["weights"], dtype=np.float64)
        else:
            probability, los_prediction = self._predict_deep(
                values, tau, mask, static
            )
        calibrated = self.calibrator.apply(np.asarray(probability, dtype=np.float32))
        return calibrated, np.asarray(los_prediction, dtype=np.float32)
