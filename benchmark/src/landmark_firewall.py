from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from landmark_protocol import REPO_ROOT, load_protocol
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file


FEATURE_NAMES = ("values", "tau", "mask", "elapsed", "static")
LABEL_NAMES = ("mortality", "los", "stay_keys", "group_keys", "age", "sex_male", "icu_type_code")


def dataset_root(dataset: str, config: dict[str, Any]) -> Path:
    if dataset not in config["datasets"]:
        raise ValueError(f"unknown landmark dataset: {dataset}")
    return REPO_ROOT / config["datasets"][dataset]["private_root"]


def _load_arrays(root: Path, names: tuple[str, ...], *, mmap: bool) -> dict[str, np.ndarray]:
    mode = "r" if mmap else None
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode=mode, allow_pickle=False)
        for name in names
    }
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"unaligned private arrays under {root}")
    return arrays


def _verify_manifest_section(dataset: str, section: str) -> Path:
    _, _, config, _ = load_protocol()
    root = dataset_root(dataset, config)
    manifest_path = root / "manifest_private.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset:
        raise RuntimeError(f"private ETL manifest dataset mismatch: {dataset}")
    key = "feature_artifacts" if section == "features" else "sealed_label_artifacts"
    expected_names = FEATURE_NAMES if section == "features" else LABEL_NAMES
    records = {Path(item["file"]).stem: item for item in manifest[key]}
    if set(records) != set(expected_names):
        raise RuntimeError(f"private ETL manifest section drift: {dataset}/{section}")
    for name, record in records.items():
        path = root / section / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"private ETL integrity failure: {path}")
    return root


def load_features(dataset: str, *, mmap: bool = True) -> dict[str, np.ndarray]:
    """Load a label- and identifier-free prediction view."""
    _, _, config, _ = load_protocol()
    root = _verify_manifest_section(dataset, "features") / "features"
    arrays = _load_arrays(root, FEATURE_NAMES, mmap=mmap)
    forbidden = set(LABEL_NAMES).intersection(arrays)
    if forbidden:
        raise PermissionError(f"target feature view exposed sealed fields: {sorted(forbidden)}")
    return arrays


def load_source_training_data(dataset: str, *, mmap: bool = True) -> dict[str, np.ndarray]:
    if dataset not in {"mimic_iv", "eicu"}:
        raise PermissionError("CareVue is target-only and cannot be opened for training")
    _, _, config, _ = load_protocol()
    root = _verify_manifest_section(dataset, "features")
    _verify_manifest_section(dataset, "sealed_labels")
    return {
        **_load_arrays(root / "features", FEATURE_NAMES, mmap=mmap),
        **_load_arrays(root / "sealed_labels", LABEL_NAMES, mmap=mmap),
    }


def lock_source_choices(path: Path, payload: dict[str, Any], *, source: str, plan_hash: str, config_hash: str) -> str:
    if source not in {"mimic_iv", "eicu"}:
        raise PermissionError("source choices can only be locked for a development source")
    target = assert_safe_output_path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("choices") != payload:
            raise RuntimeError("immutable source-choice payload mismatch")
        hash_path = target.with_suffix(".json.sha256")
        if not hash_path.is_file() or sha256_file(target) != hash_path.read_text(
            encoding="utf-8"
        ).split()[0]:
            raise RuntimeError("source-choice lock integrity failure")
        return sha256_file(target)
    safe_write_json(target, {
        "schema_version": "1.0.0",
        "source_dataset": source,
        "target_information_used": False,
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "choices": payload,
    })
    digest = sha256_file(target)
    safe_write_text(target.with_suffix(".json.sha256"), f"{digest}  {target.name}\n")
    return digest


def lock_target_predictions(path: Path, *, dataset: str, source_choices_path: Path) -> dict[str, str]:
    if dataset not in {"mimic_iv", "eicu", "mimic_iii_carevue"}:
        raise ValueError(f"unknown target dataset: {dataset}")
    target = assert_safe_output_path(path)
    if not target.is_file():
        raise FileNotFoundError("target predictions must exist before label access")
    choices_hash_path = source_choices_path.with_suffix(".json.sha256")
    if not source_choices_path.is_file() or not choices_hash_path.is_file():
        raise PermissionError("source choices must be hash-locked before target predictions")
    choices_hash = choices_hash_path.read_text(encoding="utf-8").split()[0]
    if sha256_file(source_choices_path) != choices_hash:
        raise RuntimeError("source-choice lock integrity failure")
    prediction_hash = sha256_file(target)
    safe_write_text(
        target.with_suffix(target.suffix + ".sha256"),
        f"{prediction_hash}  {target.name}\n",
    )
    lock_path = target.with_suffix(target.suffix + ".lock.json")
    safe_write_json(lock_path, {
        "dataset": dataset,
        "source_choices_sha256": choices_hash,
        "target_predictions_sha256": prediction_hash,
        "labels_opened": False,
    })
    safe_write_text(
        lock_path.with_suffix(".json.sha256"),
        f"{sha256_file(lock_path)}  {lock_path.name}\n",
    )
    return {
        "source_choices_sha256": choices_hash,
        "target_predictions_sha256": prediction_hash,
        "lock_sha256": sha256_file(lock_path),
    }


def open_target_labels_for_locked_evaluation(dataset: str, *, prediction_path: Path, source_choices_path: Path, mmap: bool = True) -> dict[str, np.ndarray]:
    lock_path = prediction_path.with_suffix(prediction_path.suffix + ".lock.json")
    prediction_hash_path = prediction_path.with_suffix(prediction_path.suffix + ".sha256")
    choices_hash_path = source_choices_path.with_suffix(".json.sha256")
    for required in (lock_path, prediction_hash_path, choices_hash_path):
        if not required.is_file():
            raise PermissionError(f"target-label firewall lock missing: {required}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("dataset") != dataset:
        raise PermissionError("target prediction lock is bound to another dataset")
    if sha256_file(prediction_path) != prediction_hash_path.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("target prediction changed after locking")
    if sha256_file(source_choices_path) != choices_hash_path.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("source choices changed after locking")
    _, _, config, _ = load_protocol()
    _verify_manifest_section(dataset, "sealed_labels")
    labels = _load_arrays(
        dataset_root(dataset, config) / "sealed_labels",
        LABEL_NAMES,
        mmap=mmap,
    )
    if len(labels["mortality"]) == 0:
        raise RuntimeError("sealed target label set is empty")
    return labels
