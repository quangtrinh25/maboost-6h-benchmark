from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from correction07_contract import (
    METRIC_CODES,
    RUST_ABI,
    RUST_ABI_VERSION,
    canonical_json_sha256,
)
from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT
from v13_guard import sha256_file


MetricName = Literal["auroc", "auprc", "brier", "los_mae", "los_rmse"]


@dataclass(frozen=True)
class BootstrapInput:
    metric: MetricName
    labels: np.ndarray
    predictions_a: np.ndarray
    predictions_b: np.ndarray
    cluster_codes: np.ndarray
    cluster_count: int

    @classmethod
    def from_arrays(
        cls,
        metric: MetricName,
        labels: np.ndarray,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        cluster_codes: np.ndarray,
    ) -> "BootstrapInput":
        if metric not in METRIC_CODES:
            raise ValueError(f"unsupported metric: {metric}")
        raw_labels = np.asarray(labels, dtype=np.float64)
        a = np.asarray(predictions_a, dtype=np.float64)
        b = np.asarray(predictions_b, dtype=np.float64)
        raw_codes = np.asarray(cluster_codes)
        if any(array.ndim != 1 for array in (raw_labels, a, b, raw_codes)):
            raise ValueError("bootstrap arrays must be one-dimensional")
        if len({len(raw_labels), len(a), len(b), len(raw_codes)}) != 1:
            raise ValueError("bootstrap arrays are not aligned")
        if len(raw_labels) == 0:
            raise ValueError("bootstrap arrays must not be empty")
        if not np.isfinite(raw_labels).all():
            raise ValueError("non-finite label")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite prediction")
        if metric in ("auroc", "auprc", "brier") and not np.isin(
            raw_labels, np.asarray([0.0, 1.0])
        ).all():
            raise ValueError("mortality labels must be binary")
        if not np.issubdtype(raw_codes.dtype, np.integer) or (raw_codes < 0).any():
            raise ValueError("cluster codes must be nonnegative integers")
        unique = np.unique(raw_codes)
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError("cluster codes must be contiguous and zero-based")
        return cls(
            metric=metric,
            labels=np.ascontiguousarray(raw_labels),
            predictions_a=np.ascontiguousarray(a),
            predictions_b=np.ascontiguousarray(b),
            cluster_codes=np.ascontiguousarray(raw_codes, dtype=np.uint32),
            cluster_count=int(len(unique)),
        )


def generate_draw_batch(
    rng: np.random.Generator, replicates: int, cluster_count: int
) -> np.ndarray:
    if replicates <= 0 or cluster_count <= 0:
        raise ValueError("replicates and cluster_count must be positive")
    draws = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    return np.ascontiguousarray(draws, dtype=np.uint32)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _metric(name: MetricName, labels: np.ndarray, predictions: np.ndarray) -> float:
    if name == "auroc":
        return float(roc_auc_score(labels, predictions))
    if name == "auprc":
        return float(average_precision_score(labels, predictions))
    if name == "brier":
        return float(brier_score_loss(labels, predictions))
    if name == "los_mae":
        return float(mean_absolute_error(labels, predictions))
    if name == "los_rmse":
        return float(math.sqrt(mean_squared_error(labels, predictions)))
    raise ValueError(f"unsupported metric: {name}")


def legacy_differences_for_draws(
    inputs: BootstrapInput, draws: np.ndarray
) -> np.ndarray:
    supplied = np.asarray(draws)
    if supplied.ndim != 2 or supplied.shape[1] != inputs.cluster_count:
        raise ValueError("bootstrap draw shape mismatch")
    if not np.issubdtype(supplied.dtype, np.integer):
        raise ValueError("bootstrap draws must be integer cluster indices")
    if (supplied < 0).any() or (supplied >= inputs.cluster_count).any():
        raise ValueError("bootstrap draw contains invalid cluster index")
    cluster_rows = [
        np.flatnonzero(inputs.cluster_codes == index)
        for index in range(inputs.cluster_count)
    ]
    differences = np.empty(len(supplied), dtype=np.float64)
    for index, draw in enumerate(supplied):
        rows = np.concatenate([cluster_rows[int(cluster)] for cluster in draw])
        differences[index] = _metric(
            inputs.metric, inputs.labels[rows], inputs.predictions_a[rows]
        ) - _metric(inputs.metric, inputs.labels[rows], inputs.predictions_b[rows])
    return differences


def _verify_build_provenance(manifest: dict) -> None:
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("Rust analysis source lock is missing")
    for record in sources:
        path = REPO_ROOT / record["file"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"Rust analysis source drift: {record['file']}")
    if canonical_json_sha256(sources) != manifest.get("source_tree_sha256"):
        raise RuntimeError("Rust analysis source-tree hash drift")

    toolchain = manifest.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or canonical_json_sha256(toolchain) != manifest.get("toolchain_sha256")
    ):
        raise RuntimeError("Rust analysis toolchain manifest drift")
    environment = dict(os.environ)
    commands = {"cargo": ["cargo", "-V"], "rustc": ["rustc", "-Vv"]}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            env=environment,
            cwd=WORKSPACE_ROOT / "rust_analysis_kernel",
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip() != toolchain[name]:
            raise RuntimeError(f"Rust analysis {name} toolchain drift")


@dataclass
class RustBootstrapKernel:
    library: ctypes.CDLL
    manifest: dict
    manifest_path: Path

    @classmethod
    def load(cls, manifest_path: Path) -> "RustBootstrapKernel":
        manifest_path = Path(manifest_path)
        try:
            sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
            fields = sidecar.read_text(encoding="utf-8").split()
            manifest_hash = sha256_file(manifest_path)
            relative_manifest = str(manifest_path.resolve().relative_to(REPO_ROOT))
            if len(fields) < 2 or fields[0] != manifest_hash or fields[1] != relative_manifest:
                raise RuntimeError("build manifest SHA mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "built_hash_locked_for_qualification":
                raise RuntimeError("build manifest is not locked for qualification")
            if manifest.get("abi") != RUST_ABI or manifest.get("abi_version") != RUST_ABI_VERSION:
                raise RuntimeError("Rust analysis ABI drift")
            for field in ("source_tree_sha256", "toolchain_sha256", "library_sha256"):
                value = manifest.get(field)
                if not isinstance(value, str) or len(value) != 64:
                    raise RuntimeError(f"missing build binding: {field}")
            if manifest.get("dependencies") != []:
                raise RuntimeError("Rust analysis dependency lock drift")
            _verify_build_provenance(manifest)
            library_path = REPO_ROOT / manifest["library"]
            if sha256_file(library_path) != manifest["library_sha256"]:
                raise RuntimeError("Rust analysis library hash drift")
            library = ctypes.CDLL(str(library_path))
            version = library.maboost_analysis_abi_version_v1
            version.argtypes = []
            version.restype = ctypes.c_uint32
            if version() != RUST_ABI_VERSION:
                raise RuntimeError("Rust analysis library ABI mismatch")
            evaluate = library.maboost_bootstrap_evaluate_v1
            evaluate.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            evaluate.restype = ctypes.c_int32
            return cls(library=library, manifest=manifest, manifest_path=manifest_path)
        except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"qualified Rust analysis kernel unavailable: {manifest_path}"
            ) from exc

    def evaluate_draws(
        self, inputs: BootstrapInput, draws: np.ndarray, *, threads: int
    ) -> np.ndarray:
        supplied = np.asarray(draws)
        if supplied.ndim != 2 or supplied.shape[1] != inputs.cluster_count:
            raise ValueError("bootstrap draw shape mismatch")
        if not np.issubdtype(supplied.dtype, np.integer):
            raise ValueError("bootstrap draws must be integer cluster indices")
        if (supplied < 0).any() or (supplied >= inputs.cluster_count).any():
            raise ValueError("bootstrap draw contains invalid cluster index")
        if threads <= 0:
            raise ValueError("Rust thread count must be positive")
        contiguous = np.ascontiguousarray(supplied, dtype=np.uint32)
        output = np.empty(len(contiguous), dtype=np.float64)
        error_replicate = ctypes.c_size_t(np.iinfo(np.uintp).max)
        evaluate = self.library.maboost_bootstrap_evaluate_v1
        code = int(
            evaluate(
                METRIC_CODES[inputs.metric],
                inputs.labels.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                inputs.predictions_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                inputs.predictions_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                inputs.cluster_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                len(inputs.labels),
                inputs.cluster_count,
                contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                len(contiguous),
                threads,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(error_replicate),
            )
        )
        if code != 0:
            messages = {
                1: "invalid metric",
                2: "length mismatch",
                3: "non-finite value",
                4: "invalid label",
                5: "invalid cluster code",
                6: "invalid bootstrap draw",
                7: f"missing class at replicate {error_replicate.value}",
                8: "null pointer",
                9: "Rust panic",
            }
            raise RuntimeError(
                f"Rust analysis kernel error {code}: {messages.get(code, 'unknown error')}"
            )
        if not np.isfinite(output).all():
            raise RuntimeError("Rust analysis kernel returned non-finite output")
        return output
