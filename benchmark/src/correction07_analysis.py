from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from correction07_bootstrap import BootstrapInput, RustBootstrapKernel, _metric, sha256_array
from correction07_checkpoints import (
    BootstrapShardStore,
    BootstrapTaskIdentity,
    run_bootstrap_task,
)


def _input_sha256(inputs: BootstrapInput) -> str:
    digest = hashlib.sha256()
    for array in (
        inputs.labels,
        inputs.predictions_a,
        inputs.predictions_b,
        inputs.cluster_codes,
    ):
        digest.update(sha256_array(array).encode("ascii"))
    return digest.hexdigest()


def corrected_clustered_paired_bootstrap(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    groups: np.ndarray,
    *,
    metric_name: str,
    replicates: int,
    seed: int,
    kernel: RustBootstrapKernel,
    threads: int,
    thread_limit: int = 16,
    shard_size: int,
    evaluation_batch_size: int | None = None,
    max_draw_buffer_bytes: int | None = None,
    checkpoint_root: Path | None = None,
    source: str = "synthetic",
    dataset: str = "synthetic",
    scope: str = "bounded",
    contrast: str = "a_vs_b",
    correction_sha256: str = "0" * 64,
    progress_callback=None,
    resource_safe_point=None,
) -> dict[str, float | int]:
    labels = np.asarray(labels)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    groups = np.asarray(groups)
    if not (
        len(labels) == len(predictions_a) == len(predictions_b) == len(groups)
    ):
        raise ValueError("clustered bootstrap inputs are not aligned")
    # Preserve the ordering semantics of the supplied group dtype.  In
    # particular, PreparedEnsemble supplies contiguous numeric codes whose
    # numeric order already represents the legacy lexical patient-key order.
    # Converting those codes to strings would reorder 10 before 2 and change
    # the RNG draw-to-patient mapping.
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    inputs = BootstrapInput.from_arrays(
        metric=metric_name,
        labels=labels,
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        cluster_codes=inverse,
    )
    task = BootstrapTaskIdentity(
        source=source,
        dataset=dataset,
        scope=scope,
        contrast=contrast,
        metric=inputs.metric,
        seed=int(seed),
        replicates=int(replicates),
        correction_sha256=correction_sha256,
        rust_binary_sha256=kernel.manifest["library_sha256"],
        input_sha256=_input_sha256(inputs),
        thread_limit=int(thread_limit),
    )
    store = BootstrapShardStore(checkpoint_root) if checkpoint_root is not None else None
    result = run_bootstrap_task(
        task,
        inputs,
        kernel,
        shard_size=shard_size,
        threads=threads,
        store=store,
        evaluation_batch_size=evaluation_batch_size,
        max_draw_buffer_bytes=max_draw_buffer_bytes,
        progress_callback=progress_callback,
        resource_safe_point=resource_safe_point,
    )
    if not result.complete:
        raise RuntimeError("corrected bootstrap task is incomplete")
    observed = _metric(inputs.metric, inputs.labels, inputs.predictions_a) - _metric(
        inputs.metric, inputs.labels, inputs.predictions_b
    )
    return {
        "difference": float(observed),
        "ci_lower": float(np.quantile(result.differences, 0.025)),
        "ci_upper": float(np.quantile(result.differences, 0.975)),
        "replicates": int(replicates),
        "clusters": int(len(unique_groups)),
    }


def corrected_paired_sign_permutation(
    paired_differences: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> dict[str, float | int]:
    differences = np.asarray(paired_differences, dtype=np.float64)
    groups = np.asarray(groups)
    if len(differences) != len(groups):
        raise ValueError("paired permutation inputs are not aligned")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    # np.unique preserves the legacy sorted cluster order while bincount avoids
    # one full boolean scan of all rows for every cluster.  This changes the
    # preprocessing cost from O(rows * clusters) to O(rows + clusters).
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    cluster_sums = np.bincount(
        inverse, weights=differences, minlength=len(unique_groups)
    )
    cluster_counts = np.bincount(inverse, minlength=len(unique_groups))
    cluster_differences = np.ascontiguousarray(
        cluster_sums / cluster_counts, dtype=np.float64
    )
    observed = float(cluster_differences.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    signs_values = np.asarray([-1.0, 1.0])
    while completed < replicates:
        count = min(batch_size, replicates - completed)
        signs = rng.choice(
            signs_values, size=(count, len(cluster_differences))
        )
        permuted = np.mean(signs * cluster_differences[None, :], axis=1)
        exceedances += int(np.sum(np.abs(permuted) >= abs(observed)))
        completed += count
    return {
        "mean_cluster_difference": observed,
        "p_value_two_sided": float(
            (1.0 + float(exceedances)) / (replicates + 1.0)
        ),
        "replicates": int(replicates),
        "clusters": int(len(unique_groups)),
    }
