from __future__ import annotations

import copy
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from correction07_bootstrap import BootstrapInput, MetricName, generate_draw_batch, sha256_array
from correction07_contract import RUST_ABI, RUST_ABI_VERSION, canonical_json_sha256
from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file


@dataclass(frozen=True)
class BootstrapTaskIdentity:
    source: str
    dataset: str
    scope: str
    contrast: str
    metric: MetricName
    seed: int
    replicates: int
    correction_sha256: str
    rust_binary_sha256: str
    input_sha256: str
    rust_abi: str = RUST_ABI
    rust_abi_version: int = RUST_ABI_VERSION
    thread_limit: int = 16

    @property
    def task_sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


@dataclass(frozen=True)
class VerifiedShard:
    start: int
    stop: int
    differences: np.ndarray
    rng_state_before: dict[str, Any]
    rng_state_after: dict[str, Any]
    manifest_path: Path


@dataclass(frozen=True)
class BootstrapTaskResult:
    differences: np.ndarray
    complete: bool
    completed_replicates: int
    progress: tuple[dict[str, Any], ...]


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _write_sidecar(path: Path) -> None:
    safe_write_text(_sidecar(path), f"{sha256_file(path)}  {_relative(path)}\n")


def _verify_sidecar(path: Path) -> str:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"checkpoint shard or sidecar missing: {path}")
    fields = sidecar.read_text(encoding="utf-8").split()
    observed = sha256_file(path)
    if len(fields) < 2 or fields[0] != observed or fields[1] != _relative(path):
        raise RuntimeError(f"checkpoint shard SHA mismatch: {path}")
    return observed


class BootstrapShardStore:
    def __init__(self, root: Path):
        private_root = (WORKSPACE_ROOT / "private").resolve()
        candidate = assert_safe_output_path(Path(root)).resolve()
        try:
            candidate.relative_to(private_root)
        except ValueError as exc:
            raise PermissionError("bootstrap checkpoints must remain under private/") from exc
        self.root = candidate

    @property
    def identity_path(self) -> Path:
        return self.root / "TASK_IDENTITY.json"

    def _ensure_identity(self, task: BootstrapTaskIdentity) -> None:
        payload = {**asdict(task), "task_sha256": task.task_sha256}
        if self.identity_path.exists():
            _verify_sidecar(self.identity_path)
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if observed != payload:
                raise RuntimeError("bootstrap task identity drift")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        safe_write_json(self.identity_path, payload)
        _write_sidecar(self.identity_path)

    def discover(
        self,
        task: BootstrapTaskIdentity,
        *,
        expected_rows: int | None = None,
        expected_clusters: int | None = None,
    ) -> list[VerifiedShard]:
        self._ensure_identity(task)
        manifests = sorted(self.root.glob("shard_*.json"))
        verified: list[VerifiedShard] = []
        expected_start = 0
        expected_rng = np.random.default_rng(task.seed)
        for manifest_path in manifests:
            _verify_sidecar(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("task_sha256") != task.task_sha256:
                raise RuntimeError(f"checkpoint shard task drift: {manifest_path}")
            start = int(manifest["start"])
            stop = int(manifest["stop"])
            if start != expected_start or stop <= start or stop > task.replicates:
                raise RuntimeError(f"checkpoint shard gap/overlap: {manifest_path}")
            rows = int(manifest.get("rows", -1))
            clusters = int(manifest.get("clusters", -1))
            threads = int(manifest.get("threads", -1))
            if expected_rows is not None and rows != expected_rows:
                raise RuntimeError(f"checkpoint shard row-count drift: {manifest_path}")
            if expected_clusters is not None and clusters != expected_clusters:
                raise RuntimeError(f"checkpoint shard cluster-count drift: {manifest_path}")
            if rows <= 0 or clusters <= 0 or not 1 <= threads <= task.thread_limit:
                raise RuntimeError(f"checkpoint shard execution metadata drift: {manifest_path}")
            expected_before = copy.deepcopy(expected_rng.bit_generator.state)
            if canonical_json_sha256(manifest["rng_state_before"]) != canonical_json_sha256(expected_before):
                raise RuntimeError(f"checkpoint shard RNG chain drift: {manifest_path}")
            draws = generate_draw_batch(expected_rng, stop - start, clusters)
            if sha256_array(draws) != manifest.get("draw_sha256"):
                raise RuntimeError(f"checkpoint shard draw hash drift: {manifest_path}")
            expected_after = copy.deepcopy(expected_rng.bit_generator.state)
            if canonical_json_sha256(manifest["rng_state_after"]) != canonical_json_sha256(expected_after):
                raise RuntimeError(f"checkpoint shard RNG end-state drift: {manifest_path}")
            data_path = REPO_ROOT / manifest["data_file"]
            observed_data_hash = _verify_sidecar(data_path)
            if observed_data_hash != manifest["data_file_sha256"]:
                raise RuntimeError(f"checkpoint shard data hash drift: {data_path}")
            try:
                with np.load(data_path, allow_pickle=False) as archive:
                    differences = np.ascontiguousarray(
                        archive["differences"], dtype=np.float64
                    )
            except Exception as exc:
                raise RuntimeError(f"checkpoint shard cannot be loaded: {data_path}") from exc
            if differences.ndim != 1 or len(differences) != stop - start:
                raise RuntimeError(f"checkpoint shard length drift: {data_path}")
            if sha256_array(differences) != manifest["difference_sha256"]:
                raise RuntimeError(f"checkpoint shard difference hash drift: {data_path}")
            verified.append(
                VerifiedShard(
                    start=start,
                    stop=stop,
                    differences=differences,
                    rng_state_before=manifest["rng_state_before"],
                    rng_state_after=manifest["rng_state_after"],
                    manifest_path=manifest_path,
                )
            )
            expected_start = stop
        return verified

    def commit(
        self,
        task: BootstrapTaskIdentity,
        *,
        start: int,
        stop: int,
        differences: np.ndarray,
        rng_state_before: dict[str, Any],
        rng_state_after: dict[str, Any],
        draw_sha256: str,
        elapsed_seconds: float,
        threads: int,
        rows: int,
        clusters: int,
        verified_start: int | None = None,
    ) -> Path:
        if verified_start is None:
            existing = self.discover(
                task, expected_rows=rows, expected_clusters=clusters
            )
            expected_start = existing[-1].stop if existing else 0
        else:
            expected_start = int(verified_start)
        if start != expected_start or stop <= start:
            raise RuntimeError("checkpoint shard commit range is not contiguous")
        values = np.ascontiguousarray(differences, dtype=np.float64)
        if len(values) != stop - start or not np.isfinite(values).all():
            raise RuntimeError("checkpoint shard differences are invalid")
        stem = f"shard_{start:06d}_{stop:06d}"
        data_path = self.root / f"{stem}.npz"
        manifest_path = self.root / f"{stem}.json"
        if any(path.exists() for path in (data_path, _sidecar(data_path), manifest_path, _sidecar(manifest_path))):
            raise FileExistsError(f"immutable checkpoint shard already exists: {stem}")
        temporary = self.root / f".{stem}.{os.getpid()}.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, differences=values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, data_path)
        _write_sidecar(data_path)
        manifest = {
            "schema_version": "1.0.0",
            "task_sha256": task.task_sha256,
            "start": start,
            "stop": stop,
            "rng_state_before": rng_state_before,
            "rng_state_after": rng_state_after,
            "draw_sha256": draw_sha256,
            "difference_sha256": sha256_array(values),
            "data_file": _relative(data_path),
            "data_file_sha256": sha256_file(data_path),
            "elapsed_seconds": float(elapsed_seconds),
            "threads": int(threads),
            "rows": int(rows),
            "clusters": int(clusters),
            "patient_level_content": True,
            "public_release_permitted": False,
        }
        safe_write_json(manifest_path, manifest)
        _write_sidecar(manifest_path)
        return manifest_path


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_bootstrap_task(
    task: BootstrapTaskIdentity,
    inputs: BootstrapInput,
    kernel: Any,
    *,
    shard_size: int,
    threads: int,
    store: BootstrapShardStore | None = None,
    stop_after_shards: int | None = None,
    evaluation_batch_size: int | None = None,
    max_draw_buffer_bytes: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    resource_safe_point: Callable[[str], None] | None = None,
) -> BootstrapTaskResult:
    if task.metric != inputs.metric:
        raise RuntimeError("bootstrap task/input metric drift")
    if task.rust_abi != RUST_ABI or task.rust_abi_version != RUST_ABI_VERSION:
        raise RuntimeError("bootstrap task Rust ABI drift")
    if task.replicates <= 0 or shard_size <= 0 or threads <= 0:
        raise ValueError("replicates, shard_size, and threads must be positive")
    if threads > task.thread_limit:
        raise RuntimeError("bootstrap thread count exceeds task identity limit")
    shards = (
        store.discover(
            task, expected_rows=len(inputs.labels), expected_clusters=inputs.cluster_count
        )
        if store is not None
        else []
    )
    completed_values = [shard.differences for shard in shards]
    start = shards[-1].stop if shards else 0
    rng = np.random.default_rng(task.seed)
    if shards:
        rng.bit_generator.state = copy.deepcopy(shards[-1].rng_state_after)
    created = 0
    progress: list[dict[str, Any]] = []
    task_started = time.monotonic()
    batch_size = shard_size if evaluation_batch_size is None else int(evaluation_batch_size)
    if batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    batch_shards = max(1, batch_size // shard_size)
    effective_batch_size = batch_shards * shard_size
    if max_draw_buffer_bytes is not None:
        # Generation plus the contiguous FFI buffer can coexist briefly.
        requested_bytes = (
            2
            * effective_batch_size
            * inputs.cluster_count
            * np.dtype(np.uint32).itemsize
        )
        if requested_bytes > int(max_draw_buffer_bytes):
            raise RuntimeError("bootstrap draw buffer exceeds the registered memory budget")

    while start < task.replicates:
        if stop_after_shards is not None and created >= stop_after_shards:
            break
        if resource_safe_point is not None:
            resource_safe_point(f"bootstrap_{task.task_sha256[:12]}_{start}")
        allowed_shards = (
            None if stop_after_shards is None else stop_after_shards - created
        )
        batch_stop_limit = min(task.replicates, start + effective_batch_size)
        shard_specs: list[tuple[int, int, dict[str, Any], dict[str, Any], np.ndarray]] = []
        shard_start = start
        while shard_start < batch_stop_limit and (
            allowed_shards is None or len(shard_specs) < allowed_shards
        ):
            shard_stop = min(task.replicates, shard_start + shard_size)
            rng_before = copy.deepcopy(rng.bit_generator.state)
            shard_draws = generate_draw_batch(
                rng, shard_stop - shard_start, inputs.cluster_count
            )
            rng_after = copy.deepcopy(rng.bit_generator.state)
            shard_specs.append(
                (shard_start, shard_stop, rng_before, rng_after, shard_draws)
            )
            shard_start = shard_stop
        if not shard_specs:
            break
        draw_started = time.monotonic()
        batch_draws = np.ascontiguousarray(
            np.concatenate([spec[4] for spec in shard_specs]), dtype=np.uint32
        )
        batch_differences = np.ascontiguousarray(
            kernel.evaluate_draws(inputs, batch_draws, threads=threads),
            dtype=np.float64,
        )
        elapsed = time.monotonic() - draw_started
        difference_offset = 0
        for shard_start, shard_stop, rng_before, rng_after, shard_draws in shard_specs:
            count = shard_stop - shard_start
            differences = np.ascontiguousarray(
                batch_differences[difference_offset:difference_offset + count],
                dtype=np.float64,
            )
            if store is not None:
                store.commit(
                    task,
                    start=shard_start,
                    stop=shard_stop,
                    differences=differences,
                    rng_state_before=rng_before,
                    rng_state_after=rng_after,
                    draw_sha256=sha256_array(shard_draws),
                    elapsed_seconds=elapsed / len(shard_specs),
                    threads=threads,
                    rows=len(inputs.labels),
                    clusters=inputs.cluster_count,
                    verified_start=shard_start,
                )
            completed_values.append(differences)
            difference_offset += count
        start = shard_specs[-1][1]
        created += len(shard_specs)
        total_elapsed = max(time.monotonic() - task_started, 1e-12)
        session_processed = sum(len(values) for values in completed_values[len(shards):])
        processed = sum(len(values) for values in completed_values)
        throughput = session_processed / total_elapsed
        record = {
            "task_sha256": task.task_sha256,
            "completed_replicates": processed,
            "total_replicates": task.replicates,
            "replicates_per_second": throughput,
            "peak_rss_bytes": _peak_rss_bytes(),
            "eta_seconds": max(0.0, (task.replicates - processed) / throughput),
        }
        progress.append(record)
        if progress_callback is not None:
            progress_callback(record)

    combined = (
        np.ascontiguousarray(np.concatenate(completed_values), dtype=np.float64)
        if completed_values
        else np.empty(0, dtype=np.float64)
    )
    return BootstrapTaskResult(
        differences=combined,
        complete=len(combined) == task.replicates,
        completed_replicates=len(combined),
        progress=tuple(progress),
    )
