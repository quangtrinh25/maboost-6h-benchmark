from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

from v13_guard import WORKSPACE_ROOT, assert_safe_output_path, safe_write_json


DEFAULT_STATUS_ROOT = WORKSPACE_ROOT / "private" / "resource_guardian"
_ACTIVE_GUARDIAN: "ResourceGuardian | None" = None


class ResourceLimitExceeded(RuntimeError):
    pass


def _process_vram_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "torch_cuda_allocated_bytes": 0,
            "torch_cuda_reserved_bytes": 0,
            "torch_cuda_peak_allocated_bytes": 0,
        }
    return {
        "torch_cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _system_memory_snapshot() -> dict[str, float | int | str]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "source": "psutil",
            "ram_total_bytes": int(memory.total),
            "ram_available_bytes": int(memory.available),
            "ram_used_fraction": float(memory.percent / 100.0),
            **_process_vram_snapshot(),
        }
    except ImportError:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", maxsplit=1)
            fields[key] = int(value.strip().split()[0]) * 1024
        total = max(fields.get("MemTotal", 1), 1)
        available = fields.get("MemAvailable", fields.get("MemFree", 0))
        return {
            "source": "proc_meminfo",
            "ram_total_bytes": int(total),
            "ram_available_bytes": int(available),
            "ram_used_fraction": float(1.0 - available / total),
            **_process_vram_snapshot(),
        }


def release_process_memory() -> dict[str, int]:
    collected = int(gc.collect())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
    else:
        allocated = 0
        reserved = 0
    return {
        "gc_collected": collected,
        "torch_cuda_allocated_bytes": allocated,
        "torch_cuda_reserved_bytes": reserved,
    }


class ResourceGuardian:
    """Cooperative RAM cleanup with a last-resort, resumable hard stop.

    The monitor thread requests cleanup when system RAM crosses the soft limit.
    Training/evaluation code performs that cleanup at explicit safe points. At
    the critical limit the process exits immediately with a dedicated code so
    the parent launcher can remove only the matching stale claim and resume the
    incomplete job later.
    """

    def __init__(
        self,
        *,
        component: str,
        status_path: Path,
        config: dict[str, Any],
        sample_fn: Callable[[], dict[str, float | int | str]] = _system_memory_snapshot,
        hard_exit: bool = True,
    ) -> None:
        self.component = str(component)
        self.status_path = assert_safe_output_path(status_path)
        self.enabled = bool(config.get("enabled", True))
        self.poll_seconds = float(config.get("poll_seconds", 2.0))
        self.soft_fraction = float(config.get("ram_soft_fraction", 0.85))
        self.critical_fraction = float(config.get("ram_critical_fraction", 0.95))
        self.vram_fraction = float(config.get("vram_process_fraction", 0.95))
        self.exit_code = int(config.get("critical_exit_code", 86))
        self.soft_cooldown_seconds = float(config.get("soft_cooldown_seconds", 30.0))
        self.status_heartbeat_seconds = max(
            float(config.get("status_heartbeat_seconds", self.poll_seconds)), 0.5
        )
        if not 0.0 < self.soft_fraction < self.critical_fraction < 1.0:
            raise ValueError("guardian RAM thresholds must satisfy 0 < soft < critical < 1")
        if not 0.0 < self.vram_fraction <= 1.0:
            raise ValueError("guardian VRAM fraction must be in (0,1]")
        if self.poll_seconds <= 0.0:
            raise ValueError("guardian poll interval must be positive")
        self.sample_fn = sample_fn
        self.hard_exit = bool(hard_exit)
        self.stop_event = threading.Event()
        self.cleanup_requested = threading.Event()
        self.critical_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.events: list[dict[str, Any]] = []
        self.last_snapshot: dict[str, Any] = {}
        self.last_safe_point = "not_started"
        self.started_at = 0.0
        self._last_soft_event = -float("inf")
        self._last_status_write = -float("inf")
        self._write_lock = threading.Lock()

    def _payload(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "component": self.component,
            "pid": os.getpid(),
            "status": status,
            "enabled": self.enabled,
            "thresholds": {
                "poll_seconds": self.poll_seconds,
                "ram_soft_fraction": self.soft_fraction,
                "ram_critical_fraction": self.critical_fraction,
                "vram_process_fraction": self.vram_fraction,
                "critical_exit_code": self.exit_code,
            },
            "last_safe_point": self.last_safe_point,
            "last_snapshot": self.last_snapshot,
            "events": self.events[-32:],
            "wall_seconds": float(time.monotonic() - self.started_at)
            if self.started_at else 0.0,
            "patient_level_content": False,
        }

    def _write_status(self, status: str) -> None:
        with self._write_lock:
            safe_write_json(self.status_path, self._payload(status))
            self._last_status_write = time.monotonic()

    def start(self) -> "ResourceGuardian":
        global _ACTIVE_GUARDIAN
        if not self.enabled:
            return self
        if _ACTIVE_GUARDIAN is not None:
            raise RuntimeError("only one resource guardian may be active per process")
        self.started_at = time.monotonic()
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.vram_fraction, device=torch.cuda.current_device()
                )
            except (RuntimeError, ValueError) as error:
                self.events.append({
                    "event": "vram_cap_unavailable",
                    "error_type": type(error).__name__,
                })
        self.last_snapshot = dict(self.sample_fn())
        self.events.append({
            "event": "started",
            "monotonic_seconds": 0.0,
            **self.last_snapshot,
        })
        _ACTIVE_GUARDIAN = self
        self._write_status("running")
        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
            name=f"v13-resource-guardian-{self.component}",
        )
        self.thread.start()
        print(
            "[Guardian] active "
            f"(RAM cleanup@{self.soft_fraction:.0%}, "
            f"critical-stop@{self.critical_fraction:.0%}, "
            f"VRAM cap={self.vram_fraction:.0%})",
            flush=True,
        )
        return self

    def _monitor(self) -> None:
        while not self.stop_event.wait(self.poll_seconds):
            try:
                snapshot = dict(self.sample_fn())
                if self.evaluate_snapshot(snapshot) == "critical_stop":
                    return
            except BaseException as error:
                self.events.append({
                    "event": "monitor_error",
                    "error_type": type(error).__name__,
                    "monotonic_seconds": float(time.monotonic() - self.started_at),
                })

    def evaluate_snapshot(
        self, snapshot: dict[str, float | int | str]
    ) -> str:
        self.last_snapshot = dict(snapshot)
        used = float(snapshot["ram_used_fraction"])
        now = time.monotonic()
        if used >= self.critical_fraction:
            self.critical_event.set()
            self.events.append({
                "event": "critical_stop",
                "monotonic_seconds": float(now - self.started_at),
                **snapshot,
            })
            self._write_status("critical_stop")
            print(
                f"[Guardian] CRITICAL RAM {used:.1%}; "
                + (
                    f"exiting with resumable code {self.exit_code}"
                    if self.hard_exit
                    else "critical stop requested"
                ),
                flush=True,
            )
            release_process_memory()
            if self.hard_exit:
                os._exit(self.exit_code)
            return "critical_stop"
        if (
            used >= self.soft_fraction
            and now - self._last_soft_event >= self.soft_cooldown_seconds
        ):
            self._last_soft_event = now
            self.cleanup_requested.set()
            self.events.append({
                "event": "soft_cleanup_requested",
                "monotonic_seconds": float(now - self.started_at),
                **snapshot,
            })
            self._write_status("cleanup_requested")
            print(
                f"[Guardian] RAM {used:.1%}; cleanup requested at next safe point",
                flush=True,
            )
            return "cleanup_requested"
        return "ok"

    def safe_point(self, label: str, *, force_cleanup: bool = False) -> bool:
        self.last_safe_point = str(label)
        if self.critical_event.is_set():
            raise ResourceLimitExceeded(
                f"resource guardian critical limit at {self.last_safe_point}"
            )
        should_clean = force_cleanup or self.cleanup_requested.is_set()
        if not should_clean:
            if time.monotonic() - self._last_status_write >= self.status_heartbeat_seconds:
                self._write_status("running")
            return False
        self.cleanup_requested.clear()
        cleanup = release_process_memory()
        self.events.append({
            "event": "cleanup_completed",
            "safe_point": self.last_safe_point,
            "forced": bool(force_cleanup),
            "monotonic_seconds": float(time.monotonic() - self.started_at),
            **cleanup,
        })
        self._write_status("running")
        return True

    def stop(self, status: str = "complete") -> None:
        global _ACTIVE_GUARDIAN
        if not self.enabled:
            return
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=max(self.poll_seconds * 2.0, 1.0))
        self.last_safe_point = "guardian_stop"
        cleanup = release_process_memory()
        self.events.append({
            "event": "cleanup_completed",
            "safe_point": self.last_safe_point,
            "forced": True,
            "monotonic_seconds": float(time.monotonic() - self.started_at),
            **cleanup,
        })
        self.events.append({
            "event": "stopped",
            "final_status": str(status),
            "monotonic_seconds": float(time.monotonic() - self.started_at),
        })
        self._write_status(str(status))
        if _ACTIVE_GUARDIAN is self:
            _ACTIVE_GUARDIAN = None


def resource_safe_point(label: str, *, force_cleanup: bool = False) -> bool:
    if _ACTIVE_GUARDIAN is None:
        if force_cleanup:
            release_process_memory()
            return True
        return False
    return _ACTIVE_GUARDIAN.safe_point(label, force_cleanup=force_cleanup)


def load_resource_guardian_config() -> dict[str, Any]:
    path = WORKSPACE_ROOT / "config" / "phase07_training_v13.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("resource_guardian", {}))


def guarded_entrypoint(
    main_fn: Callable[[], int],
    *,
    component: str,
    status_path: Path | None = None,
) -> int:
    """Run one heavy non-training stage under the locked resource guardian."""
    if "--list-only" in sys.argv:
        return int(main_fn())
    guardian = ResourceGuardian(
        component=component,
        status_path=(
            status_path
            if status_path is not None
            else DEFAULT_STATUS_ROOT / f"{component}.json"
        ),
        config=load_resource_guardian_config(),
    ).start()
    try:
        result = int(main_fn())
    except BaseException:
        guardian.stop("failed")
        raise
    guardian.stop("complete")
    return result
