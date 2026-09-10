from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from correction07_contract import (
    RUST_ABI,
    RUST_ABI_VERSION,
    canonical_json_sha256,
)
from landmark_protocol import REPO_ROOT, WORKSPACE_ROOT
from v13_guard import assert_safe_output_path, safe_write_json, safe_write_text, sha256_file


CRATE_ROOT = WORKSPACE_ROOT / "rust_analysis_kernel"
CARGO = Path(shutil.which("cargo") or "cargo")
RUSTC = Path(shutil.which("rustc") or "rustc")
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "private" / "correction07_rust_build"
SOURCE_FILES = (
    CRATE_ROOT / "rust-toolchain.toml",
    CRATE_ROOT / "Cargo.toml",
    CRATE_ROOT / "Cargo.lock",
    CRATE_ROOT / "src" / "lib.rs",
    CRATE_ROOT / "tests" / "kernel_contract.rs",
)


def _environment():
    return dict(os.environ)


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=CRATE_ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Rust source lock input missing: {path}")
        records.append(
            {
                "file": str(path.relative_to(REPO_ROOT)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def build_correction07_rust(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    if not CARGO.is_file() or not RUSTC.is_file():
        raise RuntimeError("pinned correction-07 Rust toolchain is unavailable")
    sources = _source_records()
    rustc_version = _run([str(RUSTC), "-Vv"])
    cargo_version = _run([str(CARGO), "-V"])
    _run([str(CARGO), "build", "--release", "--locked"])
    built_library = CRATE_ROOT / "target" / "release" / "libmaboost_analysis_kernel.so"
    if not built_library.is_file():
        raise RuntimeError("correction-07 Rust build did not emit the shared library")

    destination_root = assert_safe_output_path(Path(output_root))
    destination_root.mkdir(parents=True, exist_ok=True)
    library_path = assert_safe_output_path(
        destination_root / "libmaboost_analysis_correction07.so"
    )
    temporary_library = library_path.with_suffix(".so.tmp")
    shutil.copyfile(built_library, temporary_library)
    os.replace(temporary_library, library_path)
    library_hash = sha256_file(library_path)

    toolchain_payload = {
        "cargo": cargo_version,
        "rustc": rustc_version,
        "target": "x86_64-unknown-linux-gnu",
    }
    manifest_path = destination_root / "BUILD_MANIFEST.json"
    payload = {
        "schema_version": "1.0.0",
        "status": "built_hash_locked_for_qualification",
        "correction_id": "implementation_correction_07_analysis_engine",
        "abi": RUST_ABI,
        "abi_version": RUST_ABI_VERSION,
        "build_command": "cargo build --release --locked",
        "build_profile": "release",
        "target": "x86_64-unknown-linux-gnu",
        "dependencies": [],
        "source_files": sources,
        "source_tree_sha256": canonical_json_sha256(sources),
        "toolchain": toolchain_payload,
        "toolchain_sha256": canonical_json_sha256(toolchain_payload),
        "library": str(library_path.relative_to(REPO_ROOT)),
        "library_bytes": int(library_path.stat().st_size),
        "library_sha256": library_hash,
        "patient_level_content": False,
        "target_information_used": False,
    }
    safe_write_json(manifest_path, payload)
    safe_write_text(
        manifest_path.with_suffix(".json.sha256"),
        f"{sha256_file(manifest_path)}  {manifest_path.relative_to(REPO_ROOT)}\n",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and hash-lock the correction-07 Rust analysis kernel"
    )
    parser.parse_args()
    manifest_path = build_correction07_rust()
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": str(manifest_path.relative_to(REPO_ROOT)),
                "manifest_sha256": sha256_file(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
