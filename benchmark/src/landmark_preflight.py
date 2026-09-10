from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from landmark_protocol import (
    REPO_ROOT,
    RESULTS_ROOT,
    WORKSPACE_ROOT,
    carevue_official_identity_probe,
    load_protocol,
    raw_manifest,
    verify_frozen_state,
    write_phase_report,
)
from v13_feature_registry import DYNAMIC_FEATURES, STATIC_FEATURES
from v13_guard import safe_write_json, safe_write_text, sha256_file
from v13_model_registry import MODEL_NAMES, build_flat_design, validate_model_registry


def _test(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_preflight(*, deep_hash: bool = False) -> tuple[dict[str, Any], int]:
    plan, plan_hash, config, config_hash = load_protocol()
    tests: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        frozen = verify_frozen_state(plan)
        tests.append(_test("frozen_v11_snapshot", True, "all three locked states match"))
    except BaseException as error:
        frozen = {}
        tests.append(_test("frozen_v11_snapshot", False, str(error)))
        errors.append(str(error))

    manifest = raw_manifest(config, deep_hash=deep_hash)
    manifest_path = RESULTS_ROOT / "preflight" / "raw_manifest_private_names_only.json"
    safe_write_json(manifest_path, manifest)
    safe_write_text(
        manifest_path.with_suffix(".json.sha256"),
        f"{sha256_file(manifest_path)}  {manifest_path.relative_to(REPO_ROOT)}\n",
    )
    for dataset, record in manifest["datasets"].items():
        tests.append(_test(
            f"raw_{dataset}",
            bool(record["complete"]),
            "required gzip CSV probes passed" if record["complete"] else (
                f"incomplete raw root: {record['raw_root']}"
            ),
        ))
        if not record["complete"]:
            missing = [
                item["path"]
                for item in record["required_files"]
                if not item.get("exists") or not item.get("gzip_header_and_sample_ok")
            ]
            errors.append(f"{dataset} raw gate incomplete: {', '.join(missing)}")

    carevue_identity = carevue_official_identity_probe(config)
    tests.append(_test(
        "official_carevue_nonoverlap_resource",
        bool(carevue_identity["verified"]),
        (
            "official v1.4 URL/DOI locked; all local ICUSTAYS DBSOURCE values are carevue"
            if carevue_identity["verified"]
            else carevue_identity.get("error", "official CareVue content invariant failed")
        ),
    ))

    package_names = {
        "numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
        "torch": "torch", "xgboost": "xgboost", "shap": "shap",
        "matplotlib": "matplotlib", "psutil": "psutil", "mamba-ssm": "mamba_ssm",
    }
    versions: dict[str, str] = {"python": platform.python_version()}
    missing_packages: list[str] = []
    for distribution, module in package_names.items():
        if importlib.util.find_spec(module) is None:
            missing_packages.append(distribution)
            continue
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "importable-version-metadata-unavailable"
    tests.append(_test(
        "runtime_dependencies",
        not missing_packages,
        f"versions={versions}" if not missing_packages else f"missing={missing_packages}",
    ))
    command_tools = {
        name: shutil.which(name)
        for name in ("cargo", "rustc")
    }
    tests.append(_test(
        "reporting_toolchain",
        all(command_tools.values()),
        str(command_tools),
    ))
    disk = shutil.disk_usage(WORKSPACE_ROOT)
    minimum_free = max(
        20_000_000_000,
        int(config["matched_augmentation"]["minimum_free_cache_bytes"]),
    )
    tests.append(_test(
        "workspace_free_space",
        disk.free >= minimum_free,
        f"free_bytes={disk.free}; required_bytes={minimum_free}",
    ))
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except BaseException as error:
        cuda = {"available": False, "error": f"{type(error).__name__}: {error}"}
    tests.append(_test(
        "cuda_training_visibility",
        True,
        (
            f"CUDA visible: {cuda['device_names']}"
            if cuda.get("available")
            else "CUDA not visible in preflight; ETL may proceed but smoke will remain blocked"
        ),
    ))

    dynamic_ok = len(DYNAMIC_FEATURES) == 40 and len(set(DYNAMIC_FEATURES)) == 40
    static_ok = tuple(STATIC_FEATURES) == tuple(plan["feature_contract"]["static_features"])
    tests.append(_test("dynamic_contract", dynamic_ok, f"{len(DYNAMIC_FEATURES)} variables"))
    tests.append(_test(
        "time_safe_static_contract",
        static_ok and len(STATIC_FEATURES) == 12,
        f"{len(STATIC_FEATURES)} admission-time/context variables; diagnosis tables excluded",
    ))
    try:
        validate_model_registry(config["models"])
        model_ok = tuple(config["models"]) == MODEL_NAMES
    except BaseException as error:
        model_ok = False
        errors.append(str(error))
    tests.append(_test("model_registry", model_ok, f"{len(config['models'])} locked labels"))

    scenarios = config["stress_scenarios"]
    scenario_ok = (
        len(scenarios) == 28
        and len(set(scenarios)) == 28
        and scenarios == plan["stress_scenarios"]
    )
    tests.append(_test("stress_registry", scenario_ok, f"{len(scenarios)} unique scenarios"))

    processed_root = WORKSPACE_ROOT / "private" / "processed"
    plan_mtime = (WORKSPACE_ROOT / "analysis_plan_v13_landmark_amendment_02.json").stat().st_mtime_ns
    earlier_processed = [
        path.relative_to(WORKSPACE_ROOT).as_posix()
        for path in processed_root.rglob("*")
        if path.is_file() and path.stat().st_mtime_ns < plan_mtime
    ] if processed_root.exists() else []
    tests.append(_test(
        "plan_precedes_corrected_etl",
        not earlier_processed,
        "no corrected processed artifact predates the active plan"
        if not earlier_processed else f"predating artifacts: {earlier_processed[:5]}",
    ))

    manuscript_exists = (WORKSPACE_ROOT / "manuscript").exists()
    tests.append(_test(
        "manuscript_gate_closed",
        not manuscript_exists,
        "no manuscript tree exists in the computational workspace",
    ))

    for test in tests:
        if not test["passed"] and test["detail"] not in errors:
            errors.append(f"{test['name']}: {test['detail']}")
    status = "pass" if not errors else "blocked"
    report_payload = {
        "status": status,
        "command": (
            f"{sys.executable} -u "
            "reproduce.py preflight"
        ),
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "frozen_v11": frozen,
        "raw_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "raw_manifest_sha256": sha256_file(manifest_path),
        "carevue_official_identity": carevue_identity,
        "runtime_environment": {
            "versions": versions,
            "command_tools": command_tools,
            "cuda": cuda,
            "disk": {
                "total_bytes": int(disk.total),
                "used_bytes": int(disk.used),
                "free_bytes": int(disk.free),
            },
        },
        "tests": tests,
        "errors": errors,
        "etl_performed": False,
        "training_performed": False,
        "ready_for_etl": not errors,
        "ready_for_gpu_smoke": bool(not errors and cuda.get("available")),
    }
    report_path, report_hash = write_phase_report("A_PREFLIGHT", report_payload)
    result = {
        "status": status,
        "analysis_plan_sha256": plan_hash,
        "config_sha256": config_hash,
        "report": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": report_hash,
        "errors": errors,
    }
    return result, 0 if status == "pass" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-hash", action="store_true")
    args = parser.parse_args()
    result, code = run_preflight(deep_hash=args.deep_hash)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
