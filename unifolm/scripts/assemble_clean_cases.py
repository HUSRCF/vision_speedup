#!/usr/bin/env python3
"""Assemble independently timed, clean case runs and validate them officially."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CASES = {
    "case-stackbox": "unitree_z1_stackbox",
    "case-dual": "unitree_z1_dual_arm_stackbox",
    "case-v2": "unitree_z1_dual_arm_stackbox_v2",
    "case-pencils": "unitree_z1_dual_arm_cleanup_pencils",
    "case-g1": "unitree_g1_pack_camera",
}

REQUIRED_PROVENANCE = (
    "UNIFOLM_FULL_DENOISER_FP16=non-v2",
    "UNIFOLM_ATTENTION_DTYPE=fp16",
    "UNIFOLM_ATTENTION_PROJECTION_DTYPE=fp16",
)


def json_load(path: Path):
    return json.loads(path.read_text())


def valid_candidate(run_dir: Path, dataset: str) -> bool:
    required = (
        run_dir / "result.json",
        run_dir / "runtime.log",
        run_dir / "provenance.txt",
        run_dir / "profiles" / f"{dataset}.json",
        run_dir / "profiles" / "execution.json",
        run_dir / "profiles" / "psnr.json",
        run_dir / "outputs" / dataset,
    )
    if not all(path.exists() for path in required):
        return False
    provenance = (run_dir / "provenance.txt").read_text(errors="replace")
    if not all(line in provenance for line in REQUIRED_PROVENANCE):
        return False
    execution = json_load(run_dir / "profiles" / "execution.json")
    if execution.get(dataset, {}).get("exit_code") != 0:
        return False
    psnr = json_load(run_dir / "profiles" / "psnr.json")
    case_psnr = psnr.get("cases", {}).get(dataset, {})
    return case_psnr.get("passed") is True


def latest_runs(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for mode, dataset in CASES.items():
        candidates = sorted(
            root.glob(f"*-{mode}-*"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        selected[dataset] = next(
            (path for path in candidates if valid_candidate(path, dataset)), None
        )
        if selected[dataset] is None:
            raise SystemExit(f"No complete clean candidate found for {dataset}")
    return selected


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("a") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(f"Command failed ({completed.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path, default=Path.home() / "benchmark-runs" / "unifolm"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = latest_runs(args.runs_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or args.runs_root / f"{stamp}-assembled-clean-cases"
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    profiles = output / "profiles"
    outputs = output / "outputs"
    profiles.mkdir(parents=True)
    outputs.mkdir()

    execution = {}
    total_process_wall_s = 0.0
    manifest = {"schema_version": 1, "cases": {}}
    runtime_log = output / "runtime.log"
    for dataset, run_dir in selected.items():
        profile = json_load(run_dir / "profiles" / f"{dataset}.json")
        total_process_wall_s += float(profile["total_process_wall_s"])
        shutil.copy2(run_dir / "profiles" / f"{dataset}.json", profiles)
        execution[dataset] = {"exit_code": 0}
        os.symlink(run_dir / "outputs" / dataset, outputs / dataset)
        manifest["cases"][dataset] = {
            "run_dir": str(run_dir),
            "process_wall_s": profile["total_process_wall_s"],
        }
        with runtime_log.open("a") as combined, (run_dir / "runtime.log").open(
            errors="replace"
        ) as source:
            combined.write(f"\n===== CLEAN CASE {dataset} FROM {run_dir} =====\n")
            shutil.copyfileobj(source, combined)

    (profiles / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2) + "\n")

    psnr_tool = Path("/share/tools/unifolm-psnr.py")
    result_tool = Path("/share/tools/unifolm-result.py")
    run_logged(
        [
            "python3",
            str(psnr_tool),
            "--results",
            str(outputs),
            "--reference",
            "/share/results/unifolm/fp32/reference/outputs",
            "--report",
            str(profiles / "psnr.json"),
        ],
        runtime_log,
    )
    started = time.monotonic() - total_process_wall_s
    run_logged(
        [
            "python3",
            str(result_tool),
            "--profiles",
            str(profiles),
            "--result",
            str(output / "result.json"),
            "--exit-code",
            "0",
            "--started",
            str(started),
        ],
        runtime_log,
    )
    result = json_load(output / "result.json")
    if result.get("status") != "accepted":
        raise SystemExit(f"Assembled result was not accepted: {result.get('reason')}")
    print(output)
    print(f"total_process_wall_s={total_process_wall_s:.6f}")


if __name__ == "__main__":
    main()
