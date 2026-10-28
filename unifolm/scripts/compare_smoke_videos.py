"""Compare the two first-round videos using the unchanged shared PSNR code.

This does not certify a full case: the one-round full video is deliberately
excluded because its frame count differs from the required complete trajectory.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("official_psnr", "/share/tools/unifolm-psnr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = Path("/share/results/unifolm/fp32/reference/outputs")
    videos = sorted(args.results.rglob("itr-0.mp4"))
    if len(videos) != 2:
        raise RuntimeError(f"Expected two first-round videos, got {len(videos)}")
    records = []
    for actual in videos:
        relative = actual.relative_to(args.results)
        result = module.compare_video(actual, reference / relative)
        records.append({"video": str(relative), **result})
        print("SMOKE_VIDEO_PSNR", relative, json.dumps(result, allow_nan=False), flush=True)
    args.report.write_text(json.dumps({
        "scope": "first-round diagnostic only; full case not validated",
        "videos": records,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
