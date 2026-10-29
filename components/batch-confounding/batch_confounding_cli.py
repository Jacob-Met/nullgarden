"""CLI for descriptive batch diagnostics; no fitting, inference or downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import zipfile
import sys
import tempfile

SCHEDULE_COMPONENT = Path(__file__).resolve().parent.parent / "batch-label-schedules"
sys.path.insert(0, str(SCHEDULE_COMPONENT))
import numpy as np
from batch_schedule_cli import (
    load_dataset, read_json, MAX_METADATA_BYTES, MAX_MATRIX_BYTES,
    MAX_SCHEDULE_BYTES,
)
from batch_confounding import describe_batch_confounding, check_schedule_invariance
from diagnostic_report import render_report

MAX_REPORT_BYTES = 64 * 1024 * 1024


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def snapshot(source, destination, limit):
    """Copy a bounded regular-file input and hash the actual copied bytes."""
    source = Path(source)
    if not source.is_file():
        raise ValueError("Input must be a regular file: " + str(source))
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, destination.open("xb") as dst:
        if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
            raise ValueError("Input is not a regular file: " + str(source))
        while True:
            data = src.read(min(1024 * 1024, limit + 1 - size))
            if not data:
                break
            size += len(data)
            if size > limit:
                raise ValueError("Input exceeds byte limit: " + str(source))
            digest.update(data)
            dst.write(data)
    return {"name": source.name, "bytes": size, "sha256": digest.hexdigest()}


def outcome_values(dataset, path):
    if path is None:
        return None
    parts = path.split(".")
    if not all(part and part == part.strip() for part in parts):
        raise ValueError("Outcome path requires nonempty dot-separated object keys")
    value = dataset
    for part in parts:
        if type(value) is not dict or part not in value:
            raise ValueError("Outcome path does not identify a metadata field: " + path)
        value = value[part]
    if not isinstance(value, (list, tuple)):
        raise ValueError("Selected outcome field must be an aligned array")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Describe observed batch-label association; no model fitting.")
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--matrix-npz", required=True)
    parser.add_argument("--outcome-path",
                        help="Optional dot-separated metadata path, e.g. source_metadata.outcome")
    parser.add_argument("--schedule", help="Optional existing schedule to validate and inspect")
    parser.add_argument("--structural-only", action="store_true",
                        help="Explicitly omit RNG replay when validating a supplied schedule")
    parser.add_argument("--max-schedule-cells", type=int, default=2_000_000)
    parser.add_argument("--out", required=True, help="New output directory; existing paths refused")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.structural_only and not args.schedule:
        parser.error("--structural-only requires --schedule")
    try:
        if out.exists() or out.is_symlink():
            raise ValueError("Output directory must not exist: " + str(out))
        if args.max_schedule_cells < 1:
            raise ValueError("--max-schedule-cells must be positive")
        with tempfile.TemporaryDirectory(prefix="nullgarden-batch-inputs-") as temp:
            tmp = Path(temp)
            inputs = {
                "metadata": snapshot(args.dataset_json, tmp / "dataset.json", MAX_METADATA_BYTES),
                "matrix": snapshot(args.matrix_npz, tmp / "dataset.npz", MAX_MATRIX_BYTES),
            }
            dataset = load_dataset(tmp / "dataset.json", tmp / "dataset.npz")
            outcomes = outcome_values(dataset, args.outcome_path)
            report = describe_batch_confounding(
                dataset, outcomes=outcomes, max_schedule_cells=args.max_schedule_cells)
            invariance = None
            if args.schedule:
                inputs["schedule"] = snapshot(
                    args.schedule, tmp / "schedule.json", MAX_SCHEDULE_BYTES)
                schedule = read_json(tmp / "schedule.json", MAX_SCHEDULE_BYTES)
                invariance = check_schedule_invariance(
                    dataset, report, schedule, outcomes=outcomes,
                    strict_replay=not args.structural_only,
                    max_schedule_cells=args.max_schedule_cells)
            input_record = {
                "format": "nullgarden.batch-diagnostic-inputs.v1",
                "inputs": inputs, "outcome_path": args.outcome_path,
                "dataset_binding": report["dataset_binding"],
                "diagnostic_input_sha256": report["diagnostic_input_sha256"],
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": sys.platform},
                "source_files": {},
                "snapshot_semantics": "Hashes cover bounded regular-file snapshots actually parsed. Inputs are captured separately and their compatibility is checked by the loader. The CLI does not write to original input files.",
                "scientific_assumptions": "unverified",
            }
            for path in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "batch_confounding.py",
                Path(__file__).resolve().parent / "diagnostic_report.py",
                SCHEDULE_COMPONENT / "batch_label_schedule.py",
                SCHEDULE_COMPONENT / "batch_schedule_cli.py",
            ]:
                raw = path.read_bytes()
                input_record["source_files"][path.name] = {
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            files = {
                "report.json": json_bytes(report),
                "REPORT.md": render_report(report, invariance).encode("utf-8"),
                "INPUTS.json": json_bytes(input_record),
            }
            if invariance is not None:
                files["invariance.json"] = json_bytes(invariance)
            if sum(len(data) for data in files.values()) > MAX_REPORT_BYTES:
                raise ValueError("Report exceeds 64 MiB output bound")
            manifest = {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()}
            files["MANIFEST.json"] = json_bytes(manifest)
        # Input validation and report computation finish before output creation.
        # If an I/O failure occurs after mkdir, preserve the partial directory.
        out.mkdir(parents=False, exist_ok=False)
        for name, data in files.items():
            (out / name).open("xb").write(data)
        print(json.dumps({
            "out": str(out), "files": list(files),
            "diagnostic_input_sha256": report["diagnostic_input_sha256"],
            "dataset_sha256": report["dataset_binding"]["dataset_sha256"],
            "auc": report["score"]["auc"],
            "design": report["design"], "totals": report["totals"],
            "schedule_status": invariance["status"] if invariance else "not_supplied",
            "draws_checked": invariance["draws_checked"] if invariance else 0,
            "scientific_assumptions": "unverified",
        }, sort_keys=True, allow_nan=False))
        print("In-sample description only. Within-batch invariance does not establish scientific validity.", file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError, EOFError, zipfile.BadZipFile) as exc:
        print("error: " + str(exc), file=sys.stderr)
        if out.exists():
            print("Inspect the existing or partial output directory; nothing was overwritten or removed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
