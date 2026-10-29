"""CLI adapter for the existing batch_label_schedule component.

Supports recovered bladderbatch or storage JSON metadata plus an NPZ containing
only X. Verifies retained NPZ SHA256 before NumPy loading; preserves metadata.
No fitting, downloads, primary-engine integration, or scientific-validity claim.
Schedule semantics, schema, cell budget and replay belong to the imported module.
"""
import argparse
import hashlib
import io
import json
import math
import sys
import zipfile

import numpy as np
from batch_label_schedule import make_schedule, validate_schedule

MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_MATRIX_BYTES = 512 * 1024 * 1024
MAX_SCHEDULE_BYTES = 256 * 1024 * 1024


def fail(message):
    raise ValueError(message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            fail(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    fail(f"Nonfinite JSON number: {value}")


def encode_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def read_json(path, limit):
    with open(path, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        fail(f"{path}: exceeds {limit} byte JSON limit")
    try:
        result = json.loads(raw, object_pairs_hook=unique_pairs,
                            parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"{path}: invalid JSON: {exc}")
    encode_json(result)  # also rejects overflow numbers such as 1e999
    if type(result) is not dict:
        fail(f"{path}: expected JSON object")
    return result


def retained_hashes(metadata, field):
    values = []
    for name in ("provenance", "storage"):
        section = metadata.get(name)
        if section is None:
            continue
        if type(section) is not dict:
            fail(f"{name}: expected object")
        if field not in section:
            continue
        value = section[field]
        if (type(value) is not str or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            fail(f"{name}.{field}: expected lowercase SHA256")
        values.append(value)
    return values


def load_dataset(metadata_path, matrix_path):
    metadata = read_json(metadata_path, MAX_METADATA_BYTES)
    if "X" in metadata:
        fail("JSON must not supply X; use the verified NPZ matrix")
    hashes = retained_hashes(metadata, "npz_sha256")
    if not hashes:
        fail("Require provenance.npz_sha256 or storage.npz_sha256")
    with open(matrix_path, "rb") as stream:
        raw = stream.read(MAX_MATRIX_BYTES + 1)
    if len(raw) > MAX_MATRIX_BYTES:
        fail("NPZ exceeds 512 MiB input limit")
    actual = hashlib.sha256(raw).hexdigest()
    if any(value != actual for value in hashes):
        fail("NPZ SHA256 mismatch; matrix was not loaded")
    # Bound decompression/allocation before np.load; load the same verified bytes.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "X.npy":
            fail("NPZ must contain exactly one member, X.npy")
        entry = entries[0]
        if entry.file_size > MAX_MATRIX_BYTES:
            fail("Decompressed X.npy exceeds 512 MiB")
        with archive.open(entry) as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                fail("Unsupported NPY header version")
            if (len(shape) != 2 or any(type(d) is not int or d < 1 for d in shape)
                    or dtype.kind not in "iuf"):
                fail("X must be a nonempty 2-D real numeric matrix")
            cells = math.prod(shape)
            if max(cells * dtype.itemsize, cells * 8) > MAX_MATRIX_BYTES:
                fail("X exceeds 512 MiB matrix working-input limit")
            if cells * dtype.itemsize != entry.file_size - member.tell():
                fail("NPY header/data size mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        X = archive["X"]
    if not np.isfinite(X).all():
        fail("X contains nonfinite values")
    matrix_hashes = retained_hashes(metadata, "matrix_sha256")
    if matrix_hashes:
        with np.errstate(over="ignore", invalid="ignore"):
            canonical_matrix = np.ascontiguousarray(X, dtype="<f8")
        if not np.isfinite(canonical_matrix).all():
            fail("X is not representable as finite canonical float64")
        matrix_hash = hashlib.sha256(canonical_matrix.tobytes(order="C")).hexdigest()
        if any(value != matrix_hash for value in matrix_hashes):
            fail("Canonical matrix SHA256 mismatch")
    # All semantic metadata, binding and schedule validation stays in the engine.
    return dict(metadata, X=X)


def report(schedule):
    names = ("binding", "diagnostics", "summary", "conditions",
             "assumptions_verified", "integrity_sha256")
    return {name: schedule[name] for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset-json", required=True)
        sub.add_argument("--matrix-npz", required=True)
        sub.add_argument("--max-schedule-cells", type=int, default=2000000,
                         help="engine budget: 2*n*(draws+1); default 2000000")
        if name == "generate":
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--draws", type=int, required=True)
            sub.add_argument("--out", required=True)
        else:
            sub.add_argument("--schedule", required=True)
            sub.add_argument("--structural-only", action="store_true",
                             help="explicitly disable default strict RNG replay")
    args = parser.parse_args(argv)
    try:
        if args.max_schedule_cells < 1:
            fail("--max-schedule-cells must be positive")
        # Additional serialized I/O cap; does not redefine the engine cell budget.
        limit = min(MAX_SCHEDULE_BYTES, args.max_schedule_cells * 32 + 1048576)
        dataset = load_dataset(args.dataset_json, args.matrix_npz)
        if args.command == "generate":
            schedule = make_schedule(dataset, seed=args.seed, draws=args.draws,
                                     max_schedule_cells=args.max_schedule_cells)
            output = report(schedule)
            payload = encode_json(schedule) + b"\n"
            if len(payload) > limit:
                fail(f"Schedule exceeds {limit} byte serialized output limit")
            output["output"] = args.out
            display = encode_json(output)
            with open(args.out, "xb") as stream:
                stream.write(payload)
        else:
            schedule = read_json(args.schedule, limit)
            validation = validate_schedule(
                dataset, schedule, strict_replay=not args.structural_only,
                max_schedule_cells=args.max_schedule_cells)
            output = report(schedule)
            output["validation"] = validation
            display = encode_json(output)
        print(display.decode("ascii"))
        print("Assumptions remain unverified. Refit the complete label-dependent "
              "preprocessing, selection and model pipeline for each draw.",
              file=sys.stderr)
        return 0
    except (ValueError, TypeError, KeyError, OSError, OverflowError, MemoryError,
            EOFError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
