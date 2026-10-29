"""Batch-conditioned label schedules; no model fitting or inference."""
import hashlib
import json
import math
import platform
from collections import Counter
from collections.abc import Mapping
import numpy as np

VERSION = "1"
ALGORITHM = "numpy.PCG64.group-index-permutation.v1"
MAX_SEED = (1 << 128) - 1
MAX_DRAWS = 100_000
DEFAULT_MAX_SCHEDULE_CELLS = 2_000_000
CONDITIONS = [
    "Within-batch exchangeability is an unverified scientific assumption.",
    "Subject independence is unverified. Repeated subject IDs require a justified dependence-aware design; this component does not permute subject blocks.",
    "Every draw requires the same complete label-dependent pipeline, including feature selection, tuning, fitting and evaluation, to be refitted.",
    "Conditioning on batch label counts cannot test between-batch confounding.",
    "Random schedules are controls, not automatic permission for inference or causal claims."
]

def _need(ok, message):
    if not ok:
        raise ValueError(message)

def _ids(value, n, name, unique=False):
    _need(isinstance(value, (list, tuple, np.ndarray)), name + ": sequence required")
    if isinstance(value, np.ndarray):
        _need(value.ndim == 1, name + ": one-dimensional sequence required")
    result = list(value)
    _need(len(result) == n, name + ": length mismatch")
    _need(all(isinstance(x, str) and x and x == x.strip() for x in result),
          name + ": nonempty strings without surrounding whitespace required")
    if unique:
        _need(len(set(result)) == n, name + ": duplicate identities")
    return result

def _target(value, n):
    _need(isinstance(value, (list, tuple, np.ndarray)), "y must be a sequence")
    if isinstance(value, np.ndarray):
        _need(value.ndim == 1, "y must be one-dimensional")
    values = list(value)
    _need(len(values) == n, "y length mismatch")
    _need(all(isinstance(v, (int, float, np.integer, np.floating))
              and not isinstance(v, (bool, np.bool_)) and v in (0, 1)
              for v in values), "y must contain numeric 0/1, never Boolean labels")
    return np.asarray(values, dtype=np.int64)

def _canonical_matrix(value):
    _need(isinstance(value, np.ndarray), "X must be an explicitly typed NumPy ndarray")
    X = value
    _need(X.ndim == 2 and min(X.shape) > 0, "X must be nonempty 2D")
    _need(X.dtype.kind in "fiu", "X must be real numeric, not bool/object/complex")
    _need(bool(np.isfinite(X).all()), "X must be finite")
    if X.dtype.kind == "f":
        _need(X.dtype.itemsize <= 8, "X floats wider than 64 bits are unsupported")
    else:
        _need(int(X.min()) >= -(1 << 53) and int(X.max()) <= (1 << 53),
              "X integers must lie within [-2**53, 2**53]")
    canonical = np.ascontiguousarray(X, dtype="<f8")
    _need(bool(np.isfinite(canonical).all()), "X is not finite after conversion")
    restored = canonical.astype(X.dtype)
    _need(bool(np.array_equal(restored, X)), "X is not losslessly representable")
    if X.dtype.kind == "f":
        _need(bool(np.array_equal(np.signbit(restored), np.signbit(X))),
              "X sign representation changed")
    return canonical

def _budget(n, seed, draws, max_schedule_cells):
    seed = _integer(seed, "seed")
    draws = _integer(draws, "draws")
    limit = _integer(max_schedule_cells, "max_schedule_cells", minimum=1)
    _need(seed <= MAX_SEED, "seed exceeds unsigned 128-bit domain")
    _need(draws <= MAX_DRAWS, "draws exceeds fixed 100000-draw cap")
    cells = 2 * n * (draws + 1)
    _need(cells <= limit, "schedule exceeds max_schedule_cells including baseline")
    return seed, draws, cells

def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)

def _same_json(observed, expected):
    try:
        return _canonical_json(observed) == _canonical_json(expected)
    except (TypeError, ValueError, OverflowError):
        return False

def _digest(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

def _integer(value, name, minimum=0):
    _need(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
          name + ": integer required")
    value = int(value)
    _need(value >= minimum, name + ": below minimum")
    return value

def _prepare(dataset):
    _need(isinstance(dataset, Mapping), "dataset must be a mapping")
    canonical = _canonical_matrix(dataset["X"])
    n, p = canonical.shape
    y = _target(dataset["y"], n)
    samples = _ids(dataset["sample_ids"], n, "sample_ids", True)
    groups = _ids(dataset["groups"], n, "groups")
    features = _ids(dataset["feature_names"], p, "feature_names", True)
    subjects = None
    if dataset.get("subject_ids") is not None:
        subjects = _ids(dataset["subject_ids"], n, "subject_ids")
    meta = {"sample_ids": samples, "groups": groups, "y": y.tolist(),
            "feature_names": features, "subject_ids": subjects}
    binding = {"shape": [n, p], "matrix_convention": "<f8 C",
        "matrix_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        "ordered_metadata_sha256": _digest(meta)}
    binding["dataset_sha256"] = _digest(binding)
    # First occurrence order is explicit and reproducible.
    grouped_positions = {}
    for position, group in enumerate(groups):
        grouped_positions.setdefault(group, []).append(position)
    names = list(grouped_positions)
    indices = [np.asarray(positions, dtype=np.int64)
               for positions in grouped_positions.values()]
    rows = []
    for name, idx in zip(names, indices):
        size, positive = len(idx), int(y[idx].sum())
        rows.append({"group": name, "n": size, "positive": positive,
            "negative": size-positive, "pure": positive in (0, size),
            "support": math.comb(size, positive),
            "max_hamming": 2*min(positive, size-positive)})
    diagnostics = {"groups": rows,
        "support": math.prod(row["support"] for row in rows),
        "frozen_labels": sum(row["n"] for row in rows if row["pure"]),
        "max_hamming": sum(row["max_hamming"] for row in rows),
        "subject_ids_provided": subjects is not None,
        "repeated_subject_ids": sorted(k for k, v in Counter(subjects or []).items() if v > 1)}
    return y, np.asarray(groups), indices, binding, meta, diagnostics

def _assignment(y, permutation, dataset_hash):
    labels = y[permutation]
    return {"dataset_sha256": dataset_hash,
            "index_permutation": permutation.tolist(),
            "labels": labels.tolist(),
            "hamming": int(np.count_nonzero(labels != y))}

def _summary(draws, original):
    keys = [tuple(row["labels"]) for row in draws]
    count = Counter(keys)
    return {"draw_count": len(keys),
        "unique_assignments": len(count),
        "repeated_assignment_draws": len(keys)-len(count),
        "assignment_patterns_repeated": sum(v > 1 for v in count.values()),
        "unchanged_assignment_draws": sum(k == tuple(original) for k in keys)}

def _schedule_digest(schedule):
    return _digest({k: v for k, v in schedule.items() if k != "integrity_sha256"})

def _support_status(diagnostics):
    return ("uninformative_under_conditioning" if diagnostics["support"] == 1
            else "nontrivial_label_support")

def _checked_summary(draws, original, diagnostics):
    summary = _summary(draws, original)
    summary["support_status"] = _support_status(diagnostics)
    return summary

def make_schedule(dataset, *, seed, draws,
                  max_schedule_cells=DEFAULT_MAX_SCHEDULE_CELLS):
    seed = _integer(seed, "seed")
    draws = _integer(draws, "draws")
    _need(seed <= MAX_SEED, "seed exceeds unsigned 128-bit domain")
    _need(draws <= MAX_DRAWS, "draws exceeds fixed 100000-draw cap")
    y, groups, indices, binding, meta, diagnostics = _prepare(dataset)
    seed, draws, cells = _budget(len(y), seed, draws, max_schedule_cells)
    rng = np.random.Generator(np.random.PCG64(seed))
    initial_state = rng.bit_generator.state
    rows = []
    for _ in range(draws):
        permutation = np.arange(len(y))
        for idx in indices:
            permutation[idx] = rng.permutation(idx)
        rows.append(_assignment(y, permutation, binding["dataset_sha256"]))
    schedule = {"schema": "nullgarden.batch-label-schedule.v1",
        "component_version": VERSION, "binding": binding,
        "ordered_metadata": meta, "diagnostics": diagnostics,
        "baseline": _assignment(y, np.arange(len(y)), binding["dataset_sha256"]),
        "draws": rows, "summary": _checked_summary(rows, y.tolist(), diagnostics),
        "resources": {"schedule_cells": cells,
            "cell_definition": "2*n_samples*(draw_count+1)",
            "generation_max_schedule_cells": int(max_schedule_cells),
            "fixed_max_draws": MAX_DRAWS},
        "rng": {"bit_generator": "PCG64", "seed": seed,
            "numpy_version": np.__version__, "python_version": platform.python_version(),
            "platform": platform.platform(),
            "initial_state": initial_state, "final_state": rng.bit_generator.state,
            "group_order": [row["group"] for row in diagnostics["groups"]],
            "algorithm": ALGORITHM},
        "conditions": CONDITIONS.copy(), "assumptions_verified": False}
    schedule["integrity_sha256"] = _schedule_digest(schedule)
    return schedule

def validate_schedule(dataset, schedule, *, strict_replay=False,
                      max_schedule_cells=DEFAULT_MAX_SCHEDULE_CELLS):
    """Integrity is unkeyed; replay verifies consistency, not authorship."""
    _need(isinstance(strict_replay, bool), "strict_replay must be Boolean")
    _need(isinstance(schedule, Mapping), "schedule must be a mapping")
    _need(isinstance(schedule.get("draws"), list), "draws must be a list")
    _need(len(schedule["draws"]) <= MAX_DRAWS, "draws exceeds fixed cap")
    y, groups, indices, binding, meta, diagnostics = _prepare(dataset)
    rng_record = schedule.get("rng")
    _need(isinstance(rng_record, Mapping), "RNG provenance missing")
    seed, draw_count, cells = _budget(
        len(y), rng_record.get("seed"), len(schedule["draws"]), max_schedule_cells)
    _need(schedule.get("integrity_sha256") == _schedule_digest(schedule),
          "whole-schedule integrity digest mismatch")
    _need(schedule.get("schema") == "nullgarden.batch-label-schedule.v1", "wrong schema")
    _need(schedule.get("component_version") == VERSION, "wrong component version")
    _need(_same_json(schedule.get("binding"), binding), "wrong dataset binding")
    _need(_same_json(schedule.get("ordered_metadata"), meta), "wrong ordered metadata")
    _need(_same_json(schedule.get("diagnostics"), diagnostics), "wrong diagnostics")
    _need(schedule.get("conditions") == CONDITIONS, "missing or changed conditions")
    _need(schedule.get("assumptions_verified") is False, "assumptions cannot be auto-verified")
    _need(isinstance(schedule.get("draws"), list), "draws must be a list")
    baseline = schedule.get("baseline")
    _need(isinstance(baseline, Mapping), "baseline missing")
    identity = np.arange(len(y))
    for number, row in enumerate([baseline] + schedule["draws"]):
        _need(isinstance(row, Mapping), "assignment must be a mapping")
        _need(row.get("dataset_sha256") == binding["dataset_sha256"], "wrong assignment binding")
        raw = row.get("index_permutation")
        _need(isinstance(raw, list) and len(raw) == len(y), "invalid permutation length")
        _need(all(isinstance(v, int) and not isinstance(v, bool) for v in raw),
              "permutation indices must be integers")
        _need(sorted(raw) == identity.tolist(), "permutation is not a bijection")
        permutation = np.asarray(raw, dtype=np.int64)
        _need(bool(np.array_equal(groups[permutation], groups)), "permutation crosses batches")
        labels = row.get("labels")
        _need(isinstance(labels, list) and len(labels) == len(y), "invalid labels")
        _need(all(isinstance(v, int) and not isinstance(v, bool) and v in (0, 1)
                  for v in labels), "labels must be binary integers")
        expected = y[permutation]
        _need(labels == expected.tolist(), "labels inconsistent with index permutation")
        for idx in indices:
            _need(int(expected[idx].sum()) == int(y[idx].sum()), "changed batch label counts")
        actual_hamming = int(np.count_nonzero(expected != y))
        _need(_integer(row.get("hamming"), "hamming") == actual_hamming,
              "wrong Hamming count")
        _need(actual_hamming <= diagnostics["max_hamming"], "Hamming bound exceeded")
        if number == 0:
            _need(raw == identity.tolist() and labels == y.tolist(), "baseline is not observed identity")
    _need(_same_json(schedule.get("summary"),
          _checked_summary(schedule["draws"], y.tolist(), diagnostics)), "wrong summary")
    resources = schedule.get("resources")
    _need(isinstance(resources, Mapping), "resource metadata missing")
    generation_limit = _integer(resources.get("generation_max_schedule_cells"),
                                "generation_max_schedule_cells", minimum=1)
    _need(cells <= generation_limit, "recorded generation budget exceeded")
    _need(_same_json(resources, {"schedule_cells": cells,
        "cell_definition": "2*n_samples*(draw_count+1)",
        "generation_max_schedule_cells": generation_limit,
        "fixed_max_draws": MAX_DRAWS}), "wrong resource metadata")
    _need(rng_record.get("bit_generator") == "PCG64", "PCG64 provenance required")
    _need(rng_record.get("algorithm") == ALGORITHM, "unsupported RNG algorithm")
    _need(rng_record.get("group_order") ==
          [row["group"] for row in diagnostics["groups"]], "wrong RNG group order")
    for key in ("numpy_version", "python_version", "platform"):
        _need(isinstance(rng_record.get(key), str) and bool(rng_record[key]),
              "missing RNG/runtime provenance")
    for key in ("initial_state", "final_state"):
        state = rng_record.get(key)
        _need(isinstance(state, Mapping), "missing RNG state")
        _need(state.get("bit_generator") == "PCG64", "RNG state is not PCG64")
    replay_status = "unverified"
    if strict_replay:
        _need(rng_record["numpy_version"] == np.__version__,
              "strict replay requires recorded NumPy version")
        replay_rng = np.random.Generator(np.random.PCG64(seed))
        _need(_digest(replay_rng.bit_generator.state) ==
              _digest(rng_record["initial_state"]), "initial RNG state/seed mismatch")
        for number, row in enumerate(schedule["draws"]):
            permutation = np.arange(len(y))
            for idx in indices:
                permutation[idx] = replay_rng.permutation(idx)
            _need(row["index_permutation"] == permutation.tolist(),
                  "strict RNG replay differs at draw " + str(number))
            _need(row["labels"] == y[permutation].tolist(),
                  "strict replay labels differ at draw " + str(number))
        _need(_digest(replay_rng.bit_generator.state) ==
              _digest(rng_record["final_state"]), "final RNG state differs")
        replay_status = "verified"
    return {"valid": True, "dataset_sha256": binding["dataset_sha256"],
        "draw_count": draw_count, "integrity": "verified_unkeyed_digest",
        "rng_replay": replay_status, "support_status": _support_status(diagnostics),
        "scientific_assumptions": "unverified",
        "scope": "dataset binding, integrity, structural invariants" +
            (" and deterministic PCG64 replay" if strict_replay else
             "; RNG replay explicitly unverified")}
