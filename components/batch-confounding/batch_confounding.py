"""Descriptive batch-confounding diagnostics; no fitting or inference."""
import hashlib
import json
from fractions import Fraction
import numpy as np
from batch_label_schedule import validate_dataset_identity, validate_schedule

SCHEMA = "nullgarden.batch-confounding-report.v1"
CHECK_SCHEMA = "nullgarden.batch-confounding-invariance.v1"
DEFAULT_MAX_SCHEDULE_CELLS = 2_000_000
LIMITS = [
    "The batch-prevalence score uses observed labels and is in-sample descriptive evidence, not held-out predictive performance.",
    "Available within-batch contrasts and design rank do not establish causal identification, stable estimation, or generalization.",
    "Within-batch randomization preserves batch label counts and cannot test discrimination arising solely from batch composition.",
    "Within-batch exchangeability and biological subject independence are unverified scientific assumptions.",
    "Recorded subject IDs are metadata; absent or unique IDs do not establish independence.",
    "Any downstream label-dependent pipeline requires complete refitting for each draw, including selection, tuning, fitting and evaluation.",
    "No p-values, fitted models, causal conclusions, or production-engine integration are supplied."
]

def _need(condition, message):
    if not condition:
        raise ValueError(message)

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)

def _same_json(left, right):
    try:
        return _json(left) == _json(right)
    except (TypeError, ValueError, OverflowError):
        return False

def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()

def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}

def _outcomes(value, n):
    if value is None:
        return {"provided": False, "values": None}
    _need(isinstance(value, (list, tuple, np.ndarray)),
          "outcomes must be a sequence or None")
    if isinstance(value, np.ndarray):
        _need(value.ndim == 1, "outcomes must be one-dimensional")
    values = list(value)
    _need(len(values) == n, "outcomes length mismatch")
    _need(all(v is None or (isinstance(v, str) and bool(v) and v == v.strip())
              for v in values),
          "outcomes must be None or nonempty trimmed strings")
    return {"provided": True, "values": values}

def _auc(counts, fixed_scores):
    """Counts map group -> [negative, positive]; scores are exact Fractions."""
    by_score = {}
    positive = negative = 0
    for group, (neg, pos) in counts.items():
        bucket = by_score.setdefault(fixed_scores[group], [0, 0])
        bucket[0] += neg
        bucket[1] += pos
        positive += pos
        negative += neg
    wins = ties = negatives_below = 0
    for score in sorted(by_score):
        neg, pos = by_score[score]
        wins += pos * negatives_below
        ties += pos * neg
        negatives_below += neg
    pairs = positive * negative
    losses = pairs - wins - ties
    _need(losses >= 0, "internal pair-count inconsistency")
    defined = positive > 0 and negative > 0
    value = Fraction(2 * wins + ties, 2 * pairs) if defined else None
    return {
        "status": "defined" if defined else "undefined_single_class",
        "wins": wins, "ties": ties, "losses": losses, "pairs": pairs,
        "auc_fraction": _fraction(value) if defined else None,
        "auc_decimal": float(value) if defined else None,
    }

def _subject_report(subject_ids, groups, labels):
    if subject_ids is None:
        return {"status": "absent", "unique_count": None,
                "repeated": [], "independence_verified": False}
    records = {}
    for subject, group, label in zip(subject_ids, groups, labels):
        record = records.setdefault(subject, {"n": 0, "batches": {}, "labels": set()})
        record["n"] += 1
        record["batches"].setdefault(group, None)
        record["labels"].add(label)
    repeated = []
    for subject, record in records.items():
        if record["n"] > 1:
            labels_seen = sorted(record["labels"])
            repeated.append({
                "subject_id": subject, "n": record["n"],
                "batches": list(record["batches"]), "labels": labels_seen,
                "label_conflict": len(labels_seen) > 1,
            })
    return {"status": "provided", "unique_count": len(records),
            "repeated": repeated, "independence_verified": False}

def describe_batch_confounding(
    dataset, *, outcomes=None,
    max_schedule_cells=DEFAULT_MAX_SCHEDULE_CELLS,
):
    """Describe observed batch/label association without fitting a model."""
    identity = validate_dataset_identity(
        dataset, max_schedule_cells=max_schedule_cells)
    binding = identity["binding"]
    metadata = identity["ordered_metadata"]
    labels, groups = metadata["y"], metadata["groups"]
    n = len(labels)
    outcome_metadata = _outcomes(outcomes, n)
    recorded = outcome_metadata["values"]
    counts, outcome_counts = {}, {}
    for i, (group, label) in enumerate(zip(groups, labels)):
        counts.setdefault(group, [0, 0])[label] += 1
        if outcome_metadata["provided"]:
            table = outcome_counts.setdefault(group, {})
            table.setdefault(recorded[i], [0, 0])[label] += 1
    batches, mixed, pure = [], [], []
    mixed_samples = mixed_positive = mixed_negative = within_pairs = 0
    fixed_scores = {}
    for group, (negative, positive) in counts.items():
        size = negative + positive
        score = Fraction(positive, size)
        fixed_scores[group] = score
        contrast = positive > 0 and negative > 0
        pairs = positive * negative
        within_pairs += pairs
        if contrast:
            mixed.append(group)
            mixed_samples += size
            mixed_positive += positive
            mixed_negative += negative
        else:
            pure.append(group)
        table = []
        for outcome, (neg, pos) in outcome_counts.get(group, {}).items():
            table.append({"outcome": outcome, "positive": pos,
                          "negative": neg, "n": neg + pos})
        batches.append({
            "group": group, "n": size, "positive": positive, "negative": negative,
            "prevalence": _fraction(score),
            "within_batch_label_contrast_available": contrast,
            "positive_negative_pairs": pairs, "outcomes": table,
        })
    positive = sum(labels)
    report = {
        "schema": SCHEMA, "dataset_binding": binding,
        "diagnostic_input_sha256": _hash({
            "dataset_binding": binding, "outcome_metadata": outcome_metadata}),
        "outcome_metadata": outcome_metadata,
        "batches": batches,
        "totals": {
            "n": n, "positive": positive, "negative": n-positive,
            "mixed_batches": mixed, "pure_batches": pure,
            "mixed_samples": mixed_samples, "mixed_positive": mixed_positive,
            "mixed_negative": mixed_negative,
            "within_batch_positive_negative_pairs": within_pairs,
        },
        "design": {
            "batch_rank": len(counts),
            "batch_plus_target_rank": len(counts) + int(bool(mixed)),
            "target_in_batch_indicator_span": not bool(mixed),
        },
        "subjects": _subject_report(metadata.get("subject_ids"), groups, labels),
        "score": {
            "name": "in_sample_batch_prevalence_score",
            "uses_observed_labels": True,
            "auc": _auc(counts, fixed_scores),
        },
        "limits": LIMITS.copy(),
    }
    return report

def check_schedule_invariance(
    dataset, report, schedule, *, outcomes=None, strict_replay=True,
    max_schedule_cells=DEFAULT_MAX_SCHEDULE_CELLS,
):
    """Check valid scheduled labels against fixed original batch scores."""
    expected = describe_batch_confounding(
        dataset, outcomes=outcomes, max_schedule_cells=max_schedule_cells)
    _need(_same_json(report, expected),
          "report differs from complete regenerated diagnostic report")
    validation = validate_schedule(
        dataset, schedule, strict_replay=strict_replay,
        max_schedule_cells=max_schedule_cells)
    # Public validation binds these ordered IDs and labels to this dataset.
    # Matrix rows and sample order remain fixed.
    groups = schedule["ordered_metadata"]["groups"]
    fixed_scores = {
        row["group"]: Fraction(row["prevalence"]["numerator"],
                               row["prevalence"]["denominator"])
        for row in expected["batches"]}
    baseline_auc = expected["score"]["auc"]
    draws = schedule["draws"]
    result = {
        "schema": CHECK_SCHEMA,
        "diagnostic_input_sha256": expected["diagnostic_input_sha256"],
        "schedule_integrity_sha256": schedule["integrity_sha256"],
        "schedule_validation": validation, "status": None,
        "draws_checked": 0, "baseline_auc": baseline_auc, "all_draws_equal": None,
    }
    # Single-class status takes precedence over the no-draw status.
    if baseline_auc["status"] == "undefined_single_class":
        result["status"] = "not_applicable_single_class"
        return result
    if not draws:
        result["status"] = "no_draws"
        return result
    for number, draw in enumerate(draws):
        counts = {group: [0, 0] for group in fixed_scores}
        for group, label in zip(groups, draw["labels"]):
            counts[group][label] += 1
        observed_auc = _auc(counts, fixed_scores)
        _need(_same_json(observed_auc, baseline_auc),
              "fixed batch-score AUC changed at draw " + str(number))
        result["draws_checked"] += 1
    result["status"] = "verified"
    result["all_draws_equal"] = True
    return result
