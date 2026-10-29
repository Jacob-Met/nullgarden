"""Independent exact-reference receiver for descriptive batch diagnostics."""
import copy
import hashlib
import itertools
import json
import sys
import traceback
from fractions import Fraction
import numpy as np
import batch_label_schedule as schedule
import batch_confounding as diagnostic

RESULTS = []

def case(name, fn):
    try:
        RESULTS.append({"name": name, "passed": True, "detail": fn()})
    except Exception:
        RESULTS.append({"name": name, "passed": False, "traceback": traceback.format_exc()})

def reject(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError("Invalid input was accepted")

def data(groups, labels):
    n = len(labels)
    return {"X": np.arange(n*2, dtype=float).reshape(n, 2),
            "y": list(labels), "groups": list(groups),
            "sample_ids": ["s"+str(i) for i in range(n)],
            "feature_names": ["a", "b"]}

def fraction(obj):
    return None if obj is None else Fraction(obj["numerator"], obj["denominator"])

def oracle(d, labels=None):
    y = d["y"] if labels is None else labels
    groups = d["groups"]
    order = list(dict.fromkeys(groups))
    scores = [
        Fraction(sum(d["y"][j] for j in range(len(groups)) if groups[j] == value),
                 groups.count(value)) for value in groups]
    wins = ties = losses = 0
    for i, positive in enumerate(y):
        if positive != 1:
            continue
        for j, negative in enumerate(y):
            if negative != 0:
                continue
            wins += scores[i] > scores[j]
            ties += scores[i] == scores[j]
            losses += scores[i] < scores[j]
    pairs = wins + ties + losses
    auc = Fraction(2*wins+ties, 2*pairs) if pairs else None
    design = np.array([[int(value == key) for key in order] for value in groups], dtype=int)
    ranks = (int(np.linalg.matrix_rank(design)),
             int(np.linalg.matrix_rank(np.column_stack((design, y)))))
    return {"wins": wins, "ties": ties, "losses": losses, "pairs": pairs,
            "auc": auc, "ranks": ranks}

def verify_report(d, report, outcomes=None):
    expected = oracle(d)
    auc = report["score"]["auc"]
    assert report["schema"] == "nullgarden.batch-confounding-report.v1"
    assert report["score"]["name"] == "in_sample_batch_prevalence_score"
    assert report["score"]["uses_observed_labels"] is True
    for key in ("wins", "ties", "losses", "pairs"):
        assert type(auc[key]) is int and auc[key] == expected[key]
    assert fraction(auc["auc_fraction"]) == expected["auc"]
    if expected["auc"] is None:
        assert auc["status"] == "undefined_single_class" and auc["auc_decimal"] is None
    else:
        assert auc["status"] == "defined"
        assert abs(auc["auc_decimal"] - float(expected["auc"])) < 1e-15
    design = report["design"]
    assert (design["batch_rank"], design["batch_plus_target_rank"]) == expected["ranks"]
    assert design["target_in_batch_indicator_span"] is (expected["ranks"][0] == expected["ranks"][1])
    assert report["outcome_metadata"] == {"provided": outcomes is not None, "values": outcomes}
    for row in report["batches"]:
        indices = [i for i, value in enumerate(d["groups"]) if value == row["group"]]
        positive = sum(d["y"][i] for i in indices)
        negative = len(indices) - positive
        assert (row["n"], row["positive"], row["negative"]) == (len(indices), positive, negative)
        assert fraction(row["prevalence"]) == Fraction(positive, len(indices))
        assert row["positive_negative_pairs"] == positive * negative
        assert row["within_batch_label_contrast_available"] is bool(positive and negative)
    return expected

FIXTURES = [
    ("single mixed", ["A", "A"], [0, 1], Fraction(1, 2)),
    ("pure", ["A", "A", "B", "B"], [0, 0, 1, 1], Fraction(1)),
    ("equal mixed", ["A", "A", "B", "B"], [0, 1, 0, 1], Fraction(1, 2)),
    ("equal ratios unequal sizes", ["A"]*2+["B"]*4, [0, 1, 0, 0, 1, 1], Fraction(1, 2)),
    ("unequal mixed", ["A"]*3+["B"]*3, [0, 0, 1, 0, 1, 1], Fraction(2, 3)),
    ("mixed pure", ["A", "A", "B", "B"], [0, 1, 1, 1], Fraction(5, 6)),
    ("unequal sizes", ["A"]*4+["B"]*2, [0, 0, 0, 1, 0, 1], Fraction(5, 8)),
    ("one class", ["A", "A", "B"], [0, 0, 0], None),
    ("one class positive", ["A", "A", "B"], [1, 1, 1], None),
]

def fixture_case(name, groups, labels, expected):
    d = data(groups, labels)
    report = diagnostic.describe_batch_confounding(d)
    baseline = verify_report(d, report)
    assert baseline["auc"] == expected
    blocks = [[i for i, value in enumerate(groups) if value == key] for key in dict.fromkeys(groups)]
    checked = 0
    for choices in itertools.product(*(list(itertools.permutations(indices)) for indices in blocks)):
        permuted = labels.copy()
        for indices, permutation in zip(blocks, choices):
            for destination, source in zip(indices, permutation):
                permuted[destination] = labels[source]
        assert oracle(d, permuted) == baseline  # Original fixed scores.
        changed = copy.deepcopy(d)
        changed["y"] = permuted
        received = diagnostic.describe_batch_confounding(changed)
        verify_report(changed, received)
        assert received["score"] == report["score"] and received["design"] == report["design"]
        checked += 1
    for draws in (0, 8):
        planned = schedule.make_schedule(d, seed=20261029, draws=draws)
        checked_schedule = diagnostic.check_schedule_invariance(d, report, planned)
        status = "not_applicable_single_class" if expected is None else ("no_draws" if draws == 0 else "verified")
        assert checked_schedule["status"] == status
        assert checked_schedule["schedule_validation"]["rng_replay"] == "verified"
        assert checked_schedule["draws_checked"] == (draws if status == "verified" else 0)
        assert checked_schedule["all_draws_equal"] is (True if status == "verified" else None)
        assert checked_schedule["baseline_auc"] == report["score"]["auc"]
    return {"exhaustive_index_permutations": checked, "auc": None if expected is None else str(expected)}

def duplicated():
    d = data(["A"]*3+["B"]*3, [0, 0, 1, 0, 1, 1])
    report = diagnostic.describe_batch_confounding(d)
    doubled = data([group for group in d["groups"] for _ in (0, 1)],
                   [label for label in d["y"] for _ in (0, 1)])
    received = diagnostic.describe_batch_confounding(doubled)
    verify_report(doubled, received)
    assert received["dataset_binding"] != report["dataset_binding"]
    assert received["score"]["auc"]["auc_fraction"] == report["score"]["auc"]["auc_fraction"]
    assert received["design"] == report["design"]
    assert received["score"]["auc"]["pairs"] == 4 * report["score"]["auc"]["pairs"]

def metadata():
    d = data(["A", "A", "B", "B"], [0, 1, 0, 1])
    d["subject_ids"] = ["p", "p", "q", "q"]
    outcomes = ["u", "v", "u", "v"]
    report = diagnostic.describe_batch_confounding(d, outcomes=outcomes)
    verify_report(d, report, outcomes)
    subjects = report["subjects"]
    assert subjects["status"] == "provided" and subjects["unique_count"] == 2
    assert subjects["independence_verified"] is False and len(subjects["repeated"]) == 2
    assert all(row["label_conflict"] is True and row["labels"] == [0, 1]
               for row in subjects["repeated"])
    planned = schedule.make_schedule(d, seed=1, draws=3)
    diagnostic.check_schedule_invariance(d, report, planned, outcomes=outcomes)
    reject(lambda: diagnostic.check_schedule_invariance(d, report, planned, outcomes=["v", "v", "u", "v"]))
    reject(lambda: diagnostic.check_schedule_invariance(d, report, planned))
    changed = copy.deepcopy(d)
    changed["subject_ids"] = ["p", "q", "r", "s"]
    reject(lambda: diagnostic.check_schedule_invariance(changed, report, planned, outcomes=outcomes))

def missing_metadata():
    d = data(["A", "A", "B", "B"], [0, 1, 0, 1])
    absent = diagnostic.describe_batch_confounding(d)
    all_null = diagnostic.describe_batch_confounding(d, outcomes=[None]*4)
    assert absent["diagnostic_input_sha256"] != all_null["diagnostic_input_sha256"]
    assert absent["outcome_metadata"]["provided"] is False
    assert all_null["outcome_metadata"]["provided"] is True
    report = diagnostic.describe_batch_confounding(d, outcomes=[None, "missing", None, "missing"])
    for batch in report["batches"]:
        table = {row["outcome"]: row for row in batch["outcomes"]}
        assert set(table) == {None, "missing"}
        assert (table[None]["n"], table[None]["negative"], table[None]["positive"]) == (1, 1, 0)
        assert (table["missing"]["n"], table["missing"]["negative"], table["missing"]["positive"]) == (1, 0, 1)
    for bad in ([None]*3, [1]*4, [" bad"]*4, [""]*4, np.array([[None]*4])):
        reject(lambda bad=bad: diagnostic.describe_batch_confounding(d, outcomes=bad))
    d["subject_ids"] = ["p", "p", "p", "q"]
    subjects = diagnostic.describe_batch_confounding(d)["subjects"]
    assert subjects["repeated"] == [{"subject_id": "p", "n": 3, "batches": ["A", "B"],
                                      "labels": [0, 1], "label_conflict": True}]

def tampering():
    d = data(["A", "A", "B", "B"], [0, 1, 0, 1])
    report = diagnostic.describe_batch_confounding(d)
    planned = schedule.make_schedule(d, seed=4, draws=8)
    mutations = [
        lambda t: t["score"]["auc"].update({"wins": False}),
        lambda t: t["design"].update({"target_in_batch_indicator_span": 0}),
        lambda t: t["subjects"].update({"independence_verified": 0}),
        lambda t: t["score"].update({"uses_observed_labels": 1}),
        lambda t: t["limits"].append("invented"),
        lambda t: t["batches"][0].update({"within_batch_label_contrast_available": 1}),
        lambda t: t.update({"schema": "nullgarden.batch-confounding-report.v99"}),
        lambda t: t.update({"unexpected": True}),
    ]
    for index, mutation in enumerate(mutations):
        def run(mutation=mutation):
            changed = copy.deepcopy(report)
            mutation(changed)
            reject(lambda: diagnostic.check_schedule_invariance(d, changed, planned))
        case("report semantic/Boolean tamper " + str(index), run)
    def rehash(value):
        value["integrity_sha256"] = hashlib.sha256(json.dumps(
            {key: item for key, item in value.items() if key != "integrity_sha256"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        return value
    changed = copy.deepcopy(planned)
    changed["draws"][0]["labels"][0] = 1 - changed["draws"][0]["labels"][0]
    rehash(changed)
    reject(lambda: diagnostic.check_schedule_invariance(d, report, changed))
    other_data = copy.deepcopy(d)
    other_data["X"][0, 0] += 1
    other = schedule.make_schedule(other_data, seed=4, draws=8)
    reject(lambda: diagnostic.check_schedule_invariance(d, report, other))
    reordered = copy.deepcopy(d)
    reordered["X"] = reordered["X"][::-1].copy()
    for key in ("y", "groups", "sample_ids"):
        reordered[key] = list(reversed(reordered[key]))
    reject(lambda: diagnostic.check_schedule_invariance(reordered, report, planned))
    one = data(["A", "A"], [0, 0])
    one_report = diagnostic.describe_batch_confounding(one)
    empty = schedule.make_schedule(one, seed=1, draws=0)
    empty["integrity_sha256"] = "0"*64
    reject(lambda: diagnostic.check_schedule_invariance(one, one_report, empty))
    empty = schedule.make_schedule(d, seed=1, draws=0)
    empty["integrity_sha256"] = "0"*64
    reject(lambda: diagnostic.check_schedule_invariance(d, report, empty))
    altered = copy.deepcopy(planned)
    altered["rng"]["seed"] = 5
    rehash(altered)
    reject(lambda: diagnostic.check_schedule_invariance(d, report, altered))
    received = diagnostic.check_schedule_invariance(d, report, altered, strict_replay=False)
    assert received["schedule_validation"]["rng_replay"] == "unverified"

if __name__ == "__main__":
    for name, groups, labels, expected in FIXTURES:
        case(name, lambda name=name, groups=groups, labels=labels, expected=expected:
             fixture_case(name, groups, labels, expected))
    case("equal duplication", duplicated)
    case("outcomes subjects binding", metadata)
    case("missing outcomes and cross-batch subjects", missing_metadata)
    case("schedule and report tampering", tampering)
    passed = all(row["passed"] for row in RESULTS)
    print(json.dumps({
        "schema": "nullgarden.independent-confounding-receiver.v1",
        "passed": passed, "cases": RESULTS,
        "limits": ["descriptive in-sample association only", "no fitted nuisance model",
                   "invariance is expected by conditioning, not causal identification"],
    }, indent=2, allow_nan=False))
    raise SystemExit(0 if passed else 1)
