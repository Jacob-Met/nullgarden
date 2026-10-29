"""Markdown rendering for validated batch-confounding diagnostic dictionaries."""
import json
import unicodedata

def _literal(value):
    """Quote untrusted values; preserve controls as visible escape sequences."""
    if value is None:
        return "Missing (null)"
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=True, allow_nan=False)
    parts = []
    for char in text:
        if char in "\n\r\t":
            parts.append({"\n": "&#92;n", "\r": "&#92;r", "\t": "&#92;t"}[char])
        elif unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp", "Cs"):
            parts.append(f"&#92;u{ord(char):04x}")
        elif char in '&<>"\'|\\*_{}[]()#+-.!~:' or char == chr(96):
            parts.append(f"&#{ord(char)};")
        else:
            parts.append(char)
    return "&quot;" + "".join(parts) + "&quot;"

def _number(value):
    return "Not available" if value is None else str(value)

def _flag(value):
    return "Yes" if value else "No"

def _fraction(value):
    return "Undefined" if value is None else f"{value['numerator']}/{value['denominator']}"

def _status(value):
    known = {"defined": "Defined", "undefined_single_class": "Undefined: single class",
             "provided": "Provided", "absent": "Absent",
             "verified": "Verified", "unverified": "Unverified"}
    return known.get(value, _literal(value)) if isinstance(value, str) else _literal(value)

def _binding_rows(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            label = f"{path}.{key}" if path else str(key)
            yield from _binding_rows(child, label)
    else:
        yield "| " + _literal(path) + " | " + _literal(value) + " |"

def render_report(report, invariance=None):
    t, d, score = report["totals"], report["design"], report["score"]
    auc = score["auc"]
    mixed, pure = t["mixed_batches"], t["pure_batches"]
    lines = [
        "# Batch-confounding diagnostic", "",
        "## Evaluated population and available comparisons", "",
        f"All input rows are included: **{t['n']} samples**, "
        f"{t['positive']} positive and {t['negative']} negative.",
        f"**{len(mixed)} mixed batches** contain both labels, providing "
        f"{t['within_batch_positive_negative_pairs']} within-batch positive-negative pairs.",
        f"Mixed batches contain {t['mixed_samples']} samples "
        f"({t['mixed_positive']} positive; {t['mixed_negative']} negative). "
        f"There are {len(pure)} pure batches.",
        "Mixed batch IDs: " + (", ".join(_literal(x) for x in mixed) or "None") + ".",
        "Pure batch IDs: " + (", ".join(_literal(x) for x in pure) or "None") + ".",
        "These are observed comparisons, not established independent or causal contrasts.", "",
        "## Exact design rank", "",
        "Design: one indicator per observed batch, without an additional intercept.",
        f"Batch rank: **{d['batch_rank']}**; batch-plus-target rank: "
        f"**{d['batch_plus_target_rank']}**.",
        ("The target lies in the batch-indicator span. No within-batch label "
         "variation is observed." if d["target_in_batch_indicator_span"] else
         "The target adds a dimension. This does not establish an unconfounded "
         "or well-supported effect."), "",
        "## Descriptive batch-prevalence score", "",
        "**Score(i) = observed positives in sample i's batch / samples in that batch.**",
        "The observed labels define and evaluate this in-sample score. "
        "It is not held-out predictive performance.",
        f"Uses observed labels: {_flag(score['uses_observed_labels'])}.",
    ]
    if t["positive"] == 0 or t["negative"] == 0:
        lines.append("**AUC undefined:** both classes are required. "
                     "There is no observed between-class comparison.")
    else:
        lines += [
            f"AUC: **{_fraction(auc['auc_fraction'])}** "
            f"(decimal: {_number(auc['auc_decimal'])}; status: {_status(auc['status'])}).",
            "AUC = (wins + ties / 2) / positive-negative pairs.",
            f"Wins: {auc['wins']}; ties: {auc['ties']}; losses: {auc['losses']}; "
            f"pairs: {auc['pairs']}.",
        ]
    lines += ["", "## Batch composition", "",
              "| Batch | N | Positive | Negative | Prevalence | Contrast available | Pairs |",
              "|---|---:|---:|---:|---|---|---:|"]
    for b in report["batches"]:
        lines.append(
            f"| {_literal(b['group'])} | {b['n']} | {b['positive']} | "
            f"{b['negative']} | {_fraction(b['prevalence'])} | "
            f"{_flag(b['within_batch_label_contrast_available'])} | "
            f"{b['positive_negative_pairs']} |")
    if report["outcome_metadata"]["provided"]:
        lines += ["", "## Recorded outcomes by batch", "",
                  "These are descriptive metadata, not additional validated targets. "
                  "Quoted strings are literal values; Missing (null) is a missing value.", "",
                  "| Batch | Outcome | N | Positive | Negative |",
                  "|---|---|---:|---:|---:|"]
        for b in report["batches"]:
            for row in b["outcomes"]:
                lines.append(
                    f"| {_literal(b['group'])} | {_literal(row['outcome'])} | "
                    f"{row['n']} | {row['positive']} | {row['negative']} |")
    else:
        lines += ["", "Outcome metadata was not provided."]
    s = report["subjects"]
    lines += ["", "## Subject metadata", "",
              f"Status: {_status(s['status'])}; unique subjects: {_number(s.get('unique_count'))}.",
              "Subject independence is not verified. Missing subject metadata "
              "cannot establish independent samples."]
    if s.get("repeated"):
        lines += ["", "| Repeated subject | N | Batches | Labels | Label conflict |",
                  "|---|---:|---|---|---|"]
        for row in s["repeated"]:
            batches = ", ".join(_literal(x) for x in row["batches"])
            labels = ", ".join(_literal(x) for x in row["labels"])
            lines.append(f"| {_literal(row['subject_id'])} | {row['n']} | "
                         f"{batches} | {labels} | {_flag(row['label_conflict'])} |")
        lines.append("Repeated records and conflicting subject labels require "
                     "investigation; this report does not resolve their interpretation.")
    lines += ["", "## Schedule invariance", ""]
    if invariance is None:
        lines.append("No schedule verification result was supplied.")
    else:
        explanations = {
            "verified": "Fixed-score AUC remained unchanged for all supplied draws. "
                        "This is expected and is not a scientific-validity pass.",
            "no_draws": "No random draws were supplied. Per-draw invariance was not checked.",
            "not_applicable_single_class": "AUC invariance is not applicable: only one class is present.",
        }
        lines += [
            explanations.get(invariance["status"],
                             "Reported status: " + _literal(invariance["status"])),
            f"Draws checked: {invariance['draws_checked']}.",
            "RNG replay: " + _status(invariance["schedule_validation"].get("rng_replay", "unverified")) + ".",
            "Schedule internal integrity SHA-256: "
            + _literal(invariance.get("schedule_integrity_sha256")) + ".",
        ]
    lines += [
        "Within-batch shuffles preserve batch class counts and therefore this score's "
        "AUC, when defined. Expected invariance does not show that leakage is absent. "
        "Pure batches cannot change label assignments.", "",
        "## Input identity", "", "| Binding field | Value |", "|---|---|",
    ]
    lines.extend(_binding_rows(report["dataset_binding"]))
    lines += ["", "Diagnostic input SHA-256: " + _literal(report["diagnostic_input_sha256"]) + ".",
              "INPUTS.json separately records the exact input-file snapshot hashes. "
              "A schedule file hash and its internal integrity digest identify different representations.",
              "", "## Limitations", "",
              "This descriptive report does not select or fit a model, provide p-values, "
              "establish exchangeability, resolve biological confounding, or validate "
              "clinical use. AUC 0.5 does not rule out other shortcuts. Downstream schedule "
              "evaluation must refit the complete label-dependent preprocessing, "
              "selection and model pipeline."]
    for item in report["limits"]:
        lines.extend(["", _literal(item)])
    return "\n".join(lines) + "\n"
