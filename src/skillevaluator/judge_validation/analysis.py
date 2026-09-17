# SPDX-License-Identifier: Apache-2.0
"""Descriptive agreement and variability without treating repeats as new cases."""

from __future__ import annotations

import itertools
import statistics
from collections import Counter, defaultdict
from typing import Any

from skillevaluator.judge_validation.runner import replay
from skillevaluator.judge_validation.schema import Annotations, Corpus, Plan, Rating, Recordings, digest
from skillevaluator.tier3.harbor.collector import _wilson_score_interval


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
        "descriptive_wilson_95": _wilson_score_interval(numerator, denominator),
    }


def _labels(annotations: Annotations | None) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    if annotations is None:
        return {}, {}
    ratings: dict[tuple[str, str], list[Rating]] = defaultdict(list)
    for rating in annotations.ratings:
        ratings[rating.case_id, rating.metric].append(rating)
    gold: dict[tuple[str, str], float] = {}
    states = {}
    for key, entries in ratings.items():
        values = {r.score for r in entries}
        if len(entries) < 2:
            states[key] = "insufficient_reviewers"
        elif len(values) == 1 and None not in values:
            gold[key] = entries[0].score
            states[key] = "consensus"
        else:
            states[key] = "unresolved"
    for resolution in annotations.adjudications:
        key = (resolution.case_id, resolution.metric)
        gold[key] = resolution.score
        states[key] = "adjudicated"
    return gold, states


def _agreement(left: list[float], right: list[float], threshold: float) -> dict[str, Any]:
    n = len(left)
    if not n:
        return {"n": 0, "exact_score_agreement": None, "decision_agreement": None, "cohen_kappa": None}
    a = [score >= threshold for score in left]
    b = [score >= threshold for score in right]
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return {
        "n": n,
        "exact_score_agreement": sum(x == y for x, y in zip(left, right, strict=True)) / n,
        "decision_agreement": observed,
        "cohen_kappa": (observed - chance) / (1 - chance) if chance < 1 else None,
    }


def _reviewer_agreement(
    annotations: Annotations | None, case_ids: set[str], metric: str, threshold: float
) -> list[dict]:
    if annotations is None:
        return []
    ratings: dict[str, dict[str, float]] = defaultdict(dict)
    for rating in annotations.ratings:
        if rating.case_id in case_ids and rating.metric == metric and rating.score is not None:
            ratings[rating.reviewer_id][rating.case_id] = rating.score
    pairs = []
    for a, b in itertools.combinations(sorted(ratings), 2):
        common = sorted(ratings[a].keys() & ratings[b].keys())
        pairs.append(
            {
                "reviewer_a": a,
                "reviewer_b": b,
                **_agreement([ratings[a][key] for key in common], [ratings[b][key] for key in common], threshold),
            }
        )
    return pairs


def _group(
    rows: list[dict], gold: dict, label_states: dict, annotations: Annotations | None, plan: Plan
) -> dict[str, Any]:
    threshold = plan.specification.threshold
    initial = [row for row in rows if row["repetition"] == 0]
    labeled = [row for row in initial if (row["case_id"], row["metric"]) in gold]
    evaluated = [row for row in labeled if row["status"] == "scored"]
    actual = [gold[row["case_id"], row["metric"]] for row in evaluated]
    predicted = [row["score"] for row in evaluated]
    counts = Counter(
        (label >= threshold, prediction >= threshold) for label, prediction in zip(actual, predicted, strict=True)
    )
    tp, fn = counts[True, True], counts[True, False]
    fp, tn = counts[False, True], counts[False, False]
    repeats: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        repeats[row["case_id"]].append(row)
    complete = [
        entries
        for entries in repeats.values()
        if len(entries) == plan.specification.repeats
        and len(entries) >= 2
        and all(row["status"] == "scored" for row in entries)
    ]
    variability = [
        {
            "case_id": entries[0]["case_id"],
            "score_range": max(row["score"] for row in entries) - min(row["score"] for row in entries),
            "score_stddev": statistics.pstdev(row["score"] for row in entries),
            "decision_flipped": len({row["score"] >= threshold for row in entries}) > 1,
        }
        for entries in complete
    ]
    metric = rows[0]["metric"]
    return {
        **{key: rows[0][key] for key in ("metric", "split", "condition", "agent")},
        "planned_cases": len(initial),
        "planned_trials": len(rows),
        "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
        "reference_label_states": dict(
            sorted(Counter(label_states.get((row["case_id"], metric), "unlabeled") for row in initial).items())
        ),
        "labeled_cases": len(labeled),
        "evaluated_labeled_cases": len(evaluated),
        "labeled_case_coverage": _rate(len(evaluated), len(labeled)),
        "confusion_matrix": {"true_pass": tp, "false_pass": fp, "true_fail": tn, "false_fail": fn},
        "false_pass_rate": _rate(fp, fp + tn),
        "false_fail_rate": _rate(fn, fn + tp),
        "mean_absolute_error": statistics.mean(abs(a - b) for a, b in zip(actual, predicted, strict=True))
        if actual
        else None,
        "judge_label_agreement": _agreement(actual, predicted, threshold),
        "reviewer_pair_agreement": _reviewer_agreement(annotations, set(repeats), metric, threshold),
        "repeatability": {
            "complete_cases": len(complete),
            "excluded_cases": len(repeats) - len(complete),
            "decision_flip_rate": _rate(sum(item["decision_flipped"] for item in variability), len(complete)),
            "mean_score_stddev": statistics.mean(item["score_stddev"] for item in variability) if variability else None,
            "cases": variability,
        },
        "errors": [
            {"case_id": row["case_id"], "repetition": row["repetition"], "status": row["status"]}
            for row in rows
            if row["status"] != "scored"
        ],
        "disagreements": [
            {"case_id": row["case_id"], "label": gold[row["case_id"], metric], "score": row["score"]}
            for row in evaluated
            if row["score"] != gold[row["case_id"], metric]
        ],
    }


def analyze(corpus: Corpus, plan: Plan, batch: Recordings, annotations: Annotations | None = None) -> dict[str, Any]:
    if annotations is not None:
        annotations.check_corpus(corpus)
    rows = replay(corpus, plan, batch)
    gold, states = _labels(annotations)
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["metric"], row["split"], row["condition"], row["agent"]].append(row)
    synthetic = batch.origin == "synthetic" or any(case.source.kind == "synthetic" for case in corpus.cases)
    if annotations is not None and annotations.origin == "synthetic":
        synthetic = True
    return {
        "schema_version": "judge-analysis/1",
        "evidence_status": "synthetic_demonstration" if synthetic else "empirical_records_unverified_origin",
        "corpus_digest": digest(corpus),
        "plan_digest": digest(plan),
        "recordings_digest": digest(batch),
        "annotations_digest": digest(annotations) if annotations is not None else None,
        "annotation_origin": annotations.origin if annotations is not None else None,
        "rubric_version": annotations.rubric_version if annotations is not None else None,
        "implementation_digest": plan.implementation_digest,
        "evaluator_version": plan.evaluator_version,
        "evaluator_revision": plan.evaluator_revision,
        "worktree_dirty": plan.worktree_dirty,
        "specification": plan.specification.model_dump(mode="json"),
        "recording_origin": batch.origin,
        "response_models": sorted({model for row in rows for model in row.get("response_models", [])}),
        "limitations": [
            "Label origin and reviewer independence are declarations, not externally authenticated facts.",
            "Synthetic examples test the harness; they do not validate a real model or establish human agreement.",
            "Primary accuracy uses repetition zero once per case, never whichever repeat scores best.",
            "Missing recordings, missing references and judge/transport errors have no numeric score.",
            "Error rates condition on scored, resolved labels; inspect labeled coverage and failures alongside them.",
            "Wilson intervals are descriptive case-level summaries; related tasks may violate independence.",
            "Results describe this corpus, rubric, threshold, model and prompt version; they do not prove generalization.",
            "Rubric scores are not calibrated probabilities of success. Aggregate Skill Lift is not measured here.",
            "Repeatability requires all planned repeats to score; excluded cases remain explicitly counted.",
        ],
        "groups": [_group(groups[key], gold, states, annotations, plan) for key in sorted(groups)],
        "observations": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    def escaped(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    def percent(rate: dict) -> str:
        return (
            "N/A"
            if rate["value"] is None
            else f"{100 * rate['value']:.1f}% ({rate['numerator']}/{rate['denominator']})"
        )

    lines = [
        "# Judge validation report",
        "",
        f"Evidence status: **{escaped(report['evidence_status'])}**",
        "",
        f"Model: {escaped(report['specification']['model'])}",
        "",
        "Primary measurements use the first planned repetition of each case.",
        "",
        "| Metric | Split | Condition | Agent | Labeled coverage | False passes | False failures | Repeat flips |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        cells = [group[key] for key in ("metric", "split", "condition", "agent")]
        cells += [percent(group[key]) for key in ("labeled_case_coverage", "false_pass_rate", "false_fail_rate")]
        cells += [percent(group["repeatability"]["decision_flip_rate"])]
        lines.append("| " + " | ".join(escaped(cell) for cell in cells) + " |")
    lines += ["", "## Limits", "", *[f"- {item}" for item in report["limitations"]], "", "## Artifact identity", ""]
    lines += [
        f"- {key}: `{report[key]}`"
        for key in ("corpus_digest", "plan_digest", "recordings_digest", "annotations_digest", "implementation_digest")
    ]
    return "\n".join(lines) + "\n"
