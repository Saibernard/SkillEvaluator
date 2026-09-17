# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic fixtures. These are not model or human evaluations."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from skillevaluator.judge_validation.analysis import analyze, markdown
from skillevaluator.judge_validation.runner import prepare
from skillevaluator.judge_validation.schema import (
    METRICS,
    Annotations,
    Call,
    Case,
    Corpus,
    Rating,
    Recording,
    Recordings,
    Source,
    Specification,
    digest,
    save,
)


def _task(skill: str, n: int) -> tuple[str, str]:
    if skill == "arithmetic":
        return f"Calculate {n} + {n + 3}.", str(n + n + 3)
    if skill == "sorting":
        values = [n + 2, n, n + 1]
        return f"Sort {values} in ascending order.", json.dumps(sorted(values))
    if skill == "json-extraction":
        value = json.dumps({"id": n, "enabled": n % 2 == 0})
        return f"Extract the id from {value}.", str(json.loads(value)["id"])
    if skill == "csv-sum":
        value = f"amount\n{n}\n{n + 2}\n"
        total = sum(int(row["amount"]) for row in csv.DictReader(io.StringIO(value)))
        return f"Sum the amount column in this CSV: {value}", str(total)
    if skill == "text-count":
        value = "local " * n + "evaluation"
        return f"Count the words in {value!r}.", str(len(value.split()))
    value = {"prefix": "trace", "number": n}
    return f"Format {value} as prefix-number, zero-padding number to three digits.", f"trace-{n:03d}"


def demo_corpus() -> tuple[Corpus, Annotations]:
    cases = []
    labels = []
    skills = ["arithmetic", "sorting", "json-extraction", "csv-sum", "text-count", "formatting"]
    for index, skill in enumerate(skills):
        for n in range(1, 6):
            question, expected = _task(skill, n)
            for condition in ("with_skill", "without_skill"):
                actual = expected if condition == "with_skill" or n % 2 == 0 else "incorrect synthetic output"
                case_id = f"{skill}-{n}-{condition}"
                cases.append(
                    Case(
                        id=case_id,
                        skill_id=skill,
                        family_id=skill,
                        task_id=f"{skill}-{n}",
                        agent="synthetic-script",
                        condition=condition,
                        split="development" if index < 4 else "heldout",
                        source=Source(
                            kind="synthetic", reference="Built-in deterministic harness demonstration", revision="1"
                        ),
                        question=question,
                        ground_truth=expected,
                        agent_text=actual,
                        conversation=f"Request: {question}\nObserved output: {actual}",
                        tool_summary="A scripted demonstration produced the shown output.",
                        expected_behaviors=[f"Return the required output: {expected}"],
                    )
                )
                for metric in METRICS:
                    for reviewer in ("synthetic-oracle-a", "synthetic-oracle-b"):
                        labels.append(
                            Rating(
                                case_id=case_id,
                                metric=metric,
                                reviewer_id=reviewer,
                                score=float(actual == expected),
                                rationale="Synthetic label from exact equality with a deterministic expected output; no human review.",
                            )
                        )
    corpus = Corpus(
        id="judge-validation-demo",
        version="1",
        license="Apache-2.0",
        description="60 synthetic traces across six toy task families, with a skill-disjoint heldout partition.",
        cases=cases,
    )
    annotations = Annotations(
        corpus_digest=digest(corpus),
        rubric_version="synthetic-exact-output/1",
        origin="synthetic",
        ratings=labels,
    )
    return corpus, annotations


def write_demo(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=False)
    corpus, annotations = demo_corpus()
    plan = prepare(corpus, Specification(model="synthetic-fault-injector", model_revision="1", repeats=3))
    gold = {(r.case_id, r.metric): r.score for r in annotations.ratings}
    records = []
    case_indexes = {case.id: index for index, case in enumerate(corpus.cases)}
    for trial in plan.trials:
        index = case_indexes[trial.case_id]
        # Missing evidence, transport failures, invalid replies, repeat flips and
        # false positives/negatives exercise the entire report with known causes.
        if index % 23 == 0:
            continue
        score = gold[trial.case_id, trial.metric]
        if index % 7 == 0 or (trial.repetition == 1 and index % 5 == 0):
            score = 1.0 - score
        if index % 19 == 0:
            calls = [Call(prompt_digest=trial.prompts[0].digest, error="Synthetic transport failure")]
        elif index % 17 == 0:
            calls = [Call(prompt_digest=p.digest, content="Synthetic malformed reply") for p in trial.prompts]
        else:
            if trial.metric == "accuracy":
                response = {"score": score, "reason": "Synthetic judge reply"}
            elif trial.metric == "goal_accuracy":
                response = {"score": score, "achieved": score >= 0.5, "reason": "Synthetic judge reply"}
            else:
                response = {"results": [{"step": 1, "passed": score >= 0.5}], "summary": "Synthetic judge reply"}
            calls = [Call(prompt_digest=trial.prompts[0].digest, content=json.dumps(response))]
        records.append(Recording(case_id=trial.case_id, metric=trial.metric, repetition=trial.repetition, calls=calls))
    batch = Recordings(
        plan_digest=digest(plan),
        origin="synthetic",
        producer="Built-in fault-injection demonstration",
        recorded_at="synthetic, no model calls",
        records=records,
    )
    report = analyze(corpus, plan, batch, annotations)
    for name, value in [
        ("corpus", corpus),
        ("annotations", annotations),
        ("plan", plan),
        ("recordings", batch),
        ("report", report),
    ]:
        save(directory / f"{name}.json", value)
    (directory / "report.md").write_text(markdown(report), encoding="utf-8")
    return {
        "directory": str(directory),
        "cases": len(corpus.cases),
        "trials": len(plan.trials),
        "evidence_status": report["evidence_status"],
    }
