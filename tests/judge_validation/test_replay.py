# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from skillevaluator.judge_validation.analysis import analyze, markdown
from skillevaluator.judge_validation.runner import prepare, replay
from skillevaluator.judge_validation.schema import Adjudication, Annotations, Call, Corpus, Specification, digest

from .conftest import recordings, response


def spec(**kwargs) -> Specification:
    return Specification(model="synthetic-model", model_revision="fixture-v1", **kwargs)


def test_confusion_matrix_rates_and_repeats_have_case_denominators(corpus: Corpus, annotations: Annotations) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=3))
    scores = {(f"case-{i}", repeat): value for i, value in enumerate([0.0, 1.0, 1.0, 0.0]) for repeat in range(3)}
    scores["case-0", 1] = 1.0
    report = analyze(corpus, plan, recordings(plan, scores), annotations)
    group = report["groups"][0]
    assert group["confusion_matrix"] == {"true_pass": 1, "false_pass": 1, "true_fail": 1, "false_fail": 1}
    assert group["false_pass_rate"]["value"] == group["false_fail_rate"]["value"] == 0.5
    assert group["false_pass_rate"]["denominator"] == 2
    assert group["mean_absolute_error"] == 0.5
    assert group["judge_label_agreement"]["cohen_kappa"] == 0.0
    assert group["repeatability"]["decision_flip_rate"]["value"] == 0.25
    assert report["evidence_status"] == "synthetic_demonstration"
    assert "synthetic_demonstration" in markdown(report)


@pytest.mark.parametrize("metric", ["accuracy", "goal_accuracy", "behavior_check"])
def test_retry_uses_real_prompt_and_parser(corpus: Corpus, metric: str) -> None:
    plan = prepare(corpus, spec(metrics=[metric], repeats=1))
    batch = recordings(plan)
    trial = plan.trials[0]
    batch.records[0].calls = [
        Call(prompt_digest=trial.prompts[0].digest, content="invalid"),
        Call(prompt_digest=trial.prompts[1].digest, content=response(metric, 0.0)),
    ]
    row = replay(corpus, plan, batch)[0]
    assert row["status"] == "scored" and row["score"] == 0.0 and row["call_count"] == 2


@pytest.mark.parametrize("metric", ["accuracy", "goal_accuracy", "behavior_check"])
def test_parse_failure_is_not_valid_zero(corpus: Corpus, metric: str) -> None:
    plan = prepare(corpus, spec(metrics=[metric], repeats=1))
    batch = recordings(plan)
    batch.records[0].calls = [Call(prompt_digest=p.digest, content="invalid") for p in plan.trials[0].prompts]
    row = replay(corpus, plan, batch)[0]
    assert row["status"] == "judge_error" and row["score"] is None


def test_missing_errors_and_references_are_separate(corpus: Corpus, annotations: Annotations) -> None:
    corpus.cases[3].ground_truth = ""
    annotations.ratings = [r for r in annotations.ratings if r.case_id != "case-3"]
    annotations.corpus_digest = digest(corpus)
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    batch = recordings(plan)
    batch.records[0].calls[0].content = ""
    batch.records[0].calls[0].error = "local server unavailable"
    batch.records.pop(1)
    report = analyze(corpus, plan, batch, annotations)
    group = report["groups"][0]
    assert group["statuses"] == {"transport_error": 1, "missing_recording": 1, "missing_reference": 1, "scored": 1}
    assert group["evaluated_labeled_cases"] == 1
    assert group["labeled_cases"] == 3
    assert group["false_pass_rate"]["value"] is None
    assert group["repeatability"]["complete_cases"] == 0


def test_disagreement_needs_explicit_adjudication(corpus: Corpus, annotations: Annotations) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    annotations.ratings[1].score = 0.0
    group = analyze(corpus, plan, recordings(plan), annotations)["groups"][0]
    assert group["labeled_cases"] == 3 and group["reference_label_states"]["unresolved"] == 1
    annotations.adjudications.append(
        Adjudication(
            case_id="case-0",
            metric="accuracy",
            reviewer_id="synthetic-adjudicator",
            score=0.0,
            rationale="Synthetic override",
        )
    )
    group = analyze(corpus, plan, recordings(plan), annotations)["groups"][0]
    assert group["labeled_cases"] == 4 and group["reference_label_states"]["adjudicated"] == 1
    assert group["reviewer_pair_agreement"][0]["decision_agreement"] == 0.75


def test_single_reviewer_and_unknown_labels_do_not_become_gold(corpus: Corpus, annotations: Annotations) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    annotations.ratings = [r for r in annotations.ratings if r.reviewer_id == "synthetic-a"]
    group = analyze(corpus, plan, recordings(plan), annotations)["groups"][0]
    assert group["labeled_cases"] == 0 and group["mean_absolute_error"] is None
    assert group["false_pass_rate"]["descriptive_wilson_95"] is None


@pytest.mark.parametrize(
    "change", ["prompt", "duplicate", "extra_retry", "missing_retry", "unknown_case", "plan_digest"]
)
def test_recording_integrity(corpus: Corpus, change: str) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    batch = recordings(plan)
    record = batch.records[0]
    if change == "prompt":
        record.calls[0].prompt_digest = "sha256:" + "0" * 64
    elif change == "duplicate":
        batch.records.append(record)
    elif change == "extra_retry":
        record.calls.append(Call(prompt_digest=plan.trials[0].prompts[1].digest, content=response("accuracy", 1.0)))
    elif change == "missing_retry":
        record.calls[0].content = "invalid"
    elif change == "unknown_case":
        record.case_id = "unknown"
    else:
        batch.plan_digest = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        replay(corpus, plan, batch)


@pytest.mark.parametrize("change", ["corpus", "implementation", "trial", "threshold"])
def test_plan_integrity(corpus: Corpus, change: str) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    batch = recordings(plan)
    if change == "corpus":
        corpus.cases[0].agent_text = "Changed trace"
    elif change == "implementation":
        plan.implementation_digest = "sha256:" + "0" * 64
    elif change == "trial":
        plan.trials.pop()
    else:
        plan.specification.threshold = 0.8
    with pytest.raises(ValueError):
        replay(corpus, plan, batch)


def test_no_annotations_has_no_invented_agreement(corpus: Corpus) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    report = analyze(corpus, plan, recordings(plan))
    assert report["annotation_origin"] is None
    assert report["groups"][0]["judge_label_agreement"]["n"] == 0


def test_silent_agent_failure_is_not_excluded_as_missing_reference(corpus: Corpus) -> None:
    corpus.cases[0].agent_text = ""
    corpus.cases[0].conversation = ""
    plan = prepare(corpus, spec(repeats=1))
    assert all(trial.prompts for trial in plan.trials)
    batch = recordings(plan, {("case-0", 0): 0.0})
    rows = replay(corpus, plan, batch)
    assert all(row["status"] == "scored" and row["score"] == 0.0 for row in rows if row["case_id"] == "case-0")


def test_different_response_models_cannot_be_pooled(corpus: Corpus) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=1))
    batch = recordings(plan)
    batch.records[0].calls[0].response_model = "first-model"
    batch.records[1].calls[0].response_model = "second-model"
    with pytest.raises(ValueError, match="mix response model"):
        replay(corpus, plan, batch)


def test_repeatability_does_not_hide_incomplete_cases(corpus: Corpus) -> None:
    plan = prepare(corpus, spec(metrics=["accuracy"], repeats=3))
    batch = recordings(plan)
    batch.records.pop(1)
    group = analyze(corpus, plan, batch)["groups"][0]
    assert group["repeatability"]["complete_cases"] == 3
    assert group["repeatability"]["excluded_cases"] == 1
