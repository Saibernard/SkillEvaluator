# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pytest

from skillevaluator.judge_validation.schema import (
    Annotations,
    Call,
    Corpus,
    Plan,
    Rating,
    Recording,
    Recordings,
    digest,
)
from skillevaluator.tier3.eval_core import llm_judge


@pytest.fixture(autouse=True)
def forbid_hosted_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("offline validation attempted to invoke the hosted provider")

    monkeypatch.setattr(llm_judge, "call_public_llm", fail)
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-secret-must-not-be-used")
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.model_validate(
        {
            "id": "synthetic-test",
            "version": "1",
            "license": "Apache-2.0",
            "description": "Synthetic harness tests",
            "cases": [
                {
                    "id": f"case-{i}",
                    "skill_id": "calculator",
                    "family_id": "addition",
                    "task_id": f"task-{i}",
                    "agent": "scripted",
                    "condition": "with_skill",
                    "split": "development",
                    "source": {"kind": "synthetic", "reference": "test fixture", "revision": "1"},
                    "question": f"Add {i} and 1",
                    "ground_truth": str(i + 1),
                    "agent_text": str(i + 1 if i % 2 == 0 else -10),
                    "conversation": f"tool: add({i}, 1), result: {i + 1 if i % 2 == 0 else -10}",
                    "expected_behaviors": ["Return the correct sum"],
                }
                for i in range(4)
            ],
        }
    )


@pytest.fixture
def annotations(corpus: Corpus) -> Annotations:
    return Annotations(
        corpus_digest=digest(corpus),
        rubric_version="synthetic-1",
        origin="synthetic",
        ratings=[
            Rating(
                case_id=case.id,
                metric=metric,
                reviewer_id=reviewer,
                score=float(i % 2 == 0),
                rationale="Synthetic oracle",
            )
            for i, case in enumerate(corpus.cases)
            for metric in ("accuracy", "goal_accuracy", "behavior_check")
            for reviewer in ("synthetic-a", "synthetic-b")
        ],
    )


def response(metric: str, score: float) -> str:
    if metric == "accuracy":
        return json.dumps({"score": score, "reason": "Synthetic response"})
    if metric == "goal_accuracy":
        return json.dumps({"score": score, "achieved": score >= 0.5, "reason": "Synthetic response"})
    return json.dumps({"results": [{"step": 1, "passed": score >= 0.5}], "summary": "Synthetic response"})


def recordings(plan: Plan, scores: dict[tuple[str, int], float] | None = None) -> Recordings:
    scores = scores or {}
    return Recordings(
        plan_digest=digest(plan),
        origin="synthetic",
        producer="test fixture",
        recorded_at="synthetic",
        records=[
            Recording(
                case_id=trial.case_id,
                metric=trial.metric,
                repetition=trial.repetition,
                calls=[
                    Call(
                        prompt_digest=trial.prompts[0].digest,
                        content=response(trial.metric, scores.get((trial.case_id, trial.repetition), 1.0)),
                    )
                ],
            )
            for trial in plan.trials
            if trial.prompts
        ],
    )
