# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from skillevaluator.judge_validation import schema
from skillevaluator.judge_validation.schema import (
    Adjudication,
    Annotations,
    Corpus,
    Rating,
    Specification,
    digest,
    load,
    save,
)


@pytest.mark.parametrize("score", [True, False, "0.5", float("nan"), float("inf"), -0.1, 1.1])
def test_scores_reject_ambiguous_values(score) -> None:
    with pytest.raises(ValidationError):
        Rating(case_id="a", metric="accuracy", reviewer_id="b", score=score, rationale="label")


@pytest.mark.parametrize("field", ["skill_id", "family_id", "task_id"])
def test_related_cases_cannot_cross_heldout_boundary(corpus: Corpus, field: str) -> None:
    data = corpus.model_dump()
    case = data["cases"][1]
    case.update(skill_id="new-skill", family_id="new-family", task_id="new-task", agent="other-agent", split="heldout")
    case[field] = data["cases"][0][field]
    with pytest.raises(ValidationError, match="crosses"):
        Corpus.model_validate(data)


def test_identical_trace_cannot_be_relabelled_into_holdout(corpus: Corpus) -> None:
    data = corpus.model_dump()
    case = dict(data["cases"][0])
    case.update(id="copied", skill_id="copy-skill", family_id="copy-family", task_id="copy-task", split="heldout")
    data["cases"].append(case)
    with pytest.raises(ValidationError, match="identical trace"):
        Corpus.model_validate(data)


def test_duplicate_and_unknown_fields_rejected(corpus: Corpus) -> None:
    data = corpus.model_dump()
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValidationError, match="duplicate case"):
        Corpus.model_validate(data)
    with pytest.raises(ValidationError, match="Extra inputs"):
        Corpus.model_validate({**corpus.model_dump(), "human_validated": True})


def test_annotation_identity_and_reviewers(corpus: Corpus, annotations: Annotations) -> None:
    annotations.check_corpus(corpus)
    annotations.ratings.append(annotations.ratings[0])
    with pytest.raises(ValueError, match="duplicate reviewer"):
        annotations.check_corpus(corpus)
    annotations.ratings.pop()
    annotations.ratings[0].case_id = "unknown"
    with pytest.raises(ValueError, match="unknown case"):
        annotations.check_corpus(corpus)
    annotations.corpus_digest = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest"):
        annotations.check_corpus(corpus)


def test_adjudication_requires_independent_initial_ratings(corpus: Corpus, annotations: Annotations) -> None:
    annotations.ratings = annotations.ratings[:1]
    annotations.adjudications = [
        Adjudication(
            case_id="case-0",
            metric="accuracy",
            reviewer_id="adjudicator",
            score=1.0,
            rationale="Reviewed artifact",
        )
    ]
    with pytest.raises(ValueError, match="two distinct"):
        annotations.check_corpus(corpus)


def test_digest_stable_across_json_formatting(corpus: Corpus, tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus.model_dump(), indent=3))
    assert digest(load(path, Corpus)) == digest(corpus)
    corpus.cases[0].ground_truth = "different"
    assert digest(load(path, Corpus)) != digest(corpus)


def test_file_boundaries(corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "corpus.json"
    save(path, corpus)
    with pytest.raises(FileExistsError):
        save(path, corpus)
    assert load(path, Corpus) == corpus
    link = tmp_path / "link.json"
    link.symlink_to(path)
    if hasattr(os, "O_NOFOLLOW"):
        with pytest.raises((OSError, ValueError)):
            load(link, Corpus)
    monkeypatch.setattr(schema, "MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="bounded regular"):
        load(path, Corpus)
    with pytest.raises(ValueError, match="output exceeds"):
        save(tmp_path / "oversized.json", corpus)
    assert not (tmp_path / "oversized.json").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        load(path, Corpus)


def test_duplicate_json_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_text('{"id":"one","id":"two"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load(path, Corpus)


@pytest.mark.parametrize(
    "kwargs", [{"repeats": True}, {"repeats": 0}, {"repeats": 21}, {"metrics": ["accuracy", "accuracy"]}]
)
def test_experiment_limits(kwargs) -> None:
    with pytest.raises(ValidationError):
        Specification(model="test", model_revision="test", **kwargs)
