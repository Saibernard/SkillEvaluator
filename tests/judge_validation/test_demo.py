# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.judge_validation.analysis import analyze
from skillevaluator.judge_validation.demo import demo_corpus
from skillevaluator.judge_validation.schema import Annotations, Corpus, Plan, Recordings, load


def test_demo_labels_come_from_observed_output_equality() -> None:
    corpus, annotations = demo_corpus()
    annotations.check_corpus(corpus)
    cases = {case.id: case for case in corpus.cases}
    assert len(cases) == 60
    assert len({case.skill_id for case in corpus.cases}) == 6
    assert Counter(case.split for case in corpus.cases) == {"development": 40, "heldout": 20}
    assert annotations.origin == "synthetic"
    for label in annotations.ratings:
        case = cases[label.case_id]
        assert label.score == float(case.agent_text == case.ground_truth)
        assert label.reviewer_id.startswith("synthetic-")


def test_demo_is_fully_replayable_and_keeps_known_failures(tmp_path: Path) -> None:
    destination = tmp_path / "demonstration"
    result = CliRunner().invoke(cli, ["judge-validation", "demo", "--out-dir", str(destination)])
    assert result.exit_code == 0, result.output
    corpus = load(destination / "corpus.json", Corpus)
    plan = load(destination / "plan.json", Plan)
    recordings = load(destination / "recordings.json", Recordings)
    annotations = load(destination / "annotations.json", Annotations)
    report = json.loads((destination / "report.json").read_text())
    assert report == analyze(corpus, plan, recordings, annotations)
    statuses = Counter(row["status"] for row in report["observations"])
    assert statuses == {"scored": 459, "missing_recording": 27, "transport_error": 27, "judge_error": 27}
    assert len(report["observations"]) == 540
    assert len(report["groups"]) == 12
    assert any(group["false_pass_rate"]["numerator"] > 0 for group in report["groups"])
    assert any(group["false_fail_rate"]["numerator"] > 0 for group in report["groups"])
    assert any(group["repeatability"]["decision_flip_rate"]["numerator"] > 0 for group in report["groups"])
    assert report["evidence_status"] == "synthetic_demonstration"
    assert CliRunner().invoke(cli, ["judge-validation", "demo", "--out-dir", str(destination)]).exit_code == 1
