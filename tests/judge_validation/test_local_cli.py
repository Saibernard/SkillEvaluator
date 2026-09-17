# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.judge_validation.analysis import analyze
from skillevaluator.judge_validation.runner import prepare, run_local, validate_endpoint
from skillevaluator.judge_validation.schema import Annotations, Corpus, Plan, Recordings, Specification, load, save

from .conftest import recordings


@contextmanager
def local_server(*, redirect: bool = False, malformed: bool = False, fault: str | None = None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((dict(self.headers), json.loads(body)))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "https://example.invalid/must-not-be-contacted")
                self.end_headers()
                return
            if fault == "broken-chunk":
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"invalid-chunk-size\r\n")
                return
            response = {
                "model": "local-fixture",
                "choices": [{"message": {"content": "invalid" if malformed else '{"score":0.0}'}}],
            }
            content = json.dumps(response).encode()
            if fault == "deep-json":
                content = b"[" * 20_000 + b"]" * 20_000
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1/chat/completions",
        "http://example.com:80/v1/chat/completions",
        "http://localhost:8000/v1/chat/completions",
        "http://127.0.0.1/v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions?key=secret",
        "http://user:secret@127.0.0.1:8000/v1/chat/completions",
        " http://127.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8000/other",
        "http://2130706433:8000/v1/chat/completions",
        "http://10.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8000/v1/chat/completions#fragment",
        "http://127.0.0.1:99999/v1/chat/completions",
    ],
)
def test_nonlocal_or_ambiguous_endpoints_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


def test_literal_ipv6_loopback_allowed() -> None:
    endpoint = "http://[::1]:8000/v1/chat/completions"
    assert validate_endpoint(endpoint) == endpoint


def test_help_places_judge_validation_in_evaluation_analysis() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    section = result.output.split("Evaluation analysis:", 1)[1].split("Expert aliases:", 1)[0]
    assert "judge-validation" in section
    assert "Other commands:" not in result.output


def test_local_transport_bypasses_proxies_and_credentials(corpus: Corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9999")
    monkeypatch.setenv("NO_PROXY", "")
    plan = prepare(corpus, Specification(model="local-fixture", model_revision="test", metrics=["accuracy"], repeats=1))
    with local_server() as (endpoint, requests):
        batch = run_local(corpus, plan, endpoint, timeout=1)
    assert len(requests) == 4 and batch.origin == "local_model"
    for headers, body in requests:
        assert not any(key.lower() in {"authorization", "proxy-authorization"} for key in headers)
        assert body["model"] == "local-fixture" and body["max_tokens"] == 4096
        assert "hosted-secret" not in json.dumps(body)
    report = analyze(corpus, plan, batch)
    assert all(row["score"] == 0.0 and row["status"] == "scored" for row in report["observations"])
    assert report["response_models"] == ["local-fixture"]


def test_redirect_is_retained_as_failure_without_following(corpus: Corpus) -> None:
    plan = prepare(corpus, Specification(model="local-fixture", model_revision="test", metrics=["accuracy"], repeats=1))
    with local_server(redirect=True) as (endpoint, requests):
        batch = run_local(corpus, plan, endpoint, timeout=1)
    assert len(requests) == 4
    assert all(call.error for record in batch.records for call in record.calls)
    assert analyze(corpus, plan, batch)["groups"][0]["statuses"] == {"transport_error": 4}


@pytest.mark.parametrize("fault", ["broken-chunk", "deep-json"])
def test_invalid_server_response_preserves_batch(corpus: Corpus, tmp_path: Path, fault: str) -> None:
    corpus_path, plan_path, batch_path = tmp_path / "corpus.json", tmp_path / "plan.json", tmp_path / "batch.json"
    save(corpus_path, corpus)
    plan = prepare(corpus, Specification(model="fixture", model_revision="1", metrics=["accuracy"], repeats=1))
    save(plan_path, plan)
    with local_server(fault=fault) as (endpoint, requests):
        result = CliRunner().invoke(
            cli,
            [
                "judge-validation",
                "run-local",
                str(corpus_path),
                str(plan_path),
                "--endpoint",
                endpoint,
                "--out",
                str(batch_path),
            ],
        )
    assert result.exit_code == 1
    assert batch_path.exists(), repr(result.exception)
    batch = load(batch_path, Recordings)
    assert len(requests) == len(batch.records) == 4
    report = analyze(corpus, plan, batch)
    assert all(row["status"] == "transport_error" and row["score"] is None for row in report["observations"])


def test_malformed_model_replies_retain_both_attempts(corpus: Corpus) -> None:
    plan = prepare(corpus, Specification(model="local-fixture", model_revision="test", metrics=["accuracy"], repeats=1))
    with local_server(malformed=True) as (endpoint, requests):
        batch = run_local(corpus, plan, endpoint, timeout=1)
    assert len(requests) == 8
    assert all(len(record.calls) == 2 for record in batch.records)
    assert analyze(corpus, plan, batch)["groups"][0]["statuses"] == {"judge_error": 4}


def test_offline_cli_end_to_end(corpus: Corpus, annotations: Annotations, tmp_path: Path) -> None:
    corpus_path, labels_path = tmp_path / "corpus.json", tmp_path / "labels.json"
    plan_path, batch_path = tmp_path / "plan.json", tmp_path / "recordings.json"
    report_path, md_path = tmp_path / "report.json", tmp_path / "report.md"
    save(corpus_path, corpus)
    save(labels_path, annotations)
    runner = CliRunner()
    result = runner.invoke(cli, ["judge-validation", "validate", str(corpus_path), "--annotations", str(labels_path)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli,
        [
            "judge-validation",
            "prepare",
            str(corpus_path),
            "--model",
            "fixture",
            "--model-revision",
            "1",
            "--out",
            str(plan_path),
        ],
    )
    assert result.exit_code == 0, result.output
    plan = load(plan_path, Plan)
    save(batch_path, recordings(plan))
    args = [
        "judge-validation",
        "replay",
        str(corpus_path),
        str(plan_path),
        str(batch_path),
        "--annotations",
        str(labels_path),
        "--out",
        str(report_path),
        "--markdown",
        str(md_path),
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert len(report["observations"]) == 36 and len(report["groups"]) == 3
    assert "synthetic_demonstration" in md_path.read_text()
    original = report_path.read_bytes()
    assert runner.invoke(cli, args).exit_code == 1
    assert report_path.read_bytes() == original


def test_blank_annotation_form_cannot_claim_completed_review(corpus: Corpus, tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    save(corpus_path, corpus)
    runner = CliRunner()
    paths = [tmp_path / "reviewer1.json", tmp_path / "reviewer2.json"]
    for index, path in enumerate(paths):
        result = runner.invoke(
            cli,
            [
                "judge-validation",
                "annotate",
                str(corpus_path),
                "--reviewer",
                f"reviewer-{index}",
                "--rubric-version",
                "1",
                "--out",
                str(path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert all(r.score is None for r in load(path, Annotations).ratings)
    merged_path = tmp_path / "merged.json"
    result = runner.invoke(
        cli, ["judge-validation", "merge-annotations", str(corpus_path), *map(str, paths), "--out", str(merged_path)]
    )
    assert result.exit_code == 0, result.output
    plan = prepare(corpus, Specification(model="fixture", model_revision="1", repeats=1))
    report = analyze(corpus, plan, recordings(plan), load(merged_path, Annotations))
    assert all(group["labeled_cases"] == 0 for group in report["groups"])


def test_call_budget_fails_before_contact(corpus: Corpus, tmp_path: Path) -> None:
    corpus_path, plan_path, batch_path = tmp_path / "corpus.json", tmp_path / "plan.json", tmp_path / "batch.json"
    save(corpus_path, corpus)
    save(plan_path, prepare(corpus, Specification(model="fixture", model_revision="1")))
    with local_server() as (endpoint, requests):
        result = CliRunner().invoke(
            cli,
            [
                "judge-validation",
                "run-local",
                str(corpus_path),
                str(plan_path),
                "--endpoint",
                endpoint,
                "--max-calls",
                "1",
                "--out",
                str(batch_path),
            ],
        )
    assert result.exit_code == 1 and "maximum call budget" in result.output
    assert not requests and not batch_path.exists()


def test_cli_retains_transport_errors_and_exits_nonzero(corpus: Corpus, tmp_path: Path) -> None:
    corpus_path, plan_path, batch_path = tmp_path / "corpus.json", tmp_path / "plan.json", tmp_path / "batch.json"
    save(corpus_path, corpus)
    save(
        plan_path, prepare(corpus, Specification(model="fixture", model_revision="1", metrics=["accuracy"], repeats=1))
    )
    with local_server(redirect=True) as (endpoint, _requests):
        result = CliRunner().invoke(
            cli,
            [
                "judge-validation",
                "run-local",
                str(corpus_path),
                str(plan_path),
                "--endpoint",
                endpoint,
                "--out",
                str(batch_path),
            ],
        )
    assert result.exit_code == 1
    assert len(load(batch_path, Recordings).records) == 4
