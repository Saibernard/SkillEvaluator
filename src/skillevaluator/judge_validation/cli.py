# SPDX-License-Identifier: Apache-2.0
"""Small CLI surface; judge and analysis implementations load only when invoked."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from skillevaluator.judge_validation.schema import (
    Annotations,
    Corpus,
    Plan,
    Rating,
    Recordings,
    Settings,
    Specification,
    digest,
    load,
    save,
)

INPUT = click.Path(exists=True, dir_okay=False, path_type=Path)
OUTPUT = click.Path(dir_okay=False, path_type=Path)


def _errors(function: Any) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except ValidationError as exc:
            details = exc.errors(include_input=False, include_url=False)
            summary = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in details[:5])
            raise click.ClickException(summary) from exc
        except (OSError, ValueError, UnicodeError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


def _new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists: {path}; choose a new experiment path")
    if not path.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {path.parent}")


@click.group("judge-validation")
def judge_validation() -> None:
    """Measure judge agreement and repeatability using local evidence."""


@judge_validation.command("demo")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@_errors
def demo(out_dir: Path) -> None:
    """Generate 60 synthetic cases and a fault-injection report without model calls."""
    from skillevaluator.judge_validation.demo import write_demo

    click.echo(json.dumps(write_demo(out_dir)))


@judge_validation.command("validate")
@click.argument("corpus_path", type=INPUT)
@click.option("--annotations", "annotation_path", type=INPUT)
@_errors
def validate(corpus_path: Path, annotation_path: Path | None) -> None:
    """Check corpus partitions, identities and optional independent annotations."""
    corpus = load(corpus_path, Corpus)
    if annotation_path:
        load(annotation_path, Annotations).check_corpus(corpus)
    click.echo(json.dumps({"corpus_digest": digest(corpus), "cases": len(corpus.cases), "valid": True}))


@judge_validation.command("annotate")
@click.argument("corpus_path", type=INPUT)
@click.option("--reviewer", required=True)
@click.option("--rubric-version", required=True)
@click.option("--out", type=OUTPUT, required=True)
@_errors
def annotate(corpus_path: Path, reviewer: str, rubric_version: str, out: Path) -> None:
    """Create an unfilled human annotation form without showing other reviewers' labels."""
    _new_output(out)
    corpus = load(corpus_path, Corpus)
    ratings = [
        Rating(
            case_id=case.id,
            metric=metric,
            reviewer_id=reviewer,
            score=None,
            rationale="Not reviewed. Replace this text with evidence supporting the score or uncertainty.",
        )
        for case in corpus.cases
        for metric in ("accuracy", "goal_accuracy", "behavior_check")
        if case.eligible(metric)
    ]
    save(out, Annotations(corpus_digest=digest(corpus), rubric_version=rubric_version, origin="human", ratings=ratings))
    click.echo(f"Unfilled annotation form written to {out}")


@judge_validation.command("merge-annotations")
@click.argument("corpus_path", type=INPUT)
@click.argument("annotation_paths", type=INPUT, nargs=-1, required=True)
@click.option("--out", type=OUTPUT, required=True)
@_errors
def merge_annotations(corpus_path: Path, annotation_paths: tuple[Path, ...], out: Path) -> None:
    """Combine independent forms; preserve uncertainty and disagreement."""
    _new_output(out)
    corpus = load(corpus_path, Corpus)
    forms = [load(path, Annotations) for path in annotation_paths]
    for form in forms:
        form.check_corpus(corpus)
    if len({(form.rubric_version, form.origin) for form in forms}) != 1:
        raise ValueError("annotation forms must use the same rubric and origin")
    merged = Annotations(
        corpus_digest=digest(corpus),
        rubric_version=forms[0].rubric_version,
        origin=forms[0].origin,
        ratings=[rating for form in forms for rating in form.ratings],
        adjudications=[item for form in forms for item in form.adjudications],
    )
    merged.check_corpus(corpus)
    save(out, merged)
    click.echo(f"Merged annotations written to {out}")


@judge_validation.command("prepare")
@click.argument("corpus_path", type=INPUT)
@click.option("--model", required=True)
@click.option("--model-revision", required=True, help="Model digest/version, or explicitly 'unknown'.")
@click.option("--repeats", type=click.IntRange(1, 20), default=3, show_default=True)
@click.option("--split", type=click.Choice(["all", "development", "heldout"]), default="all")
@click.option("--threshold", type=click.FloatRange(0, 1), default=0.5, show_default=True)
@click.option("--max-tokens", type=click.IntRange(1, 16_384), default=4096)
@click.option("--temperature", type=click.FloatRange(0, 2))
@click.option("--out", type=OUTPUT, required=True)
@_errors
def prepare_command(
    corpus_path: Path,
    model: str,
    model_revision: str,
    repeats: int,
    split: str,
    threshold: float,
    max_tokens: int,
    temperature: float | None,
    out: Path,
) -> None:
    """Freeze exact judge prompts and repetitions without making model calls."""
    from skillevaluator.judge_validation.runner import prepare

    _new_output(out)
    corpus = load(corpus_path, Corpus)
    spec = Specification(
        model=model,
        model_revision=model_revision,
        repeats=repeats,
        split=split,
        threshold=threshold,
        settings=Settings(max_tokens=max_tokens, temperature=temperature),
    )
    plan = prepare(corpus, spec)
    save(out, plan)
    click.echo(
        json.dumps(
            {"plan": str(out), "trials": len(plan.trials), "maximum_calls": sum(len(t.prompts) for t in plan.trials)}
        )
    )


@judge_validation.command("run-local")
@click.argument("corpus_path", type=INPUT)
@click.argument("plan_path", type=INPUT)
@click.option("--endpoint", required=True, help="Literal loopback HTTP chat-completions endpoint.")
@click.option("--max-calls", type=click.IntRange(1, 40_000), default=200, show_default=True)
@click.option("--timeout", type=click.FloatRange(min=0, max=120, min_open=True), default=60.0)
@click.option("--out", type=OUTPUT, required=True)
@_errors
def run_local_command(
    corpus_path: Path, plan_path: Path, endpoint: str, max_calls: int, timeout: float, out: Path
) -> None:
    """Record a local model's responses. Hosted endpoints, proxies and redirects are rejected."""
    from skillevaluator.judge_validation.runner import run_local

    _new_output(out)
    corpus, plan = load(corpus_path, Corpus), load(plan_path, Plan)
    if sum(len(trial.prompts) for trial in plan.trials) > max_calls:
        raise ValueError("plan exceeds the maximum call budget, including retries")
    batch = run_local(corpus, plan, endpoint, timeout=timeout)
    save(out, batch)
    errors = sum(call.error is not None for record in batch.records for call in record.calls)
    click.echo(json.dumps({"recordings": str(out), "trials": len(batch.records), "transport_error_calls": errors}))
    if errors:
        raise click.exceptions.Exit(1)


@judge_validation.command("replay")
@click.argument("corpus_path", type=INPUT)
@click.argument("plan_path", type=INPUT)
@click.argument("recordings_path", type=INPUT)
@click.option("--annotations", "annotation_path", type=INPUT)
@click.option("--out", type=OUTPUT, required=True)
@click.option("--markdown", "markdown_path", type=OUTPUT)
@_errors
def replay_command(
    corpus_path: Path,
    plan_path: Path,
    recordings_path: Path,
    annotation_path: Path | None,
    out: Path,
    markdown_path: Path | None,
) -> None:
    """Replay saved responses and report measurements entirely offline."""
    from skillevaluator.judge_validation.analysis import analyze, markdown

    _new_output(out)
    if markdown_path:
        _new_output(markdown_path)
        if out.resolve() == markdown_path.resolve():
            raise ValueError("JSON and Markdown output paths must be distinct")
    corpus, plan, recordings = load(corpus_path, Corpus), load(plan_path, Plan), load(recordings_path, Recordings)
    annotations = load(annotation_path, Annotations) if annotation_path else None
    report = analyze(corpus, plan, recordings, annotations)
    save(out, report)
    if markdown_path:
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(markdown(report))
    click.echo(json.dumps({"report": str(out), "evidence_status": report["evidence_status"]}))
