# SPDX-License-Identifier: Apache-2.0
"""Replay the real judge prompts, parsers, retries, and failure semantics locally."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from skillevaluator import __version__
from skillevaluator.judge_validation.schema import (
    MAX_TRIALS,
    Call,
    Case,
    Corpus,
    Metric,
    Plan,
    Prompt,
    Recording,
    Recordings,
    Specification,
    Trial,
    digest,
)
from skillevaluator.tier3.eval_core import llm_judge

Caller = Callable[..., tuple[str, str | None]]


def implementation_digest() -> str:
    """Include imported scoring/statistics helpers, not just the direct caller."""
    package = Path(__file__).resolve().parent.parent
    entries = [
        (path.relative_to(package).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(package.rglob("*.py"))
    ]
    return digest(entries)


def invoke(case: Case, metric: Metric, caller: Caller, spec: Specification) -> dict[str, Any]:
    kwargs = spec.settings.model_dump(exclude_none=True)
    if metric == "accuracy":
        return llm_judge.judge_accuracy(case.question, case.ground_truth, case.agent_text, caller=caller, **kwargs)
    if metric == "goal_accuracy":
        return llm_judge.judge_goal_accuracy(
            case.question, case.ground_truth, case.agent_text, case.tool_summary, caller=caller, **kwargs
        )
    return llm_judge.judge_behavior_check(case.conversation, case.expected_behaviors, caller=caller, **kwargs)


def _trials(corpus: Corpus, spec: Specification) -> list[Trial]:
    cases = [case for case in corpus.cases if spec.split in {"all", case.split}]
    count = len(cases) * len(spec.metrics) * spec.repeats
    if not 0 < count <= MAX_TRIALS:
        raise ValueError(f"selected corpus must produce 1..{MAX_TRIALS} trials")
    trials = []
    for case in cases:
        for metric in spec.metrics:
            prompts: list[Prompt] = []

            def capture(prompt: str, _prompts: list[Prompt] = prompts, **_kwargs: Any) -> tuple[str, None]:
                _prompts.append(Prompt(text=prompt, digest=digest(prompt)))
                return "", None

            if case.eligible(metric):
                invoke(case, metric, capture, spec)
            for repetition in range(spec.repeats):
                trials.append(Trial(case_id=case.id, metric=metric, repetition=repetition, prompts=prompts))
    return trials


def prepare(corpus: Corpus, spec: Specification) -> Plan:
    root = Path(__file__).resolve().parents[3]
    revision = None
    dirty = None
    if (root / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            revision = None
            dirty = None
    return Plan(
        corpus_digest=digest(corpus),
        implementation_digest=implementation_digest(),
        evaluator_version=__version__,
        evaluator_revision=revision,
        worktree_dirty=dirty,
        specification=spec,
        trials=_trials(corpus, spec),
    )


def check_plan(corpus: Corpus, plan: Plan) -> None:
    if plan.corpus_digest != digest(corpus):
        raise ValueError("plan does not match corpus digest")
    if plan.implementation_digest != implementation_digest():
        raise ValueError("judge-validation implementation changed; restore its revision or prepare a new experiment")
    if plan.trials != _trials(corpus, plan.specification):
        raise ValueError("planned cases, prompts, or repetitions do not match the corpus and specification")


def replay(corpus: Corpus, plan: Plan, batch: Recordings) -> list[dict[str, Any]]:
    check_plan(corpus, plan)
    if batch.plan_digest != digest(plan):
        raise ValueError("recordings do not match plan digest")
    cases = {case.id: case for case in corpus.cases}
    expected = {trial.key: trial for trial in plan.trials}
    records: dict[tuple[str, str, int], Recording] = {}
    for record in batch.records:
        if record.key not in expected or record.key in records or not expected[record.key].prompts:
            raise ValueError("recording is duplicate, unplanned, or refers to missing reference evidence")
        records[record.key] = record
    response_models = {call.response_model for record in batch.records for call in record.calls if call.response_model}
    if len(response_models) > 1:
        raise ValueError("recordings mix response model identities; use separate experiments per model")
    rows = []
    for trial in plan.trials:
        case = cases[trial.case_id]
        row: dict[str, Any] = {
            "case_id": case.id,
            "skill_id": case.skill_id,
            "family_id": case.family_id,
            "task_id": case.task_id,
            "split": case.split,
            "condition": case.condition,
            "agent": case.agent,
            "metric": trial.metric,
            "repetition": trial.repetition,
            "score": None,
        }
        if not trial.prompts:
            rows.append({**row, "status": "missing_reference"})
            continue
        if trial.key not in records:
            rows.append({**row, "status": "missing_recording"})
            continue
        record = records[trial.key]
        consumed = 0

        def recorded_call(prompt: str, _record: Recording = record, **_kwargs: Any) -> tuple[str, str | None]:
            nonlocal consumed
            if consumed >= len(_record.calls):
                raise ValueError("recording is missing a required retry response")
            call = _record.calls[consumed]
            consumed += 1
            if call.prompt_digest != digest(prompt):
                raise ValueError("recorded response does not match the exact judge prompt")
            return call.content, call.error

        result = invoke(case, trial.metric, recorded_call, plan.specification)
        if consumed != len(record.calls):
            raise ValueError("recording contains unused responses")
        transport_errors = sum(call.error is not None for call in record.calls)
        if result.get("status") == "error":
            status = "transport_error" if transport_errors else "judge_error"
        else:
            status = "scored"
        rows.append(
            {
                **row,
                "status": status,
                "score": result.get("score"),
                "result": result,
                "call_count": consumed,
                "transport_error_calls": transport_errors,
                "elapsed_ms": sum(call.elapsed_ms for call in record.calls),
                "response_models": sorted({call.response_model for call in record.calls if call.response_model}),
            }
        )
    return rows


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req: Any, _fp: Any, _code: int, _msg: str, _headers: Any, _newurl: str) -> None:
        raise ValueError("local judge redirects are forbidden")


def validate_endpoint(endpoint: str) -> str:
    if any(ord(char) <= 32 or ord(char) == 127 for char in endpoint):
        raise ValueError("endpoint contains whitespace or control characters")
    try:
        parsed = urlsplit(endpoint)
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint must use a literal loopback IP address") from exc
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "%" in endpoint
        or parsed.path not in {"/v1/chat/completions", "/chat/completions"}
    ):
        raise ValueError(
            "use http://127.0.0.1:PORT/v1/chat/completions (literal loopback, no credentials or redirects)"
        )
    return endpoint


def run_local(corpus: Corpus, plan: Plan, endpoint: str, *, timeout: float = 60.0) -> Recordings:
    """Only contact a literal loopback endpoint; inherited proxies/keys are ignored."""
    endpoint = validate_endpoint(endpoint)
    if not 0 < timeout <= 120:
        raise ValueError("timeout must be in (0, 120] seconds")
    check_plan(corpus, plan)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    cases = {case.id: case for case in corpus.cases}
    records: list[Recording] = []
    for trial in plan.trials:
        if not trial.prompts:
            continue
        calls: list[Call] = []

        def local_call(prompt: str, _calls: list[Call] = calls, **kwargs: Any) -> tuple[str, str | None]:
            payload = {"model": plan.specification.model, "messages": [{"role": "user", "content": prompt}], **kwargs}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload, allow_nan=False).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start = time.monotonic()
            content = ""
            error = None
            response_model = None
            try:
                with opener.open(request, timeout=timeout) as response:
                    raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise ValueError("local judge response exceeds byte limit")
                value = json.loads(raw)
                content = value["choices"][0]["message"]["content"]
                if not isinstance(content, str) or len(content) > 100_000:
                    raise ValueError("local judge must return bounded text content")
                response_model = value.get("model")
                if response_model is not None and (
                    not isinstance(response_model, str) or not response_model.strip() or len(response_model) > 4000
                ):
                    raise ValueError("invalid response model identity")
            except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError, RecursionError) as exc:
                # Do not retain arbitrary server error bodies or environment credentials.
                content = ""
                error = f"Local judge request failed ({type(exc).__name__})"
                response_model = None
            _calls.append(
                Call(
                    prompt_digest=digest(prompt),
                    content=content,
                    error=error,
                    response_model=response_model,
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )
            )
            return content, error

        invoke(cases[trial.case_id], trial.metric, local_call, plan.specification)
        records.append(Recording(case_id=trial.case_id, metric=trial.metric, repetition=trial.repetition, calls=calls))
    return Recordings(
        plan_digest=digest(plan),
        origin="local_model",
        producer=endpoint,
        recorded_at=datetime.now(UTC).isoformat(),
        records=records,
    )
