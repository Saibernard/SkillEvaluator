# SPDX-License-Identifier: Apache-2.0
"""Versioned corpus, annotation, and recording contracts. No implicit coercion."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")]
Text = Annotated[str, StringConstraints(max_length=100_000)]
Nonempty = Annotated[str, StringConstraints(min_length=1, max_length=4_000, strip_whitespace=True)]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Score = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
Metric = Literal["accuracy", "goal_accuracy", "behavior_check"]
Split = Literal["development", "heldout"]
METRICS: tuple[Metric, ...] = ("accuracy", "goal_accuracy", "behavior_check")
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TRIALS = 20_000


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)


class Source(Contract):
    kind: Literal["synthetic", "captured"]
    reference: Nonempty
    revision: Nonempty


class Case(Contract):
    id: Identifier
    skill_id: Identifier
    family_id: Identifier
    task_id: Identifier
    agent: Nonempty
    condition: Literal["with_skill", "without_skill"]
    split: Split
    source: Source
    question: Nonempty
    ground_truth: Text
    agent_text: Text
    conversation: Text
    tool_summary: Text = ""
    expected_behaviors: list[Nonempty] = Field(default_factory=list, max_length=100)

    def eligible(self, metric: Metric) -> bool:
        if metric == "behavior_check":
            return bool(self.expected_behaviors)
        # An empty agent response is an observable outcome, not absent reference
        # evidence. Preserve it so silent task failures are not selected out.
        return bool(self.ground_truth.strip())


class Corpus(Contract):
    schema_version: Literal["judge-corpus/1"] = "judge-corpus/1"
    id: Identifier
    version: Nonempty
    license: Nonempty
    description: Nonempty
    cases: list[Case] = Field(min_length=1, max_length=5_000)

    @model_validator(mode="after")
    def check_partitions(self) -> Self:
        ids: set[str] = set()
        logical_cases: set[tuple[str, str, str]] = set()
        groups: dict[tuple[str, str], str] = {}
        traces: dict[str, str] = {}
        for case in self.cases:
            if case.id in ids:
                raise ValueError(f"duplicate case id: {case.id}")
            ids.add(case.id)
            logical = (case.task_id, case.agent, case.condition)
            if logical in logical_cases:
                raise ValueError("duplicate task/agent/condition; judge repeats belong in the experiment plan")
            logical_cases.add(logical)
            for field in ("skill_id", "family_id", "task_id"):
                group = (field, getattr(case, field))
                if groups.setdefault(group, case.split) != case.split:
                    raise ValueError(f"{field} crosses development/heldout partitions: {group[1]}")
            trace = digest([case.question, case.agent_text, case.conversation, case.tool_summary])
            if traces.setdefault(trace, case.split) != case.split:
                raise ValueError("identical trace crosses development/heldout partitions")
        return self


class Rating(Contract):
    case_id: Identifier
    metric: Metric
    reviewer_id: Identifier
    score: Score | None
    rationale: Nonempty


class Adjudication(Contract):
    case_id: Identifier
    metric: Metric
    reviewer_id: Identifier
    score: Score
    rationale: Nonempty


class Annotations(Contract):
    schema_version: Literal["judge-annotations/1"] = "judge-annotations/1"
    corpus_digest: Digest
    rubric_version: Nonempty
    origin: Literal["human", "synthetic"]
    ratings: list[Rating] = Field(max_length=60_000)
    adjudications: list[Adjudication] = Field(default_factory=list, max_length=15_000)

    def check_corpus(self, corpus: Corpus) -> None:
        if self.corpus_digest != digest(corpus):
            raise ValueError("annotations do not match this corpus digest")
        cases = {c.id: c for c in corpus.cases}
        seen: set[tuple[str, str, str]] = set()
        reviewers: dict[tuple[str, str], set[str]] = {}
        for rating in self.ratings:
            key = (rating.case_id, rating.metric, rating.reviewer_id)
            if rating.case_id not in cases or key in seen:
                raise ValueError("unknown case or duplicate reviewer rating")
            if not cases[rating.case_id].eligible(rating.metric):
                raise ValueError("rating refers to missing judge evidence/reference")
            seen.add(key)
            reviewers.setdefault(key[:2], set()).add(rating.reviewer_id)
        if len({rating.reviewer_id for rating in self.ratings}) > 32 or any(len(ids) > 8 for ids in reviewers.values()):
            raise ValueError("annotation limit: 32 reviewers overall and 8 per case/metric")
        resolved: set[tuple[str, str]] = set()
        for resolution in self.adjudications:
            key = (resolution.case_id, resolution.metric)
            if key in resolved or len(reviewers.get(key, set())) < 2:
                raise ValueError("adjudication needs two distinct initial reviewers and a unique case/metric")
            resolved.add(key)


class Settings(Contract):
    max_tokens: int = Field(default=4096, ge=1, le=16_384)
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None


class Specification(Contract):
    model: Nonempty
    model_revision: Nonempty
    repeats: int = Field(default=3, ge=1, le=20)
    split: Literal["all", "development", "heldout"] = "all"
    threshold: Score = 0.5
    metrics: list[Metric] = Field(default_factory=lambda: list(METRICS), min_length=1, max_length=3)
    settings: Settings = Field(default_factory=Settings)

    @model_validator(mode="after")
    def unique_metrics(self) -> Self:
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be unique")
        return self


class Prompt(Contract):
    text: Text
    digest: Digest


class Trial(Contract):
    case_id: Identifier
    metric: Metric
    repetition: int = Field(ge=0, lt=20)
    prompts: list[Prompt] = Field(max_length=2)

    @property
    def key(self) -> tuple[str, str, int]:
        return self.case_id, self.metric, self.repetition


class Plan(Contract):
    schema_version: Literal["judge-plan/1"] = "judge-plan/1"
    corpus_digest: Digest
    implementation_digest: Digest
    evaluator_version: Nonempty
    evaluator_revision: Nonempty | None = None
    worktree_dirty: bool | None = None
    specification: Specification
    trials: list[Trial] = Field(min_length=1, max_length=MAX_TRIALS)


class Call(Contract):
    prompt_digest: Digest
    content: Text = ""
    error: Annotated[str, StringConstraints(min_length=1, max_length=4_000)] | None = None
    elapsed_ms: Annotated[float, Field(ge=0, le=3_600_000, allow_inf_nan=False)] = 0.0
    response_model: Nonempty | None = None

    @model_validator(mode="after")
    def unambiguous_response(self) -> Self:
        if self.error is not None and self.content:
            raise ValueError("a failed call cannot also contain a scored response")
        return self


class Recording(Contract):
    case_id: Identifier
    metric: Metric
    repetition: int = Field(ge=0, lt=20)
    calls: list[Call] = Field(min_length=1, max_length=2)

    @property
    def key(self) -> tuple[str, str, int]:
        return self.case_id, self.metric, self.repetition


class Recordings(Contract):
    schema_version: Literal["judge-recordings/1"] = "judge-recordings/1"
    plan_digest: Digest
    origin: Literal["synthetic", "local_model", "imported"]
    producer: Nonempty
    recorded_at: Nonempty
    records: list[Recording] = Field(max_length=MAX_TRIALS)


def digest(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def load[T: Contract](path: Path, model: type[T]) -> T:
    # A FIFO/device must not block a read. O_NONBLOCK also closes the stat/open race.
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("input must be a regular JSON file, not a link or device")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("input must be a bounded regular JSON file")
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("JSON input exceeds size limit")
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_object)
        return model.model_validate(obj)
    except RecursionError as exc:
        raise ValueError("JSON input exceeds nesting limit") from exc


def save(path: Path, value: object) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError("output exceeds size limit; use a smaller corpus or fewer repeats")
    # Exclusive creation protects source data and earlier experiments from overwrite.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
