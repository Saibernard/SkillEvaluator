# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 synthetic dataset creation and live agent evaluation."""

from typing import Any

__all__ = [
    "compare_results",
    "create_dataset",
    "doctor",
    "evaluate",
    "validate_evals",
    "view_results",
]


def __getattr__(name: str) -> Any:
    # Pure judge replay must remain usable without Harbor or provider extras.
    # Preserve the public convenience exports while loading live runtimes only
    # when an exported command is actually requested.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from skillevaluator.tier3 import commands

    return getattr(commands, name)
