"""Exceptions shared by the stages, the trainer and the job system (PLAN.md §7.3-7.4)."""

from __future__ import annotations


class Cancelled(Exception):
    """The job's cancel flag was seen at a safe point; the step stopped cleanly (step status `cancelled`)."""


class InsufficientData(ValueError):
    """Not enough real or synthetic training data to fill an arm's equal token budget. The message says what to
    change (data.pool_size / data.pool_max or data.token_budget)."""


class NoSignificantWeakness(RuntimeError):
    """`target: auto` but diagnosis found no significant weakness (step error `no_significant_weakness`)."""
