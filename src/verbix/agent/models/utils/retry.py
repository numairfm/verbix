"""Retry utility for model queries."""

from __future__ import annotations

import logging

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry(*, logger: logging.Logger, abort_exceptions: list[type[BaseException]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying for synchronous model queries."""
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
