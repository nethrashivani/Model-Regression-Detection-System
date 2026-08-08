"""
Retry-with-backoff for API calls, specifically to handle 429 rate limits
gracefully instead of counting a rate-limited case as a real failure.

Free-tier rate limits (Groq's included) are hit routinely under normal use --
a 429 means "try again shortly," not "this case failed." Treating it as a
failure (as the first version of the runner did) silently corrupts the eval's
accuracy numbers with rate-limit noise instead of real model behavior.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 6
DEFAULT_BASE_DELAY = 5.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds

# Connection errors and timeouts are transient network blips, not rate limits --
# they don't come with a Retry-After header, so back off much faster.
CONNECTION_ERROR_BASE_DELAY = 2.0
CONNECTION_ERROR_MAX_DELAY = 15.0


def _extract_retry_after(exc: APIStatusError) -> float | None:
    """Some providers send a Retry-After header telling you exactly how long to wait."""
    try:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        header = response.headers.get("retry-after")
        return float(header) if header is not None else None
    except (ValueError, AttributeError):
        return None


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> T:
    """
    Calls `fn()` (a zero-arg async callable, e.g. a lambda wrapping the real
    call) and retries on RateLimitError with exponential backoff, honoring a
    Retry-After header when the API provides one. Re-raises the last error
    if all retries are exhausted -- callers still need to handle a genuine
    persistent failure.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except RateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = _extract_retry_after(e)
            if wait is None:
                wait = min(base_delay * (2 ** (attempt - 1)), max_delay)
            await asyncio.sleep(wait)
        except (APIConnectionError, APITimeoutError):
            attempt += 1
            if attempt > max_retries:
                raise
            wait = min(CONNECTION_ERROR_BASE_DELAY * (2 ** (attempt - 1)), CONNECTION_ERROR_MAX_DELAY)
            await asyncio.sleep(wait)
