import httpx
import pytest
from openai import RateLimitError

from src.eval.retry import with_retries


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, request=req, headers=headers)
    return RateLimitError("rate limited", response=resp, body=None)


@pytest.mark.asyncio
async def test_succeeds_on_first_try_without_retrying():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "ok"

    result = await with_retries(fn, max_retries=3, base_delay=0.01)
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _make_rate_limit_error()
        return "ok"

    result = await with_retries(fn, max_retries=5, base_delay=0.01)
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise _make_rate_limit_error()

    with pytest.raises(RateLimitError):
        await with_retries(fn, max_retries=2, base_delay=0.01)
    assert calls == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_honors_retry_after_header():
    import time

    calls = 0
    start = time.perf_counter()

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _make_rate_limit_error(retry_after="0.05")
        return "ok"

    result = await with_retries(fn, max_retries=3, base_delay=10.0)  # base_delay way higher than the header value
    elapsed = time.perf_counter() - start
    assert result == "ok"
    assert elapsed < 1.0  # proves it used the 0.05s header, not the 10s base_delay


@pytest.mark.asyncio
async def test_non_rate_limit_errors_are_not_retried():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ValueError("some other error")

    with pytest.raises(ValueError):
        await with_retries(fn, max_retries=5, base_delay=0.01)
    assert calls == 1  # not retried at all
