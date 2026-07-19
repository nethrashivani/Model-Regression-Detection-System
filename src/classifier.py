"""
The LLM feature under test: a customer support email classifier.

Deliberately kept as a single, simple function with the prompt injected as a
parameter (PromptConfig) rather than hardcoded. This is what lets the eval
pipeline run the *same* function against many prompt versions and diff the
results — the function is the harness, PromptConfig is the variable.

LLM provider (Groq / OpenAI / Ollama) is chosen via LLM_PROVIDER env var --
see src/llm_provider.py. All three speak the OpenAI-compatible API, so no
branching is needed here beyond which client to construct.
"""

from __future__ import annotations

import json
import re

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from src.llm_provider import get_api_key, get_provider_config
from src.models import EmailClassification, PromptConfig

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object of the form "
    '{"category": "<billing|technical|account|general>", "summary": "<string>"}. '
    "No other text, no markdown code fences, no explanation."
)

# Some open-weight models wrap JSON in ```json ... ``` fences or add a
# sentence before/after despite instructions not to. Extract the first
# {...} block as a fallback before giving up.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _build_messages(email_text: str, config: PromptConfig) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": config.system_prompt + _JSON_INSTRUCTION}
    ]
    for example in config.few_shot_examples:
        messages.append({"role": "user", "content": example.input})
        messages.append(
            {"role": "assistant", "content": example.output.model_dump_json()}
        )
    messages.append({"role": "user", "content": email_text})
    return messages


def _parse_response(raw_content: str) -> EmailClassification:
    data = None
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw_content)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_content!r}"
        )

    try:
        return EmailClassification(**data)
    except ValidationError as e:
        raise ValueError(
            f"Model output failed the EmailClassification schema: {e}. "
            f"Raw output: {raw_content!r}"
        ) from e


def _make_sync_client() -> OpenAI:
    cfg = get_provider_config()
    return OpenAI(api_key=get_api_key(cfg), base_url=cfg.base_url)


def _make_async_client() -> AsyncOpenAI:
    cfg = get_provider_config()
    return AsyncOpenAI(api_key=get_api_key(cfg), base_url=cfg.base_url)


def classify_email(email_text: str, config: PromptConfig) -> EmailClassification:
    """Synchronous single-email classification. Used for quick manual testing."""
    client = _make_sync_client()
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
        messages=_build_messages(email_text, config),
    )
    raw_content = response.choices[0].message.content or ""
    return _parse_response(raw_content)


async def classify_email_async(
    email_text: str, config: PromptConfig, client: AsyncOpenAI | None = None
) -> EmailClassification:
    """
    Async classification, for use by the eval runner's batching in Phase 3.

    Accepts an optional shared AsyncOpenAI client so the eval runner can reuse
    one client across hundreds of concurrent calls instead of opening a new
    connection per request.
    """
    owns_client = client is None
    client = client or _make_async_client()
    try:
        response = await client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
            messages=_build_messages(email_text, config),
        )
        raw_content = response.choices[0].message.content or ""
        return _parse_response(raw_content)
    finally:
        if owns_client:
            await client.close()
