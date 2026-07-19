"""
The LLM feature under test: a customer support email classifier.

Deliberately kept as a single, simple function with the prompt injected as a
parameter (PromptConfig) rather than hardcoded. This is what lets the eval
pipeline run the *same* function against many prompt versions and diff the
results — the function is the harness, PromptConfig is the variable.
"""

from __future__ import annotations

import json
import os

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from src.models import EmailClassification, PromptConfig

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object of the form "
    '{"category": "<billing|technical|account|general>", "summary": "<string>"}. '
    "No other text."
)


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
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_content!r}"
        ) from e

    try:
        return EmailClassification(**data)
    except ValidationError as e:
        raise ValueError(
            f"Model output failed the EmailClassification schema: {e}. "
            f"Raw output: {raw_content!r}"
        ) from e


def classify_email(email_text: str, config: PromptConfig) -> EmailClassification:
    """Synchronous single-email classification. Used for quick manual testing."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
    client = client or AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
