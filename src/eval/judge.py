"""
LLM-as-judge: scores how well an actual summary captures the same meaning as
the expected (ground truth) summary, 1-5.

Kept as a separate, narrow call rather than folded into the classifier call
itself -- the judge's job is fundamentally different (compare two texts for
semantic equivalence) from the classifier's job (read one email and decide),
and keeping them separate means you can swap/tune the judge prompt without
touching the thing under test.
"""

from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from src.llm_provider import get_extra_body

_JUDGE_SYSTEM_PROMPT = """You are grading whether a generated summary captures the same meaning as a reference summary.

Score 1-5:
5 = Captures the same core request/issue, no meaningful information lost or added.
4 = Captures the core request, minor phrasing differences only.
3 = Captures the general topic but misses a meaningful detail.
2 = Only loosely related to the reference; misses the actual request.
1 = Wrong or unrelated to the reference summary.

Respond with ONLY a JSON object: {"score": <1-5 integer>, "reasoning": "<one short sentence>"}. No other text."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> tuple[int, str]:
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    if data is None:
        return 0, f"judge returned unparseable output: {raw!r}"

    score = data.get("score")
    reasoning = data.get("reasoning", "")
    if not isinstance(score, int) or not (1 <= score <= 5):
        return 0, f"judge returned invalid score: {data!r}"
    return score, str(reasoning)


async def judge_summary(
    email_text: str,
    expected_summary: str,
    actual_summary: str,
    client: AsyncOpenAI,
    judge_model: str,
) -> tuple[int, str]:
    """Returns (score 1-5, one-sentence reasoning). Returns (0, reason) if the judge call itself fails."""
    user_content = (
        f"Original email:\n{email_text}\n\n"
        f"Reference summary:\n{expected_summary}\n\n"
        f"Generated summary to grade:\n{actual_summary}"
    )
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            extra_body=get_extra_body(judge_model),
        )
    except Exception as e:  # judge failures shouldn't crash the whole eval run
        return 0, f"judge call failed: {e}"

    raw_content = response.choices[0].message.content or ""
    return _parse_judge_response(raw_content)
