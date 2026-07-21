"""
The eval runner: classifies every case in the golden dataset against a given
prompt version, scores each result, diffs against the previous stored run,
and persists everything to SQLite.

Concurrency is capped with a semaphore rather than firing all 80 requests at
once -- free-tier API rate limits (Groq's included) are requests-per-minute
limited, and an uncapped asyncio.gather over 80 cases would burst past that.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from src.classifier import _build_messages, _parse_response  # reuse the exact prompt-building/parsing logic under test
from src.eval import storage
from src.eval.judge import judge_summary
from src.eval.scoring import CaseOutcome, diff_runs
from src.golden_dataset_loader import load_golden_dataset
from src.llm_provider import get_api_key, get_extra_body, get_provider_config
from src.models import GoldenCase, PromptConfig
from src.prompt_loader import load_prompt_config

DEFAULT_CONCURRENCY = 5
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


@dataclass
class SingleCaseResult:
    case: GoldenCase
    actual_category: str | None
    actual_summary: str | None
    category_match: bool
    summary_score: int | None
    summary_score_reasoning: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None


async def _run_one_case(
    case: GoldenCase,
    config: PromptConfig,
    client: AsyncOpenAI,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> SingleCaseResult:
    async with semaphore:
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"},
                messages=_build_messages(case.input, config),
                extra_body=get_extra_body(config.model),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            raw_content = response.choices[0].message.content or ""
            parsed = _parse_response(raw_content)  # raises ValueError on bad JSON/schema

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None

            category_match = parsed.category == case.expected_category
            score, reasoning = await judge_summary(
                case.input, case.expected_summary, parsed.summary, client, judge_model
            )

            return SingleCaseResult(
                case=case,
                actual_category=parsed.category,
                actual_summary=parsed.summary,
                category_match=category_match,
                summary_score=score,
                summary_score_reasoning=reasoning,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=None,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return SingleCaseResult(
                case=case,
                actual_category=None,
                actual_summary=None,
                category_match=False,
                summary_score=None,
                summary_score_reasoning=None,
                latency_ms=latency_ms,
                prompt_tokens=None,
                completion_tokens=None,
                error=str(e),
            )


async def run_eval(
    prompt_version: str = "v1",
    dataset_version: str = "v1",
    judge_model: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    """Runs the full eval and returns a dict with run_id, diff, case_results, run_meta."""
    config = load_prompt_config(prompt_version)
    cases = load_golden_dataset(dataset_version)
    judge_model = judge_model or config.model

    provider_cfg = get_provider_config()
    client = AsyncOpenAI(api_key=get_api_key(provider_cfg), base_url=provider_cfg.base_url)

    semaphore = asyncio.Semaphore(concurrency)
    try:
        results = await asyncio.gather(
            *[_run_one_case(case, config, client, judge_model, semaphore) for case in cases]
        )
    finally:
        await client.close()

    # --- persist ---
    case_records = [
        storage.CaseResultRecord(
            case_id=r.case.id,
            expected_category=r.case.expected_category,
            actual_category=r.actual_category,
            category_match=r.category_match,
            expected_summary=r.case.expected_summary,
            actual_summary=r.actual_summary,
            summary_score=r.summary_score,
            summary_score_reasoning=r.summary_score_reasoning,
            latency_ms=r.latency_ms,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            error=r.error,
            expected_difficulty=r.case.expected_difficulty,
        )
        for r in results
    ]

    outcomes = [
        CaseOutcome(case_id=r.case.id, category_match=r.category_match, error=r.error)
        for r in results
    ]
    case_categories = {r.case.id: r.case.expected_category for r in results}

    passed = sum(1 for o in outcomes if o.passed)
    pass_rate = passed / len(outcomes) if outcomes else 0.0
    scored = [r.summary_score for r in results if r.summary_score]
    avg_summary_score = sum(scored) / len(scored) if scored else None
    avg_latency_ms = sum(r.latency_ms for r in results) / len(results) if results else None
    total_tokens = sum(
        (r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in results
    ) or None

    with storage.get_connection() as conn:
        previous_run_id = storage.get_latest_run_id(conn)
        previous_outcomes = None
        if previous_run_id is not None:
            prev_rows = storage.get_case_results(conn, previous_run_id)
            previous_outcomes = [
                CaseOutcome(
                    case_id=row["case_id"],
                    category_match=bool(row["category_match"]),
                    error=row["error"],
                )
                for row in prev_rows
            ]

        diff = diff_runs(outcomes, previous_outcomes, case_categories)

        run_record = storage.RunRecord(
            prompt_version=prompt_version,
            dataset_version=dataset_version,
            model=config.model,
            total_cases=len(outcomes),
            passed=passed,
            pass_rate=pass_rate,
            avg_summary_score=avg_summary_score,
            avg_latency_ms=avg_latency_ms,
            total_tokens=total_tokens,
            status=diff.status,
        )
        run_id = storage.save_run(conn, run_record, case_records)
        case_rows = storage.get_case_results(conn, run_id)
        run_row = storage.get_run(conn, run_id)

    return {
        "run_id": run_id,
        "run_meta": {
            "run_id": run_id,
            "prompt_version": prompt_version,
            "dataset_version": dataset_version,
            "model": config.model,
            "created_at": run_row["created_at"],
        },
        "diff": diff,
        "case_rows": case_rows,
        "pass_rate": pass_rate,
        "passed": passed,
        "total": len(outcomes),
        "avg_summary_score": avg_summary_score,
        "avg_latency_ms": avg_latency_ms,
        "total_tokens": total_tokens,
    }
