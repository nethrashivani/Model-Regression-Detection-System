"""
CLI entrypoint for Phase 3's eval engine.

Usage:
    python scripts/run_eval.py                       # prompt v1 against dataset v1
    python scripts/run_eval.py --prompt-version v2
    python scripts/run_eval.py --concurrency 3        # lower if you're hitting rate limits
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.report_html import generate_html_report
from src.eval.runner import REPORTS_DIR, run_eval

STATUS_ICON = {"pass": "✅", "warning": "⚠️", "critical": "🚨"}


def print_terminal_summary(result: dict) -> None:
    diff = result["diff"]
    icon = STATUS_ICON.get(diff.status, "")

    print()
    print(f"{icon}  Eval run #{result['run_id']} — status: {diff.status.upper()}")
    print(f"    Prompt: {result['run_meta']['prompt_version']}  |  Model: {result['run_meta']['model']}")
    print(f"    Pass rate: {result['passed']}/{result['total']} ({result['pass_rate']*100:.1f}%)", end="")
    if diff.pass_rate_delta is not None:
        sign = "+" if diff.pass_rate_delta >= 0 else ""
        print(f"   (vs. previous run: {sign}{diff.pass_rate_delta*100:.1f}%)")
    else:
        print("   (no previous run to compare against)")

    if result["avg_summary_score"] is not None:
        print(f"    Avg summary score: {result['avg_summary_score']:.2f}/5")
    if result["avg_latency_ms"] is not None:
        print(f"    Avg latency: {result['avg_latency_ms']:.0f} ms")
    if result["total_tokens"] is not None:
        print(f"    Total tokens used: {result['total_tokens']}")

    if diff.regressions:
        print(f"\n    🔻 {len(diff.regressions)} regression(s): {', '.join(diff.regressions)}")
    if diff.improvements:
        print(f"    🔺 {len(diff.improvements)} improvement(s): {', '.join(diff.improvements)}")

    errors = [r for r in result["case_rows"] if r["error"]]
    if errors:
        print(f"\n    ⚠️  {len(errors)} case(s) errored (parse/API failure, counted as failed):")
        for row in errors[:5]:
            print(f"       {row['case_id']}: {row['error'][:120]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the golden-dataset eval.")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--concurrency", type=int, default=5, help="max concurrent API calls")
    parser.add_argument("--judge-model", default=None, help="defaults to the same model being evaluated")
    args = parser.parse_args()

    result = asyncio.run(
        run_eval(
            prompt_version=args.prompt_version,
            dataset_version=args.dataset_version,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )
    )

    print_terminal_summary(result)

    report_path = REPORTS_DIR / f"run_{result['run_id']}.html"
    generate_html_report(result["run_meta"], result["case_rows"], result["diff"], report_path)
    print(f"📄 HTML report: {report_path}")

    if result["diff"].status == "critical":
        sys.exit(1)  # non-zero exit so CI (Phase 5) can block merge on this


if __name__ == "__main__":
    main()
