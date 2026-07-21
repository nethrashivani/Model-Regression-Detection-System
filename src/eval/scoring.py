"""
Pure comparison logic: given this run's per-case results and the previous
run's, compute deltas, regressions, improvements, and a warn/critical status.

Kept dependency-free (no DB, no network) on purpose so it's fully unit
testable -- this is the part of the system where a bug is most costly (a
threshold bug either hides a real regression or cries wolf on noise), so it
gets the most direct test coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WARNING_THRESHOLD = 0.03  # pass-rate drop that triggers a "warning"
CRITICAL_THRESHOLD = 0.08  # pass-rate drop that triggers a "critical" / merge-block


@dataclass
class CaseOutcome:
    case_id: str
    category_match: bool
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.category_match and self.error is None


@dataclass
class RunDiff:
    current_pass_rate: float
    previous_pass_rate: float | None
    pass_rate_delta: float | None
    regressions: list[str] = field(default_factory=list)  # case_ids: pass -> fail
    improvements: list[str] = field(default_factory=list)  # case_ids: fail -> pass
    status: str = "pass"  # "pass" | "warning" | "critical"
    category_deltas: dict[str, float] = field(default_factory=dict)


def compute_pass_rate(outcomes: list[CaseOutcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.passed) / len(outcomes)


def compute_category_pass_rates(
    outcomes: list[CaseOutcome], case_categories: dict[str, str]
) -> dict[str, float]:
    by_category: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        cat = case_categories.get(o.case_id, "unknown")
        by_category.setdefault(cat, []).append(o)
    return {cat: compute_pass_rate(group) for cat, group in by_category.items()}


def classify_status(pass_rate_delta: float | None) -> str:
    """delta is current - previous. Negative = got worse."""
    if pass_rate_delta is None:
        return "pass"  # no baseline to compare against yet (first-ever run)
    if pass_rate_delta <= -CRITICAL_THRESHOLD:
        return "critical"
    if pass_rate_delta <= -WARNING_THRESHOLD:
        return "warning"
    return "pass"


def diff_runs(
    current: list[CaseOutcome],
    previous: list[CaseOutcome] | None,
    case_categories: dict[str, str],
) -> RunDiff:
    current_rate = compute_pass_rate(current)

    if previous is None:
        return RunDiff(
            current_pass_rate=current_rate,
            previous_pass_rate=None,
            pass_rate_delta=None,
            status="pass",
            category_deltas={},
        )

    prev_by_id = {o.case_id: o for o in previous}
    curr_by_id = {o.case_id: o for o in current}

    regressions = []
    improvements = []
    for case_id, curr in curr_by_id.items():
        prev = prev_by_id.get(case_id)
        if prev is None:
            continue  # new case not present in the baseline run; nothing to diff
        if prev.passed and not curr.passed:
            regressions.append(case_id)
        elif not prev.passed and curr.passed:
            improvements.append(case_id)

    prev_rate = compute_pass_rate(previous)
    delta = current_rate - prev_rate

    current_cat_rates = compute_category_pass_rates(current, case_categories)
    prev_cat_rates = compute_category_pass_rates(previous, case_categories)
    category_deltas = {
        cat: current_cat_rates.get(cat, 0.0) - prev_cat_rates.get(cat, 0.0)
        for cat in set(current_cat_rates) | set(prev_cat_rates)
    }

    return RunDiff(
        current_pass_rate=current_rate,
        previous_pass_rate=prev_rate,
        pass_rate_delta=delta,
        regressions=sorted(regressions),
        improvements=sorted(improvements),
        status=classify_status(delta),
        category_deltas=category_deltas,
    )
