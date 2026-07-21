from src.eval.scoring import (
    CaseOutcome,
    classify_status,
    compute_category_pass_rates,
    compute_pass_rate,
    diff_runs,
)


def test_compute_pass_rate_empty():
    assert compute_pass_rate([]) == 0.0


def test_compute_pass_rate_basic():
    outcomes = [
        CaseOutcome("a", category_match=True),
        CaseOutcome("b", category_match=True),
        CaseOutcome("c", category_match=False),
        CaseOutcome("d", category_match=False),
    ]
    assert compute_pass_rate(outcomes) == 0.5


def test_error_counts_as_failed_even_if_category_matches():
    # category_match could technically be True if error happened after parsing,
    # but passed should always require error is None
    outcome = CaseOutcome("a", category_match=True, error="timeout")
    assert outcome.passed is False


def test_classify_status_thresholds():
    assert classify_status(None) == "pass"           # no baseline yet
    assert classify_status(0.05) == "pass"            # improved
    assert classify_status(-0.02) == "pass"           # within noise
    assert classify_status(-0.03) == "warning"        # exactly at warning threshold
    assert classify_status(-0.05) == "warning"
    assert classify_status(-0.08) == "critical"       # exactly at critical threshold
    assert classify_status(-0.15) == "critical"


def test_diff_runs_first_ever_run_has_no_baseline():
    current = [CaseOutcome("a", True), CaseOutcome("b", False)]
    diff = diff_runs(current, previous=None, case_categories={"a": "billing", "b": "billing"})
    assert diff.previous_pass_rate is None
    assert diff.pass_rate_delta is None
    assert diff.status == "pass"
    assert diff.regressions == []
    assert diff.improvements == []


def test_diff_runs_detects_regression():
    previous = [CaseOutcome("a", True), CaseOutcome("b", True)]
    current = [CaseOutcome("a", True), CaseOutcome("b", False)]  # b flipped to fail
    diff = diff_runs(current, previous, case_categories={"a": "billing", "b": "billing"})
    assert diff.regressions == ["b"]
    assert diff.improvements == []


def test_diff_runs_detects_improvement():
    previous = [CaseOutcome("a", False), CaseOutcome("b", True)]
    current = [CaseOutcome("a", True), CaseOutcome("b", True)]  # a flipped to pass
    diff = diff_runs(current, previous, case_categories={"a": "billing", "b": "billing"})
    assert diff.improvements == ["a"]
    assert diff.regressions == []


def test_diff_runs_ignores_cases_not_in_baseline():
    previous = [CaseOutcome("a", True)]
    current = [CaseOutcome("a", True), CaseOutcome("new_case", False)]
    diff = diff_runs(current, previous, case_categories={"a": "billing", "new_case": "billing"})
    assert diff.regressions == []  # new_case has no prior state to regress from
    assert diff.improvements == []


def test_diff_runs_flags_critical_status():
    # 10 cases, previous 9/10 passed (90%), current 0/10 passed (0%) -> -90% delta
    previous = [CaseOutcome(str(i), True) for i in range(9)] + [CaseOutcome("9", False)]
    current = [CaseOutcome(str(i), False) for i in range(10)]
    diff = diff_runs(current, previous, case_categories={str(i): "billing" for i in range(10)})
    assert diff.status == "critical"


def test_compute_category_pass_rates():
    outcomes = [
        CaseOutcome("a", True),
        CaseOutcome("b", False),
        CaseOutcome("c", True),
    ]
    categories = {"a": "billing", "b": "billing", "c": "technical"}
    rates = compute_category_pass_rates(outcomes, categories)
    assert rates["billing"] == 0.5
    assert rates["technical"] == 1.0
