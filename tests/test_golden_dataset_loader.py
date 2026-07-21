import pytest

from src.golden_dataset_loader import load_golden_dataset


def test_loads_all_80_cases():
    cases = load_golden_dataset("v1")
    assert len(cases) == 80


def test_balanced_across_categories():
    cases = load_golden_dataset("v1")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.expected_category] = counts.get(c.expected_category, 0) + 1
    assert counts == {"billing": 20, "account": 20, "general": 20, "technical": 20}


def test_missing_version_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="No golden dataset found"):
        load_golden_dataset("v99")


def test_every_case_has_nonempty_input_and_summary():
    cases = load_golden_dataset("v1")
    for c in cases:
        assert c.input.strip()
        assert c.expected_summary.strip()
