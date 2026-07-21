from pathlib import Path

from src.eval import storage


def test_save_and_retrieve_run(tmp_path: Path):
    db_path = tmp_path / "test.db"
    with storage.get_connection(db_path) as conn:
        run = storage.RunRecord(
            prompt_version="v1",
            dataset_version="v1",
            model="openai/gpt-oss-20b",
            total_cases=2,
            passed=1,
            pass_rate=0.5,
            avg_summary_score=4.0,
            avg_latency_ms=500.0,
            total_tokens=100,
            status="pass",
        )
        case_records = [
            storage.CaseResultRecord(
                case_id="billing-0001",
                expected_category="billing",
                actual_category="billing",
                category_match=True,
                expected_summary="Customer wants a refund.",
                actual_summary="Customer is requesting a refund.",
                summary_score=5,
                summary_score_reasoning="Matches closely.",
                latency_ms=450.0,
                prompt_tokens=50,
                completion_tokens=20,
                error=None,
                expected_difficulty="easy",
            ),
            storage.CaseResultRecord(
                case_id="technical-0001",
                expected_category="technical",
                actual_category="general",
                category_match=False,
                expected_summary="App crashes on export.",
                actual_summary="Customer has a question.",
                summary_score=1,
                summary_score_reasoning="Wrong category entirely.",
                latency_ms=550.0,
                prompt_tokens=60,
                completion_tokens=15,
                error=None,
                expected_difficulty="medium",
            ),
        ]
        run_id = storage.save_run(conn, run, case_records)
        assert run_id == 1

        fetched_run = storage.get_run(conn, run_id)
        assert fetched_run["pass_rate"] == 0.5
        assert fetched_run["model"] == "openai/gpt-oss-20b"

        fetched_cases = storage.get_case_results(conn, run_id)
        assert len(fetched_cases) == 2
        assert {c["case_id"] for c in fetched_cases} == {"billing-0001", "technical-0001"}


def test_get_latest_run_id_returns_none_when_empty(tmp_path: Path):
    db_path = tmp_path / "test.db"
    with storage.get_connection(db_path) as conn:
        assert storage.get_latest_run_id(conn) is None


def test_get_latest_run_id_returns_most_recent(tmp_path: Path):
    db_path = tmp_path / "test.db"
    minimal_run = storage.RunRecord(
        prompt_version="v1", dataset_version="v1", model="m",
        total_cases=1, passed=1, pass_rate=1.0,
        avg_summary_score=None, avg_latency_ms=None, total_tokens=None, status="pass",
    )
    with storage.get_connection(db_path) as conn:
        first_id = storage.save_run(conn, minimal_run, [])
        second_id = storage.save_run(conn, minimal_run, [])
        assert storage.get_latest_run_id(conn) == second_id
        assert second_id > first_id
