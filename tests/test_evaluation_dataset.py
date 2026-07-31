from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
ALLOWED_BEHAVIORS = {"answer", "no_answer", "clarification"}


def load_json(filename: str) -> dict:
    path = DATA_DIR / filename

    assert path.exists(), f"Test veri dosyası bulunamadı: {path}"

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def test_evaluation_dataset_has_expected_counts() -> None:
    dataset = load_json("evaluation_questions.json")
    counts = dataset["counts"]

    assert counts["answer"] == 20
    assert counts["no_answer"] >= 10
    assert counts["total"] == len(dataset["cases"])


def test_case_ids_and_questions_are_unique() -> None:
    cases = load_json("evaluation_questions.json")["cases"]

    case_ids = [
        case["case_id"]
        for case in cases
    ]

    questions = [
        case["question"].strip().casefold()
        for case in cases
    ]

    assert len(case_ids) == len(set(case_ids))
    assert len(questions) == len(set(questions))


def test_evaluation_cases_have_valid_structure() -> None:
    cases = load_json("evaluation_questions.json")["cases"]

    for case in cases:
        assert case["question"].strip()
        assert case["expected_behavior"] in ALLOWED_BEHAVIORS
        assert case["category"].strip()

        if case["expected_behavior"] == "answer":
            assert isinstance(
                case["expected_answer"],
                str,
            )

            assert case["expected_answer"].strip()

        else:
            assert case["expected_answer"] is None


def test_known_regressions_have_unique_issue_ids() -> None:
    cases = load_json("known_regressions.json")["cases"]

    issue_ids = [
        case["issue_id"]
        for case in cases
    ]

    assert len(issue_ids) == len(set(issue_ids))


def test_known_regressions_have_valid_behaviors() -> None:
    cases = load_json("known_regressions.json")["cases"]

    for case in cases:
        assert case["question"].strip()
        assert case["expected_behavior"] in ALLOWED_BEHAVIORS
        assert case["observed_behavior"] in ALLOWED_BEHAVIORS
        assert case["note"].strip()
