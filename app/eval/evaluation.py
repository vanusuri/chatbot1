import json
from pathlib import Path
from typing import Dict, Any, List

from app.orchestrator import Orchestrator
from app.db.dao import init_db


def load_test_cases(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_cases_path: str = "app/eval/test_cases.json") -> None:
    init_db()
    orchestrator = Orchestrator()
    cases = load_test_cases(test_cases_path)

    total = len(cases)
    correct_category = 0

    for case in cases:
        msg = case["input"]
        expected_category = case["expected_category"]
        result = orchestrator.handle_message(
            message=msg,
            session_id="eval-session",
            customer_name=None,
        )
        predicted_category = result.get("category")
        if predicted_category == expected_category:
            correct_category += 1

    accuracy = correct_category / total if total else 0.0
    print(f"Total test cases: {total}")
    print(f"Correct category predictions: {correct_category}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    run_evaluation()
