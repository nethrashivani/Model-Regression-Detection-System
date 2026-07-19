"""
Builds the golden dataset (billing / account / general portion) by sampling
real rows from the Bitext CSV.

Edge-case selection uses Bitext's OWN 'flags' column, which tags real
linguistic phenomena already present in the data:
  Z = typos / spelling errors      Q = colloquial phrasing
  N = negation                     I = interrogative structure
We are not inventing edge cases -- we're selecting real examples of them.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from entity_filler import fill_entities
from intent_map import INTENT_MAP

SRC_CSV = Path(__file__).resolve().parent / "raw" / "bitext" / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "golden_dataset" / "bitext_cases.json"

TARGET_PER_CATEGORY = 20  # for billing / account / general
EDGE_CASES_PER_CATEGORY = 4  # subset of the above, deliberately picked for difficulty
SEED = 42


def load_rows() -> list[dict]:
    with SRC_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def difficulty_for_flags(flags: str) -> tuple[str, str]:
    """Return (difficulty, note) based on Bitext's real linguistic tags."""
    if "Z" in flags:
        return "hard", "Contains a real typo/spelling error in the source text."
    if "Q" in flags:
        return "medium", "Real colloquial/informal phrasing."
    if "N" in flags:
        return "medium", "Real negation structure -- tests the model isn't fooled by surface keywords."
    if "C" in flags:
        return "medium", "Real coordinated/compound sentence structure."
    return "easy", "Standard phrasing."


def build() -> None:
    rows = load_rows()
    rng = random.Random(SEED)

    by_category: dict[str, list[dict]] = {"billing": [], "account": [], "general": []}

    for row in rows:
        mapping = INTENT_MAP.get(row["intent"])
        if mapping is None:
            continue
        category, summary = mapping
        by_category[category].append({**row, "_summary": summary})

    cases = []
    case_id = 1

    for category, candidates in by_category.items():
        rng.shuffle(candidates)

        # Deliberately grab some flagged edge cases first, then fill the rest with plain ones.
        edge = [c for c in candidates if any(f in c["flags"] for f in "ZQNC")][:EDGE_CASES_PER_CATEGORY]
        plain = [c for c in candidates if c not in edge]
        selected = edge + plain[: TARGET_PER_CATEGORY - len(edge)]

        for row in selected:
            row_id = f"{category}-{case_id:04d}"
            email_text = fill_entities(row["instruction"], row_id)
            difficulty, note = difficulty_for_flags(row["flags"])

            cases.append(
                {
                    "id": row_id,
                    "input": email_text,
                    "expected_category": category,
                    "expected_summary": row["_summary"],
                    "expected_difficulty": difficulty,
                    "notes": note,
                    "source": "bitext-real",
                    "source_intent": row["intent"],
                }
            )
            case_id += 1

    OUT_PATH.parent.mkdir(exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(cases)} cases to {OUT_PATH}")
    for cat in by_category:
        n = sum(1 for c in cases if c["expected_category"] == cat)
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    build()
