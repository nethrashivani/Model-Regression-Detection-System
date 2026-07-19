"""
The technical-category golden cases. Each entry below was selected and
summarized by hand after reading the real GitHub issue -- this is the
"human-verified ground truth" step for this category, same spirit as
hand-writing summaries for the Bitext-derived cases.

(repo, issue_number, expected_summary, expected_difficulty, note)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "golden_dataset" / "github_candidates.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "golden_dataset" / "github_cases.json"

MAX_LEN = 500

SELECTED = [
    ("desktop/desktop", 22572, "easy",
     "Customer reports GitHub Desktop shows no repositories when trying to clone, after it worked the day before.", ""),
    ("desktop/desktop", 22569, "medium",
     "Customer is having trouble deleting a feature branch while using git worktrees.",
     "Multi-step technical scenario; tests whether the model still tags it 'technical' despite the git jargon."),
    ("desktop/desktop", 22520, "easy",
     "Customer reports the worktree toolbar shows 'never fetched' even though a fetch succeeded.", ""),
    ("desktop/desktop", 22510, "hard",
     "Customer reports all their repositories disappeared from GitHub Desktop after an app auto-update.",
     "High-severity data-loss-sounding bug; good test that severity doesn't change the category."),
    ("desktop/desktop", 22478, "easy",
     "Customer reports that cancelling a repository clone does not actually stop the download.", ""),
    ("desktop/desktop", 22393, "easy",
     "Customer reports that middle mouse button clicks are incorrectly treated as left clicks when selecting code hunks.", ""),
    ("desktop/desktop", 22387, "medium",
     "Customer reports the repository list occasionally gets stuck and won't scroll after updating the app.", ""),
    ("desktop/desktop", 22380, "medium",
     "Customer reports an error when trying to generate a commit message with the built-in Copilot feature.", ""),
    ("desktop/desktop", 22365, "medium",
     "Customer reports the pull request refresh button does nothing when they are not logged in, with no error shown.",
     "Silent failure (no error message) -- tests whether 'nothing happens' still reads as technical, not general."),
    ("desktop/desktop", 22237, "hard",
     "Customer reports the app crashes on startup for users who also have a specific third-party tool installed.",
     "Root cause involves a third-party interaction; tests generalization beyond simple single-app bugs."),
    ("signalapp/Signal-Desktop", 7903, "medium",
     "Customer found a minor UI bug where selecting a button while pressing ctrl+c copies internal button data instead of expected content.", ""),
    ("signalapp/Signal-Desktop", 7902, "easy",
     "Customer reports the app crashes when trying to set up desktop backups.", ""),
    ("signalapp/Signal-Desktop", 7841, "medium",
     "Customer reports pinned messages don't always show as pinned after not opening the app for a day or so.", ""),
    ("signalapp/Signal-Desktop", 7838, "medium",
     "Customer reports getting excessive repeated admin alerts in group chat threads.", ""),
    ("signalapp/Signal-Desktop", 7786, "easy",
     "Customer reports Chinese contact names are displayed incorrectly, showing only part of the name.", ""),
    ("signalapp/Signal-Desktop", 7765, "hard",
     "Customer reports the auto-delete timer setting isn't being applied to messages forwarded to new contacts from a group chat.",
     "Multi-clause technical scenario describing a specific reproduction path."),
    ("signalapp/Signal-Desktop", 7761, "easy",
     "Customer reports the app stopped working on Windows after a recent Windows security update.", ""),
    ("signalapp/Signal-Desktop", 7719, "medium",
     "Customer reports hearing an audible hiss during calls when background audio is very quiet, starting from a specific app version.", ""),
    ("signalapp/Signal-Desktop", 7678, "medium",
     "Customer reports the microphone indicator stays on indefinitely after ending calls, a recent regression.", ""),
    ("signalapp/Signal-Desktop", 7666, "easy",
     "Customer reports a keyboard shortcut that used to open the chat menu no longer works after an update.", ""),
]

_TRAILING_JUNK = re.compile(r"\s+$")


def truncate(text: str, max_len: int = MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # cut at the last sentence boundary if reasonably close to the end
    last_period = cut.rfind(". ")
    if last_period > max_len * 0.6:
        cut = cut[: last_period + 1]
    return _TRAILING_JUNK.sub("", cut)


def main() -> None:
    candidates = {(c["repo"], c["issue_number"]): c for c in json.loads(CANDIDATES_PATH.read_text())}

    cases = []
    for idx, (repo, num, difficulty, summary, note) in enumerate(SELECTED, start=1):
        c = candidates.get((repo, num))
        if c is None:
            print(f"WARNING: missing candidate {repo}#{num}, skipping")
            continue
        cases.append(
            {
                "id": f"technical-{idx:04d}",
                "input": truncate(c["cleaned_body"]),
                "expected_category": "technical",
                "expected_summary": summary,
                "expected_difficulty": difficulty,
                "notes": note or "Real GitHub issue, cleaned of template scaffolding.",
                "source": "github-real",
                "source_url": c["url"],
            }
        )

    OUT_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(cases)} technical cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
