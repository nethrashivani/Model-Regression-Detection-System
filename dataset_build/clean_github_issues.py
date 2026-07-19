"""
Cleans real GitHub issue bodies into email-like technical support text.

We strip issue-template scaffolding (checkbox headers, "### Steps to
reproduce" labels, code fences) because that's UI chrome the repo's template
injects, not something the user wrote. What remains is the user's own
description of their problem, verbatim -- we don't rewrite or paraphrase it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw" / "github-issues"
OUT_PATH = Path(__file__).resolve().parent.parent / "golden_dataset" / "github_candidates.json"

_CHECKBOX_LINE = re.compile(r"^\s*-\s*\[[ xX]\].*$", re.MULTILINE)
_HEADER_LINE = re.compile(r"^#{1,6}\s*.*$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_BLANK = re.compile(r"\n{3,}")


def clean_body(raw: str) -> str:
    text = raw or ""
    text = _HTML_COMMENT.sub("", text)
    text = _CODE_FENCE.sub("", text)
    text = _CHECKBOX_LINE.sub("", text)
    text = _HEADER_LINE.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def main() -> None:
    candidates = []
    for path in RAW_DIR.glob("*.json"):
        repo = path.stem.replace("_", "/", 1)
        issues = json.loads(path.read_text(encoding="utf-8"))
        for issue in issues:
            if issue.get("pull_request"):
                continue
            cleaned = clean_body(issue.get("body", ""))
            if len(cleaned) < 80 or len(cleaned) > 900:
                continue  # too thin to be useful, or too long to read like an email
            candidates.append(
                {
                    "repo": repo,
                    "issue_number": issue["number"],
                    "title": issue["title"],
                    "cleaned_body": cleaned,
                    "url": issue["html_url"],
                }
            )

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(candidates)} candidates written to {OUT_PATH}")


if __name__ == "__main__":
    main()
