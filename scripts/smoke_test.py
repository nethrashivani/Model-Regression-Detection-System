"""
Quick manual sanity check for Phase 1: loads v1.yaml, classifies a couple of
sample emails, and prints the result. Not a substitute for the golden dataset
eval in Phase 2/3 — just confirms the wiring works before you build on top of it.

Usage (Groq, the default -- free, no credit card):
    1. Put GROQ_API_KEY=gsk_... in your .env file (see .env.example)
    2. python scripts/smoke_test.py
    (python-dotenv loads .env automatically -- no manual export/$env: needed,
    works the same on bash, zsh, and PowerShell)

Usage (Ollama, fully local/offline):
    ollama pull llama3.1   # once
    ollama serve            # keep running in another terminal
    Set LLM_PROVIDER=ollama in .env
    python scripts/smoke_test.py
    # also change prompts/v1.yaml's `model:` to match what you pulled
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import classify_email
from src.prompt_loader import load_prompt_config

SAMPLE_EMAILS = [
    "My card was declined but the money still left my bank account. Please help.",
    "Every time I export a report to PDF the file comes out blank. Anyone else seeing this?",
    "Hey, just wondering what time zone your support team operates in.",
]


def main() -> None:
    config = load_prompt_config("v1")
    print(f"Loaded prompt config: {config.version} ({config.description})\n")

    for email in SAMPLE_EMAILS:
        result = classify_email(email, config)
        print(f"INPUT:    {email}")
        print(f"CATEGORY: {result.category}")
        print(f"SUMMARY:  {result.summary}")
        print("-" * 60)


if __name__ == "__main__":
    main()
