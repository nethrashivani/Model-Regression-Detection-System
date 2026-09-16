
# Model Regression Detection System

A testing and evaluation pipeline for LLM-powered applications that detects quality regressions when prompts or models change.

## What it does

The system runs an LLM against a fixed **golden dataset** of 80 real, hand-verified customer support cases.

For each case, it evaluates:

- Classification accuracy
- Summary quality
- Response latency
- Token usage

It then compares the current evaluation with the previous run to detect:

- Regressions
- Improvements
- Changes in overall and per-category performance

Results are stored in **SQLite** and a self-contained **HTML report** is generated after each run.

## Current Implementation

- LLM-powered customer support classifier
- Versioned prompts using YAML
- 80-case golden dataset
- Async batch evaluation
- Multi-dimensional scoring
- Run-over-run regression detection
- SQLite evaluation history
- HTML evaluation reports
- Retry handling for API rate limits
- Groq, OpenAI and Ollama support

## Tech Stack

**Python · OpenAI-compatible APIs · Pydantic · YAML · SQLite · Pytest**

## Project Flow

```text
Golden Dataset
      ↓
LLM Classifier
      ↓
Scoring
      ↓
Compare with Previous Run
      ↓
Regression Detection
      ↓
SQLite + HTML Report
```

## Run

```bash
pip install -r requirements.txt
python scripts/run_eval.py
```

## Future Enhancements

- GitHub Actions CI/CD
- Slack regression alerts
- Drift detection
- Docker containerization
- Streamlit dashboard
