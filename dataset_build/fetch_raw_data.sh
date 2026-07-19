#!/usr/bin/env bash
# Fetches all raw real-world data used to build the golden dataset.
# Run this before build_bitext_cases.py / build_github_cases.py if /raw is empty.
#
# Requires: GITHUB_TOKEN env var set (used to avoid the 60/hr unauthenticated
# GitHub API rate limit -- create one at
# https://github.com/settings/personal-access-tokens/new with no special scopes,
# public read access is enough).

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw/github-issues

echo "== Cloning Bitext customer support dataset (CDLA-Sharing-1.0) =="
if [ ! -d raw/bitext ]; then
  git clone --depth 1 https://github.com/bitext/customer-support-llm-chatbot-training-dataset.git raw/bitext
else
  echo "raw/bitext already exists, skipping clone"
fi

echo "== Fetching real GitHub issues for the technical category =="
AUTH_HEADER=()
if [ -n "${GITHUB_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer $GITHUB_TOKEN")
fi

fetch_issues() {
  local repo="$1" label="$2"
  local fname
  fname=$(echo "$repo" | tr '/' '_')
  echo "  fetching $repo (label=$label)"
  curl -s "${AUTH_HEADER[@]}" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${repo}/issues?state=all&labels=${label}&per_page=25" \
    -o "raw/github-issues/${fname}.json"
}

fetch_issues "signalapp/Signal-Desktop" "bug"
fetch_issues "desktop/desktop" "bug"
fetch_issues "obsproject/obs-studio" "kind/bug"

echo "== Done. Raw data is in dataset_build/raw/ (gitignored) =="
