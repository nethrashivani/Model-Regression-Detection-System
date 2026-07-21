"""
Generates a single self-contained HTML report for one eval run: a scorecard,
the list of regressions/improvements, and a full per-case results table.

Deliberately plain inline CSS, no JS framework, no external requests -- this
needs to render correctly as a static file attached to a Slack message or a
GitHub Actions artifact, not depend on a live server.
"""

from __future__ import annotations

import html
from pathlib import Path

STATUS_COLORS = {"pass": "#2e7d32", "warning": "#e6a700", "critical": "#c0392b"}


def _esc(s: str | None) -> str:
    return html.escape(s or "")


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _fmt_delta(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.1f}%"


def generate_html_report(run_meta: dict, case_rows: list[dict], diff, out_path: Path) -> Path:
    """
    run_meta: dict with prompt_version, dataset_version, model, created_at, run_id
    case_rows: list of dicts (sqlite3.Row-like) from storage.get_case_results
    diff: scoring.RunDiff for this run vs. the previous one
    """
    status = diff.status
    status_color = STATUS_COLORS.get(status, "#555")

    regression_set = set(diff.regressions)
    improvement_set = set(diff.improvements)

    rows_html = []
    for row in case_rows:
        case_id = row["case_id"]
        row_class = ""
        if case_id in regression_set:
            row_class = "regressed"
        elif case_id in improvement_set:
            row_class = "improved"
        elif not row["category_match"] or row["error"]:
            row_class = "failed"

        match_icon = "✅" if row["category_match"] else "❌"
        score = row["summary_score"]
        score_display = f"{score}/5" if score is not None else "—"
        error_html = f'<div class="error-note">{_esc(row["error"])}</div>' if row["error"] else ""

        rows_html.append(
            f"""
            <tr class="{row_class}">
                <td>{_esc(case_id)}</td>
                <td>{_esc(row["expected_difficulty"])}</td>
                <td>{match_icon}</td>
                <td>{_esc(row["expected_category"])}</td>
                <td>{_esc(row["actual_category"])}</td>
                <td>{score_display}</td>
                <td class="text-cell">{_esc(row["expected_summary"])}</td>
                <td class="text-cell">{_esc(row["actual_summary"])}{error_html}</td>
                <td>{row["latency_ms"]:.0f} ms</td>
            </tr>"""
        )

    def _case_list_html(case_ids: list[str], empty_label: str) -> str:
        if not case_ids:
            return f'<p class="muted">{empty_label}</p>'
        return "<ul>" + "".join(f"<li>{_esc(c)}</li>" for c in case_ids) + "</ul>"

    category_deltas_html = "".join(
        f'<li>{_esc(cat)}: {_fmt_delta(delta)}</li>'
        for cat, delta in sorted(diff.category_deltas.items())
    ) or '<li class="muted">No baseline to compare against yet</li>'

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Report — {_esc(run_meta['prompt_version'])} — run #{run_meta['run_id']}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 32px; background: #fafafa; color: #1a1a1a; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .scorecard {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; min-width: 140px; }}
  .card .label {{ font-size: 12px; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }}
  .card .value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; color: white; font-weight: 600; font-size: 13px; background: {status_color}; }}
  .diff-section {{ display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }}
  .diff-box {{ background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 220px; }}
  .diff-box h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
  .muted {{ color: #999; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{ background: #f0f0f0; position: sticky; top: 0; }}
  .text-cell {{ max-width: 260px; }}
  tr.regressed {{ background: #fdecea; }}
  tr.improved {{ background: #eaf6ec; }}
  tr.failed {{ background: #fff8e1; }}
  .error-note {{ color: #c0392b; font-size: 11px; margin-top: 2px; }}
  ul {{ margin: 4px 0; padding-left: 18px; font-size: 13px; }}
</style>
</head>
<body>
  <h1>Eval Report <span class="status-badge">{status.upper()}</span></h1>
  <div class="meta">
    Prompt <strong>{_esc(run_meta['prompt_version'])}</strong> ·
    Dataset <strong>{_esc(run_meta['dataset_version'])}</strong> ·
    Model <strong>{_esc(run_meta['model'])}</strong> ·
    Run #{run_meta['run_id']} · {_esc(run_meta['created_at'])}
  </div>

  <div class="scorecard">
    <div class="card"><div class="label">Pass rate</div><div class="value">{_fmt_pct(diff.current_pass_rate)}</div></div>
    <div class="card"><div class="label">vs. previous run</div><div class="value">{_fmt_delta(diff.pass_rate_delta)}</div></div>
    <div class="card"><div class="label">Regressions</div><div class="value">{len(diff.regressions)}</div></div>
    <div class="card"><div class="label">Improvements</div><div class="value">{len(diff.improvements)}</div></div>
  </div>

  <div class="diff-section">
    <div class="diff-box">
      <h3>Regressed cases (passed before, fail now)</h3>
      {_case_list_html(diff.regressions, "None 🎉")}
    </div>
    <div class="diff-box">
      <h3>Improved cases (failed before, pass now)</h3>
      {_case_list_html(diff.improvements, "None")}
    </div>
    <div class="diff-box">
      <h3>Per-category pass-rate delta</h3>
      <ul>{category_deltas_html}</ul>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Case ID</th><th>Difficulty</th><th>Match</th>
        <th>Expected cat.</th><th>Actual cat.</th><th>Summary score</th>
        <th>Expected summary</th><th>Actual summary</th><th>Latency</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
