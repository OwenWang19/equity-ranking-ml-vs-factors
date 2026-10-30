"""Render a saved source-qualification audit without rereading or changing data."""

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_source_audit(audit_path: Path, output: Path):
    document = json.loads(audit_path.read_text())
    audit, gate = document["audit"], document["gate"]
    output.mkdir(parents=True, exist_ok=False)
    daily = pd.DataFrame.from_dict(audit["membership_by_date"], orient="index")
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    ratio = daily.members_with_close / daily.members
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(daily.index, daily.members, label="Provider-claimed members", color="#355d7a")
    axes[0].plot(daily.index, daily.members_with_close, label="With observed close", color="#b26333")
    axes[0].set(ylabel="Names", title="Claimed membership and price coverage")
    axes[0].legend()
    axes[1].plot(daily.index, ratio, color="#b26333")
    axes[1].axhline(
        gate["minimum_membership_close_coverage"], color="#355d7a", ls="--", label="Coverage gate"
    )
    axes[1].set(
        ylabel="Observed fraction", ylim=(0, 1.04), title="Quote availability within claimed membership"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    image_path = output / "source_coverage.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in gate["reasons"])
    missing = ", ".join(audit["tickers_without_close_data"])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Equity Alpha Research — Source Qualification</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#1d2c39}}
main{{max-width:1000px;margin:30px auto;padding:35px;background:white;border-radius:10px}}
img{{width:100%}}.status{{background:#fff0e8;padding:18px;border-left:4px solid #b26333}}
code{{font-size:12px;overflow-wrap:anywhere}}h1{{line-height:1.2}}
</style><main><h1>Public equity source qualification</h1>
<div class="status"><strong>Status: {html.escape(gate["status"].upper())}</strong>
<ul>{reasons}</ul></div>
<p>Archive calendar: {audit["calendar_start"]}–{audit["calendar_end"]};
{audit["n_tickers"]} historical symbol labels; {audit["membership_record_count"]} provider membership spans.
These spans are provider claims, not verified historical index membership.</p>
<p>Explicit vendor punctuation mapping increases observed coverage to
<strong>{audit["membership_close_coverage"]:.2%}</strong>.
{len(audit["tickers_without_close_data"])} labels have no closing observations;
{len(audit["membership_exits_without_next_session_close"])} membership exits lack a next-session close.
No missing quote is filled, and no missing historical member is silently removed.</p>
<img alt="Provider-claimed membership and quote coverage over time" src="data:image/png;base64,{encoded}">
<h2>Independently verified membership error</h2>
<p>The archive dates BRK.B membership to 1999. Berkshire's January 2010 issuer announcement
describes its new inclusion in the S&amp;P 500. This contradicts the archive's historical-membership claim.
The dates above are not repaired or presented as an authoritative historical universe.</p>
<p><a href="https://www.berkshirehathaway.com/news/jan2810.pdf">Primary issuer announcement</a> ·
<a href="{html.escape(audit["source_url"])}">Source archive</a> ·
<a href="{html.escape(audit["normalization_source"])}">Provider normalization code</a></p>
<h2>Labels with no close data</h2><p>{html.escape(missing)}</p>
<h2>Research consequence</h2>
<p>This source is retained for data-quality investigation. It is disqualified from the main broad-universe
alpha experiment. Ticker punctuation fixes recover observations but do not resolve missing bankrupt
companies, terminal payouts, historical identity changes or incorrect membership dates.</p>
<p>Original archive SHA-256: <code>{audit["source_sha256"]}</code>.<br>
The machine-readable audit retains per-name and per-date coverage, symbol mappings, binary offsets,
normalization conventions and specific exit gaps.</p></main></html>'''
    (output / "report.html").write_text(page)
    return output / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render_source_audit(args.audit, args.output))


if __name__ == "__main__":
    main()
