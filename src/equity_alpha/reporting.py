"""Static scientific exhibits and a self-contained, evidence-first HTML report."""

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_report(output: Path, result, manifest, trials, splits, importance):
    output.mkdir(parents=True, exist_ok=True)
    for key in (
        "summary",
        "folds",
        "uncertainty",
        "paired",
        "sensitivity",
        "daily_ic",
        "ic_decay",
        "quintiles",
        "contributions",
        "schedule",
        "forecast_mse",
        "daily_mse",
        "paired_mse",
        "global_raw_ic",
    ):
        result[key].to_csv(output / f"{key}.csv")
    trials.to_csv(output / "trial_ledger.csv", index=False)
    splits.to_csv(output / "split_audit.csv", index=False)
    importance.to_csv(output / "validation_feature_importance.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    for name, frame in result["curves"].items():
        frame.to_csv(output / f"daily_{name}.csv")
        result["targets"][name].to_csv(output / f"targets_{name}.csv")
    plt.rcParams.update(
        {"font.family": "sans-serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    is_v2 = manifest["config"].get("selection_metric") == "sector_mse"
    shown = (
        ["ridge_ungated", "hgb_ungated", "ridge", "hgb"]
        if is_v2
        else ["factor_composite", "ridge", "hgb", "selected"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for name in shown:
        curve = result["curves"][name]
        axes[0, 0].plot(curve.index, curve.equity, label=name, lw=1.4)
        axes[0, 1].plot(curve.index, curve.equity / curve.equity.cummax() - 1, label=name, lw=1)
        axes[1, 0].plot(
            result["daily_ic"].index, result["daily_ic"][name].rolling(60).mean(), label=name, lw=1
        )
    axes[0, 0].set(title="Development walk-forward: net NAV", ylabel="Initial NAV = 1")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Net drawdown", ylabel="Fraction")
    axes[1, 0].set(title="60-session mean daily Rank IC", ylabel="Rank correlation")
    axes[1, 0].axhline(0, color="black", lw=0.6)
    costs = result["sensitivity"].query("scenario == 'one_way_fee'")
    for name in ["factor_composite", "ridge", "hgb", "selected"]:
        subset = costs.loc[costs.model == name]
        axes[1, 1].plot(subset.cost_bps, subset.net_sharpe, marker="o", label=name)
    axes[1, 1].set(title="Cost stress (borrow retained)", xlabel="One-way cost, bps", ylabel="Net Sharpe")
    axes[1, 1].axhline(0, color="black", lw=0.6)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    figure_path = output / "research_exhibits.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    plot_data = base64.b64encode(figure_path.read_bytes()).decode()
    summary = result["summary"].loc[
        shown,
        [
            "mean_daily_rank_ic",
            "net_sharpe",
            "annualized_return",
            "max_drawdown",
            "annualized_turnover",
            "realized_panel_beta",
        ],
    ]
    intervals = result["paired"]
    tree_vs_linear = intervals.query("model_a == 'hgb' and model_b == 'ridge'").iloc[0]
    if tree_vs_linear.lower_95 <= 0 <= tree_vs_linear.upper_95:
        verdict = (
            "The paired block-bootstrap interval for tree-minus-linear net Sharpe includes zero. "
            "These development results do not establish a reliable nonlinear advantage."
        )
    elif tree_vs_linear.lower_95 > 0:
        verdict = (
            "The tree model outperforms the linear comparator in this development sample under the stated "
            "paired interval. Survivor selection, research iterations and the reserved final evaluation "
            "still limit the claim."
        )
    else:
        verdict = "The tree model underperforms the linear comparator under the stated paired interval."
    if is_v2:
        comparison = result["paired_mse"].iloc[0]
        if comparison.lower_95 <= 0 <= comparison.upper_95:
            verdict = (
                "The paired interval for ungated tree-versus-linear forecast MSE reduction includes zero. "
                "This experiment does not establish incremental nonlinear predictive value."
            )
        elif comparison.lower_95 > 0:
            verdict = (
                "The ungated tree has lower forecast MSE under this development comparison. "
                "The reserved final evaluation and data limitations still constrain the finding."
            )
        else:
            verdict = "The ungated tree has higher forecast MSE than Ridge under the paired interval."
    predictive = (
        "<h2>Primary forecast comparison before deployment gates</h2>"
        + result["forecast_mse"].to_html(index=False, float_format=lambda x: f"{x:.6g}")
        + result["paired_mse"].to_html(index=False, float_format=lambda x: f"{x:.6g}")
    )
    gate_explanation = (
        "Selection minimizes validation return-unit sector-relative MSE. A model deploys only "
        "when its validation MSE beats the zero forecast and its sector Rank IC is positive. "
        "Otherwise it holds cash. Ungated forecasts remain visible for the primary comparison."
        if is_v2
        else "Selection uses inner-validation mean daily Rank IC."
    )
    gate_table = (
        trials.loc[
            trials.kind.str.startswith("selected_"), ["fold", "kind", "candidate", "active", "gate_reason"]
        ].to_html(index=False)
        if is_v2
        else ""
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Equity Alpha Research | Development Review</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#172535;background:#f4f6f8;margin:0}}
main{{max-width:1180px;margin:36px auto;background:white;padding:40px;border-radius:12px}}
h1{{font-size:34px;line-height:1.2}}h2{{margin-top:36px}}.tag{{color:#526c82;font-weight:600}}
.notice{{padding:18px;background:#edf3f8;border-left:4px solid #315c7b}}img{{width:100%}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #dce3e8;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}code{{font-size:13px}}
</style><main><p class="tag">{
        html.escape(manifest["config"]["experiment_id"].upper())
    } · FIXED 30-NAME PILOT · DEVELOPMENT ONLY</p>
<h1>Does nonlinear ML add value beyond simple equity signals?</h1>
<p>Price-only cross-sectional five-session forecasts, delayed closing-price execution,
annual nested chronological validation and self-financing portfolio accounting.</p>
<div class="notice"><strong>Finding:</strong> {html.escape(verdict)}<br>
2025 onward remains reserved; its prediction and portfolio metrics have not been evaluated.</div>
{predictive}
<p>MSE is measured in squared basis points, with positive paired reduction favoring trees.
Only return-unit model forecasts enter this comparison; arbitrary rule ranks do not.
{html.escape(gate_explanation)}</p>{gate_table}
<h2>Comparable model evidence</h2><div class="scroll">{
        summary.to_html(float_format=lambda x: f"{x:.4f}")
    }</div>
<p>All models use the same sector-neutral target constructor, 8% target position cap, gross exposure ≤1,
{manifest["config"]["one_way_cost_bps"]:g} bps one-way fees and
{
        manifest["config"][
            "annual_short_borrow_bps"
        ]:g} bps annual assumed short borrow. Exposure drifts between trades.
<code>selected</code> follows the frozen validation rule; it is not chosen using test returns.</p>
<img alt="NAV, drawdown, rolling IC and fee sensitivity" src="data:image/png;base64,{plot_data}">
<h2>Paired uncertainty</h2>{intervals.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
<p>95% circular date-block bootstrap intervals, 20-session blocks, paired return paths.
Intervals are conditional on this universe and frozen candidate set; they do not correct unknown prior
research trials or establish live profitability. Three within-date label permutation runs are diagnostics,
not a formal permutation p-value.</p>
<h2>Temporal split audit</h2><div class="scroll">{splits.to_html(index=False)}</div>
<p>Features at close t → assumed close t+1 execution → close t+6 label maturity.
Training, validation and refit samples are purged by actual label end; every name on a date shares its partition.
Fitted scalers use training only and gradient boosting has no random internal validation.</p>
<h2>Predeclared controls and sensitivity</h2><div class="scroll">{
        result["summary"][["mean_daily_rank_ic", "net_sharpe", "max_drawdown"]].to_html(
            float_format=lambda x: f"{x:.4f}"
        )
    }</div>
<p>Leave-family-out models use the selected tree hyperparameters without retuning. Cost stress preserves
borrow charges. The extra-session-delay scenario reports its own completed holding-interval sample in
sensitivity.csv. Horizon decay and quintile returns are descriptive diagnostics with overlapping labels.
quintiles.csv separates raw forward returns from the aligned target forward return; global_raw_ic.csv
is a secondary raw-return rank diagnostic, separate from the V2 sector-relative primary metric.
contributions.csv sums per-name arithmetic pre-cost daily return contributions; these are not compounded
total-return contributions or net P&amp;L allocated by name.</p>
<h2>Data and economic limits</h2>
<p>{manifest["data"]["n_tickers"]} selected surviving names, {manifest["data"]["n_dates"]} dates,
{html.escape(manifest["data"]["start_date"])} to {html.escape(manifest["data"]["end_date"])}.
Inputs are reconciled adjusted prices and returns. No point-in-time constituent history, publication-dated
fundamentals, volume, spreads or borrow quotes are available. Sector labels are static. Results cannot establish
capacity, executable auction fills or broad-market generalization.</p>
<p>Holdings are synthetic adjusted-price total-return units. Cash earns zero; short proceeds earn no rebate.
No live exchange or broker integration is present. Source data was already used in coursework: the reserved
period is a frozen retrospective evaluation, not a claim that its history has never been seen.</p>
<h2>Audit artifacts</h2><p>manifest.json records input/config/source hashes and software versions;
trial_ledger.csv records all candidate selections and controls; split_audit.csv records label maturity;
daily_*.csv and targets_*.csv expose accounting. No model is promoted based on this report alone.</p>
<p>Config SHA-256: <code>{manifest["config_sha256"]}</code></p></main></html>"""
    (output / "report.html").write_text(page)
    return verdict
