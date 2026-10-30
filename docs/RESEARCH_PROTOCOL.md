# Equity Alpha Research: predeclared pilot protocol

Status: implementation and audit protocol. This document specifies the experiment; it is not evidence that the experiment has passed its gates or discovered alpha. The machine-readable settings are in `configs/research.json`. Material changes require a new experiment identifier and a recorded reason.

## Research question

Does regularized nonlinear learning improve five-session cross-sectional equity return forecasts, and their cost-adjusted portfolio performance, relative to simple momentum/reversal rules and Ridge regression on the same information set?

The contribution being tested is incremental predictive value. A higher in-sample fit, a more complex model, or a positive gross backtest is insufficient. A useful negative result identifies which apparent improvement disappears under chronological validation, exposure control, or trading costs.

## Source evidence and scope

The starting point is the FE5108 course project. Its provided daily panel contains `date`, `ticker`, `sector`, `adj_close`, and `adj_return`, with observations from 2018 through 25 September 2026. Monthly inputs also contain market capitalization, book value, and ROE, but do not provide publication timestamps or historical data vintages.

The pilot uses the supplied fixed 30-stock daily panel. The course notebook refers to a broader S&P daily file and reports historical results from a larger universe, but that required daily input was not present among the supplied source files at initial inspection. Saved notebook output cannot substitute for reproducible source data.

The following limitations remain material even if every software test passes:

- The fixed, surviving stock list is not a historical point-in-time investment universe. Results cannot establish performance across delisted stocks, changing index constituents, or the wider US equity market.
- Thirty cross-sectional observations per date, together with overlapping forward returns, do not provide thirty independent observations per date. Model capacity and inference must reflect that dependence.
- Sector labels are supplied classifications. Their point-in-time validity has not been established.
- Adjusted prices support an approximate total-return experiment, not reconstructed historical exchange fills. Volume, bid/ask quotes, borrow availability, and corporate-action cash flows are not available.
- The course project previously examined this historical period. The final segment can be frozen for this prediction pipeline, but cannot honestly be described as never-seen market data.

Book value, ROE, and other fundamentals without verified publication times are excluded from the predictive information set. Broader data and fundamentals may enter only through a separately versioned experiment with documented availability rules. The original course files remain source material; the new project must not overwrite them.

## Information set, labels, and execution clock

For a decision date `t`:

1. Features use only observations available at or before the close of `t`.
2. The assumed first possible execution is the close one trading session later, `t+1`.
3. The five-session gross forward return is `adjusted_close[t+6] / adjusted_close[t+1] - 1`.
4. The training target is that forward return minus the equal-weight cross-sectional mean forward return for the same decision date.

The future cross-sectional mean is part of the label definition, not a feature. Prediction never receives a future cross-sectional return. The target is not a promise of beta neutrality. Actual portfolio neutrality and realized exposures require separate checks.

The panel's observed common trading calendar determines session offsets; calendar-day arithmetic must not substitute for trading-session offsets. The pipeline retains decision, execution, and label-end dates explicitly. Rows without complete features, an execution price, or a mature label are ineligible for the corresponding supervised calculation. Returns are not forward-filled.

Cross-sectional ranks or demeaning of features use only the current decision date's available names. Any fitted scaling, imputation, or other learned transformation belongs inside the training pipeline. Centered rolling windows, full-history normalization, and random stock-row train/test splits are prohibited.

Daily training targets overlap. A daily predictive metric is therefore not an independent-observation statistical test. Five-session rebalancing uses a globally fixed calendar so model comparisons do not select favorable start dates independently.

## Feature families

The configured families are `momentum`, `reversal`, `risk`, and `relative`, using a maximum historical lookback of 252 sessions. They extend the original project's economically motivated return, volatility, and cross-sectional research. Each implemented feature must have a direct formula and stated availability time in code or generated feature metadata.

Price history is the only raw predictive source in this pilot. Sector and same-date panel aggregates may describe relative positions but cannot be presented as an independent broad-market benchmark. Higher feature counts are not a success criterion. Adding a family after seeing an outer-test result is a new research trial and must be logged.

## Chronological development protocol

Outer test years are 2022, 2023, and 2024. For each test year, the preceding calendar year is the validation window and earlier eligible dates form the expanding training window. Every stock on a decision date belongs to the same partition. Validation selects hyperparameters; outer tests estimate the resulting selection procedure.

Before any fit, purge earlier-partition rows whose label end is on or after the next partition's first decision date. When refitting a selected model before the outer test, use only labels whose entire outcome was observable before that test boundary. Apply the same rule to transformations. The execution delay makes a generic five-row gap insufficient: the stored dates determine eligibility.

The 252-session feature warmup leaves the first 2022 test fold with approximately 2019–2020 training history and 2021 validation, subject to data coverage and purging. This is a short pilot training window. The configuration's minimum of 252 training dates and 120 validation dates is a failure threshold, not a claim of adequate statistical power. A stronger follow-on study should support at least three years of eligible training history in its first outer fold and a substantially broader historical universe.

The default final reservation starts on 1 January 2025. Default development runs must not fit, score, chart, or select using that segment's outcomes. A final evaluation requires an explicit release action and a frozen configuration, feature list, model-selection rule, portfolio rule, and trial history. Results must be labeled a frozen retrospective evaluation because of prior course exposure. Once examined, the reservation is consumed for that experiment; repeated improvements cannot reuse it as an untouched final test.

## Models and selection

The first experiment configures four fitted candidates:

- `ridge_1` and `ridge_100`: two predetermined Ridge regularization settings.
- `hgb_shallow` and `hgb_medium`: two predetermined histogram gradient-boosting capacities.

Simple momentum and reversal rules provide economic baselines alongside fitted-model comparisons. All candidates receive the same eligible data, target, decision clock, and portfolio construction rules. The tree models must not use internal randomly shuffled early stopping that bypasses chronological validation.

Within each fold, select the fitted candidate by validation mean daily Spearman rank information coefficient, with a deterministic recorded tie rule. No outer-test results may change that fold's selection. Record the score and eligibility of every candidate, including failures. The experiment tests the entire predeclared selection procedure as well as individual models.

The primary question is whether any improvement survives outside the selection window and after costs. Report the selected procedure against rules and Ridge; do not retrospectively label the best outer-test candidate as the model chosen in advance. The limited sample does not justify an unconstrained architecture or hyperparameter search.

## Portfolio construction and accounting

Use one constructor across all model and baseline forecasts. Project scores to sector-neutral, dollar-neutral target weights, target gross exposure 1.0, and absolute position weights at most 0.08. A cap adjustment must preserve neutrality; if it requires reducing gross exposure, report the reduction rather than silently violating a constraint or claiming exact unit gross. Infeasible or degenerate allocations must be explicitly handled and recorded.

Rebalance every five trading sessions on the fixed schedule. Predictions made at `t` can change holdings only at `t+1` close. Mark existing holdings to the execution close before calculating trades. Costs apply to actual traded dollar exposure, not merely the difference between consecutive target weights. Holdings drift with prices between rebalances; report that realized exposure separately from target constraints.

With adjusted closes, holdings are synthetic total-return units. Daily cash plus the value of holdings determines NAV. This is a self-financing accounting approximation under the stated price convention, not a full securities-lending or margin simulation. State cash-interest and financing assumptions in the report; do not infer historical financing or borrow availability from the input panel. Include final liquidation and its costs.

The configured base case charges 5 basis points per dollar traded, one way, and 100 basis points annualized borrow on short exposure. These are scenario assumptions, not measured implementation costs. A round trip incurs both entry and exit trading costs. Borrow accrues over actual elapsed calendar days, including weekends. The one-way trading-cost grid is 0, 5, 10, and 20 basis points; zero trading cost does not automatically mean zero borrow. Keep borrow assumptions explicit in every scenario.

Attribute gross trading performance, transaction costs, borrow, and net performance separately. Report turnover, concentration, cash, gross/net exposure, and sector exposure. Without volume and spread inputs, do not claim capacity, participation rates, market-impact calibration, or executable institutional scale.

## Evaluation and uncertainty

Report every outer fold and their chronological concatenation, including daily Spearman IC, rank-based cross-sectional diagnostics, gross and net portfolio returns, volatility, Sharpe ratio under an explicit annualization convention, drawdown, turnover, exposures, and concentration. Keep forecast metrics separate from trading metrics. Display model comparisons even when nonlinear learning adds no value.

The configured uncertainty procedure uses 1,000 moving-block bootstrap samples and 20-session blocks. Resample whole dates jointly across stocks and models; for model differences, use the same sampled blocks. Confidence intervals are conditional on this panel, specification, and research process. They do not correct survivor selection or undisclosed multiple testing. Distinguish uncertainty procedures actually executed from planned analyses.

Required robustness investigations before a strong research claim include:

- Paired fitted-model versus baseline comparisons and contributions by year, sector, and name.
- Execution delayed by one additional session and the configured cost scenarios.
- Feature-family ablations whose selection is confined to development data.
- A label-permutation negative control preserving date groups while breaking the stock-to-outcome relationship, with its exact construction and seeds recorded.
- Sensitivity to concentrated profit days or names and to the chosen rebalance phase.

These analyses must be logged as trials. A negative control need not produce precisely zero IC in a small sample; unusually strong control performance is a diagnostic requiring investigation. Hypotheses, thresholds, and comparisons must be set before consuming the final reservation.

## Reproducibility and research artifacts

Persist input file hashes, source locations, row/date/universe coverage, configuration, random seed 1729, feature list, partition boundaries and purge counts, candidate scores, model identity, execution settings, dependency versions, and exact run command. Record every completed or abandoned experiment in a trial ledger with its relationship to the frozen configuration. Generated results must identify their input and configuration hashes.

The deliverable is an inspectable chain from source data to forecasts, holdings, costs, and reported conclusions. Reproduction must not depend on an executed notebook's saved state or undeclared local files. Reports must show data or validation failures, not quietly substitute synthetic inputs for the source experiment.

The forecasting runner can optionally persist the already selected, refitted estimator for each fold and family as an atomic local joblib bundle. Metadata includes the exact feature order, universe and sector map, training cutoff, latest fitted label end, configuration hash, and Python/package versions. `predict_asof` scores an observed close from historical features alone: it requires no future label, rejects dates before the fitted model becomes usable, and verifies that every fitted label matured before the decision. Cross-sectional universe changes require a new specification. These are trusted local research artifacts and research forecasts; model serialization and as-of scoring do not establish a live trading deployment, execution connectivity, operational monitoring, or prospective profitability.

## Claim gates

Passing software checks supports correctness of the implemented assumptions. Reproducing a historical result supports reproducibility. Incremental predictive evidence requires outer-test improvement with defensible uncertainty and robustness. A credible live strategy additionally requires genuinely forward predictions, executable data, realistic financing/borrow, and monitored operational controls.

Until the relevant evidence exists, describe the project as a research platform and limited-data pilot. Reserve terms such as "production ready," "institutional capacity," and "validated alpha" for the gates they actually require. Good engineering and honest negative findings remain valuable deliverables.
