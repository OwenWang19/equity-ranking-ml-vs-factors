# Adversarial review checklist

This is an acceptance checklist, not a record of completed checks. Mark an item complete only with a test, artifact, or explicit documented decision. Applicable settings are in `configs/research.json`; the research assumptions are in `RESEARCH_PROTOCOL.md`.

## Data and availability

- [ ] Source hashes and row/date/ticker coverage are recorded; source files remain unchanged.
- [ ] Duplicate stock/date rows, nonfinite values, invalid prices, and inconsistent returns fail visibly.
- [ ] Missing observations and eligibility changes are reported; returns are never forward-filled.
- [ ] The fixed 30-name survivor universe and lack of historical membership are stated beside results.
- [ ] Supplied sector labels are not claimed to be historically point-in-time classifications.
- [ ] No book value, ROE, or other timestamp-uncertain fundamental enters predictive features.
- [ ] The missing broader S&P daily source is not replaced by saved notebook outputs.
- [ ] Adjusted prices and synthetic total-return holdings are distinguished from executable share prices.

## Temporal integrity

- [ ] Each row records decision, execution, and label-end sessions.
- [ ] Features at close `t` use no observations after `t`; the first trade is at `t+1` close.
- [ ] Five-session labels span `t+1` close to `t+6` close, with cross-sectional demeaning confined to the target.
- [ ] A perturbation of future source data leaves all eligible earlier features and decisions unchanged.
- [ ] Training, validation, and test assign entire decision dates together.
- [ ] Purging uses actual label-end dates and removes outcomes crossing the next partition boundary.
- [ ] Refitting and all learned transformations use only labels and information mature before the test boundary.
- [ ] Internal tree early stopping does not introduce a random time split.
- [ ] Default development runs exclude the reserved period beginning 1 January 2025 from model fitting, scoring, and selection.
- [ ] Final evaluation requires explicit release, records the frozen specification, and is labeled retrospective given prior course exposure.

## Research selection

- [ ] All configured candidate scores and failures are saved, not only the winner.
- [ ] The validation mean daily Spearman IC selection rule and deterministic ties are recorded.
- [ ] Simple momentum/reversal and Ridge receive the same clocks, eligible rows, and portfolio rules as boosting.
- [ ] Outer-test results did not alter selection for that fold.
- [ ] The short first-fold history is disclosed; the 252-date minimum is not described as adequate statistical power.
- [ ] Every feature, parameter, universe, cost, and allocation change is recorded in a trial ledger.
- [ ] Any reported "best" result identifies whether it was chosen by validation or inspected after testing.

## Portfolio and accounting

- [ ] Sector and dollar neutrality survive position-cap enforcement at rebalance.
- [ ] The 8% absolute target-weight cap and gross-exposure target are checked together; reduced gross is reported when necessary.
- [ ] Realized drifting exposures are reported separately from target weights.
- [ ] All models use the same globally fixed five-session rebalance schedule.
- [ ] Previous holdings are marked to the execution close before trade sizes and costs are calculated.
- [ ] A hand-worked two-asset long/short path agrees with daily NAV and cash accounting.
- [ ] A constant-price round trip loses the entry/exit fees and any short borrow expected under the assumptions.
- [ ] Transaction costs use actual dollar trades; round trips charge both sides.
- [ ] Borrow charges elapsed calendar days, including weekends, and is not accidentally removed in the zero-trading-cost case.
- [ ] Final liquidation and its costs are included.
- [ ] Cash-interest, financing, dividends/adjusted-price treatment, and borrow-availability assumptions are explicit.
- [ ] Gross return, trading fees, borrow, and net return reconcile to NAV.
- [ ] Infeasible allocations, absent eligible stocks, and degenerate scores fail or fall back visibly.

## Statistical evidence

- [ ] Predictive IC and portfolio profitability are reported as separate measurements.
- [ ] All 2022, 2023, and 2024 folds appear, including unfavorable results.
- [ ] Results identify whether the reported series is each model or the predeclared model-selection procedure.
- [ ] Bootstrap samples whole date blocks jointly; paired model comparisons use identical sampled blocks.
- [ ] Confidence intervals disclose 1,000 samples, 20-session blocks, and their conditional limitations.
- [ ] Contributions by date, sector, and stock reveal concentrated sources of performance.
- [ ] Cost assumptions of 0/5/10/20 bps one way are labeled scenarios rather than calibrated fills.
- [ ] Extra execution delay, rebalance-phase sensitivity, and feature ablations are executed and logged before strong robustness claims.
- [ ] A documented label-permutation control checks whether the pipeline can manufacture apparent signal.
- [ ] Multiple research trials and overlapping labels are acknowledged; stock-date rows are not treated as independent evidence.
- [ ] No capacity, market-impact, delisting resilience, or live-alpha claim is inferred from the limited panel.

## Reproduction and communication

- [ ] A clean-environment command reproduces outputs without notebook execution history or undeclared inputs.
- [ ] Input/configuration hashes, code revision when available, seed, versions, boundaries, purge counts, and commands accompany results.
- [ ] A reviewer can trace one prediction into its target holdings, executed trades, costs, and subsequent NAV.
- [ ] Test results state what was actually verified; planned checks remain visibly pending.
- [ ] The final report distinguishes an engineering pilot, retrospective predictive evidence, and a forward executable strategy.
- [ ] Negative findings and unresolved limitations are retained in the report.
- [ ] "Production ready" or "validated alpha" is used only after the corresponding research and operational gates are met.

## Questions a senior reviewer should be able to answer

1. What information was available at each decision, and how can I falsify the claimed timing?
2. Why should any result generalize beyond these selected surviving stocks?
3. Does boosting improve on Ridge and simple factors after matching exposures, turnover, and costs?
4. Which decisions were made before viewing each test segment, and how many alternatives were tried?
5. Which names, sectors, dates, or accounting assumptions explain most of the result?
6. What breaks when execution is slower, borrow is unavailable, or costs are higher?
7. What remains untested before a prospective paper or live implementation?
