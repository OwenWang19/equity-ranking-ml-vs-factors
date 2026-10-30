# Findings V1: a functioning research experiment with weak economic evidence

Review date: 29 October 2026. Experiment: `price_alpha_v1`. Reviewed run: `artifacts/development-v2`. This is a development finding, not a final-holdout or live-strategy result.

The current experiment does **not establish that nonlinear learning improves on Ridge regression**. Histogram gradient boosting has a slightly higher realized net Sharpe, but its paired uncertainty interval is much too wide to support a reliable advantage. Learned portfolios outperform the specified factor composite, which loses money in this sample. That is a narrower result than establishing a profitable or deployable alpha strategy.

## Evidence reviewed

All numerical findings below come from these local pilot artifacts. They are
retained in the original workspace, not distributed in the public repository;
the two separately reviewed broad-security releases are the published evidence.

- `artifacts/development-v2/summary.csv`: portfolio and predictive aggregates.
- `artifacts/development-v2/paired.csv`: paired circular date-block bootstrap comparisons.
- `artifacts/development-v2/folds.csv`: calendar-year predictive and realized portfolio slices.
- `artifacts/development-v2/trial_ledger.csv`: candidate selection, negative controls, and ablation roles.
- `artifacts/development-v2/manifest.json`: data, configuration, software, source hashes, and holdout status.

The realized portfolio sample is 752 sessions, 3 January 2022 through 30 December 2024. The source panel has 30 names and 2,195 dates from 2 January 2018 through 25 September 2026. Source coverage is not the evaluated return sample: the manifest says `development_walk_forward_only`, with holdout performance beginning in 2025 not evaluated.

## Aggregate result

| Procedure | Mean daily Rank IC | Net Sharpe | Annualized net return | Total net return | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Factor composite | -0.01743 | -1.4766 | -6.25% | -17.53% | -19.15% |
| Ridge family | 0.00988 | 0.2175 | 0.78% | 2.35% | -5.45% |
| Boosting family | 0.01545 | 0.2618 | 0.85% | 2.55% | -4.30% |
| Validation-selected procedure | 0.01151 | 0.1172 | 0.37% | 1.11% | -6.08% |
| Zero-exposure control | 0.00000 | 0.0000* | 0.00% | 0.00% | 0.00% |

`*` The software represents the degenerate cash control's undefined return-to-volatility ratio as zero. That convention is not an estimated risk-adjusted investment performance.

The base assumptions are a one-session execution delay, five-session holding/rebalance interval, 5 bps per dollar traded one way, and 100 bps annual assumed short borrow. Cash earns zero. These are synthetic adjusted-close total-return portfolios with a target gross ceiling of one and an 8% target position cap. Costs and borrow are assumptions, not observed execution or lending quotes.

The observed net Sharpe values of approximately 0.12–0.26 for the learned procedures are economically weak. Positive realized cumulative returns alone do not demonstrate a stable alpha. The factor composite is also a weak hurdle here: improving on a strategy that loses 17.5% is different from establishing positive expected net return.

Costs materially reduce the observed result. Boosting's reported gross diagnostic Sharpe is 0.8287 versus a net Sharpe of 0.2618. Its transaction fees total approximately 4.87% of initial NAV and short-borrow charges approximately 1.03% of initial NAV over the entire sample. These are accumulated cash charges, not annual return estimates or an independently simulated no-cost portfolio. Annualized two-sided gross traded-notional turnover is approximately 32.13 times NAV for boosting and 34.94 times NAV for Ridge.

## Nonlinear versus linear evidence

Paired intervals use 1,000 circular date-block resamples with 20-session blocks and common date samples within each comparison:

| Comparison: net Sharpe difference | Estimate | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
| Boosting minus Ridge | 0.0442 | -1.2548 | 1.2999 |
| Selected procedure minus factor composite | 1.5938 | 0.3495 | 3.0997 |
| Boosting minus factor composite | 1.7384 | 0.3361 | 3.1438 |

The first interval includes large differences of either sign. The experiment does not establish nonlinear superiority, and the approximately 0.044 realized difference should not be promoted as a meaningful improvement on its own.

The two comparisons with the composite have positive lower endpoints under the stated procedure. They support an advantage relative to this particular losing composite within this development sample and implementation. They do not establish positive expected standalone Sharpe, solve universe selection bias, or adjust for an unknown total number of research trials.

The shared portfolio construction rule does not produce identical risk or gross exposures: boosting's average realized gross exposure is 0.6795 versus Ridge's 0.8069 and the selected procedure's 0.7533. The intervals compare these actual implemented procedures, not a hypothetical equal-risk pure forecasting experiment. Further model-mechanism claims would require appropriate exposure and turnover diagnostics rather than attributing the entire difference to nonlinear prediction.

## Chronological behavior and selection

| Calendar year | Ridge IC | Ridge net return | Boosting IC | Boosting net return | Selected net return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | -0.02019 | -1.93% | -0.01693 | -0.27% | -1.93% |
| 2023 | 0.03423 | 3.14% | 0.03910 | 1.68% | 1.54% |
| 2024 | 0.01582 | 1.18% | 0.02445 | 1.13% | 1.54% |

The sign of IC changes across years. Neither learned family has consistently positive predictive performance across all three periods. Higher boosting IC in 2023 does not translate into a higher realized return than Ridge, illustrating that cross-sectional forecast quality, portfolio exposures, costs, and realized trading performance are different measurements.

Calendar-year portfolio figures are slices of a continuing account. Positions and costs are not reset on 1 January. Consequently, two procedures using the same new-year forecast model can still have different calendar-year returns because their inherited holdings differ.

The ledger identifies the predeclared selected procedure as:

- 2022: `ridge_1`, validation mean Rank IC -0.00619.
- 2023: `hgb_shallow`, validation mean Rank IC -0.02324.
- 2024: `ridge_100`, validation mean Rank IC 0.03568.

The first two validation winners still have negative validation IC. The selection rule chooses the best configured candidate and contains no predeclared abstention or cash hurdle. That fact is relevant to the weak realized selected-procedure performance. Testing a cash hurdle could be a subsequent hypothesis, but adding one after inspecting these outcomes would create another research trial; it cannot retroactively improve the frozen experiment.

## Controls and ablations

All three within-date label-permutation controls have negative aggregate net Sharpe, approximately -0.851, -0.288, and -0.696. Their aggregate mean Rank IC values are approximately -0.00058, 0.01062, and 0.00418. A null run can therefore display positive IC: the second permutation's mean IC is slightly above Ridge's mean IC in this sample. Three runs are diagnostic examples, not a calibrated null distribution or formal permutation p-value. They cannot establish that all remaining leakage or chance explanations have been eliminated.

The strongest displayed ablation is `without_risk`, with net Sharpe 0.6681 and annualized net return 2.25%, but mean Rank IC only 0.00060. This disagreement is a research question about portfolio mapping, exposures, and sampling variation. It does not by itself establish that risk features are harmful, that the ablation is superior, or that it should replace the chosen model.

The ledger explicitly marks leave-family-out runs as `prespecified_ablation` with `diagnostic_only` selection status. No paired superiority interval for that ablation is included in the evidence reviewed here. Promoting the best outer-test ablation after seeing its result would be post-test model selection. Any subsequent change must be recorded as a new hypothesis and evaluated under a revised, prospectively frozen procedure.

## Completed evidence versus remaining gates

The reviewed artifacts establish that this run produced a full chronological development evaluation, saved candidate decisions and controls, estimated paired uncertainty, and retained explicit data/configuration/source provenance. The input audit reports 65,820 price-return comparisons with zero discrepancies above a tolerance of `1e-8`; its maximum discrepancy is approximately `5.43e-16`. This verifies internal consistency of the supplied adjusted-price and return series, not external vendor accuracy or point-in-time availability.

The manifest records seed 1729, Python 3.14.3, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, and scikit-learn 1.9.1. It records that controls were enabled and final-holdout performance was not evaluated. Reproduction and timing/accounting tests are separate engineering evidence; numerical tables alone do not prove all code paths correct.

The principal remaining gates are substantive:

1. Establish a credible historical investment universe, including delistings and membership changes, and increase the cross-sectional breadth. The current fixed survivor sample cannot support broad-market generalization.
2. Obtain publication-dated data before using fundamentals. Static sectors, retrospective price adjustments, and the lack of volume, spreads, and borrow quotes remain limitations.
3. Diagnose weak net economics and the mismatch between IC and traded outcomes. Log any changes to features, selection, risk mapping, rebalance phase, or costs as additional trials.
4. Require stronger evidence for a nonlinear improvement and positive expected net performance. Three development years, wide paired intervals, and a short initial training history leave substantial uncertainty.
5. Freeze any eventual final specification before consuming the reserved 2025-onward evaluation. Because source prices were already used in coursework, describe it as a reserved retrospective evaluation rather than never-seen historical data.
6. Log genuinely forward predictions and assess implementability, financing, borrow availability, and operational behavior before asserting a live edge or production readiness.

The current project can demonstrate auditable research engineering and honest interpretation of a limited-data experiment. It should not be represented as a validated trading alpha, a production trading system, or evidence of institutional capacity.

## Run history and provenance

`development-v1` is retained as the unsuccessful first run. It reached forecast generation and then encountered a pandas compatibility failure involving DataFrame metadata (`attrs`) during analytics concatenation. `development-v2` applies the analytics compatibility correction. The maintainer verified that v1 and v2 forecast outputs are unchanged; v2 is a repaired completion of the same predictive experiment, not a favorable replacement model chosen after testing. That operational verification is recorded here separately from the numerical CSV evidence.

The configuration experiment identifier remains `price_alpha_v1`. Its normalized configuration SHA-256 is:

`c8f4f2d9d1f01ec27a1998cb889e350ce73fe3f1a8b2fa05c2b5fe2263eea4f7`

The source daily-data SHA-256 recorded by the manifest is:

`db470f1ecce2b3da70c837c6b10c227c3af40c637f939c8bac48c992bfffcedd`

For this written review, the exact inspected artifacts have the following SHA-256 values:

| Artifact in `artifacts/development-v2` | SHA-256 |
| --- | --- |
| `summary.csv` | `af38ecd44157f0c21d2597522d263d60d40a819ab2b9b7c8f8c74660c81547fa` |
| `paired.csv` | `a74f993dccc8b195099f2fb5fc6ee768cd64829c825ea8e62b454ffb1fa53f4a` |
| `folds.csv` | `0f891f6d6d3295e807cb5cc192c8d1729531cdd34b8685ad2fb4c2af39011e3e` |
| `trial_ledger.csv` | `d68ab64a72d8fbe815bd6e328f3076cb0fb805ab8c904b2fd0873d267e585483` |
| `manifest.json` | `e37be06d723ce12f218846d79e9e69b786f24b72882e4b7cb581ee27b97fc4b3` |

The manifest also records per-source-file hashes, including the analytics implementation used for the completed run. This findings document was written after the run and is not claimed to have been part of the original run's frozen source manifest.
