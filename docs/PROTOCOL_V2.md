# V2: align return prediction with the tradable sector-neutral component

Protocol frozen on 29 October 2026 before executing the revised procedure.
Experiment identifier: `price_alpha_v2`. The numerical results and diagnostics
of `price_alpha_v1` on 2022–2024 were already observed. This revision is an
additional research trial motivated by those observations. Its subsequent
development results cannot restore those years to an unseen evaluation sample.
This document specifies methods, not a forecast of favorable numerical results.

## Research hypothesis and the economic object

Can shallow nonlinear learning improve prediction of the five-session
within-sector component of stock returns relative to Ridge, and can either
learned procedure support a useful delayed sector-neutral portfolio after costs?

At decision close `t`, the raw outcome for stock `i` is:

`r_i,t = adjusted_close_i[t+6] / adjusted_close_i[t+1] - 1`.

The revised supervised target is:

`y_i,t = r_i,t - equal_weight_mean(r_j,t for j in sector(i))`.

For sector-neutral weights, `sum_i w_i r_i = sum_i w_i y_i`. The label now
describes precisely the return component a sector-neutral portfolio can capture.
Future sector means define training outcomes only; they never enter features,
decision-time forecasts, risk estimates or allocations. Static sector labels
remain a limitation of the supplied data, not verified historical classifications.

If a sector/date has an unavailable return, every relative label in that group
is unavailable; the target must not silently switch to a smaller future universe.
A complete singleton sector has a zero relative target and zero projected
forecast. It offers no within-sector trade and is excluded from sector Rank IC.

All V1 feature families, raw data sources, execution delay, horizon and the four
candidate estimators remain declared. The favorable V1 `without_risk` ablation
does not authorize removing risk features in this trial. No additional
hyperparameter search is introduced by this protocol.

## Forecasts, selection and abstention

Ridge and histogram gradient boosting estimate decimal sector-relative returns.
Every prediction is projected by subtracting the same-date sector mean before
selection diagnostics and portfolio construction. Projection uses only forecasts
and known sector labels. It does not require future outcomes.

Use the existing expanding annual training / preceding-year validation / outer
test folds, with every stock on a date in the same partition. Keep the label
maturity purge and fitted-transform boundaries. All score comparisons use the
same eligible rows and the same decision dates.

For each candidate, compute the equally weighted mean of its date-level
cross-sectional squared errors on validation data. Each date first averages
across eligible names; dates receive equal weight. Choose the smallest validation
MSE within each family and across families. Exact ties follow the declared
configuration order. These comparisons preserve return units and do not compare
uncalibrated factor ranks with return predictions.

For each chosen family procedure and the across-family selected procedure,
require both of the following before enabling its outer-year positions:

1. Its mean validation MSE is strictly smaller than that of zero return forecasts
   on the identical target observations.
2. Its mean validation sector Rank IC is strictly positive.

Otherwise the procedure produces zero portfolio forecasts for the outer year.
There is no sign reversal or selection of the least negative score. Record the
chosen candidate, validation MSE, zero MSE, sector Rank IC, both gate decisions,
and the explicit cash reason. The gate has no tuned positive threshold and makes
no claim of statistical significance. Passing it does not certify positive
expected net performance.

Refit the chosen estimator using all mature pre-test labels as in V1. Freeze the
model, gate result, feature schema, sector map and configuration in its artifact.
Retain pre-gate family predictions for predictive assessment even when the
implemented family portfolio remains cash. Cash deployment and predictive model
quality are distinct objects: evaluating zeros alone would hide model failures.

The rule-based momentum, reversal and composite scores remain portfolio and
rank-diagnostic baselines. Their arbitrary rank units are not calibrated expected
returns, so MSE skill comparisons and return-unit allocation objectives must
exclude them unless a separately specified chronological calibration is added.

## Primary predictive diagnostics

The primary nonlinear comparison is paired date-level MSE difference between
pre-gate HGB and Ridge forecasts on the same outer observations. Report the sign
convention, absolute MSE and skill relative to zero. Use common date blocks for
confidence intervals; individual stocks and overlapping forward labels are not
independent trials.

The aligned rank statistic is daily Spearman correlation between projected
forecast scores and sector-relative outcomes. This is one cross-sectional
correlation per date, not an average of tiny-sector correlations. Flat forecasts
have IC zero. Flat outcomes or fewer than three eligible names have undefined
IC. Also retain V1-style global IC as a secondary mechanism diagnostic with an
explicitly different interpretation.

Report each fold separately, including failed gates, and date-block uncertainty.
A higher Rank IC can coexist with worse MSE when forecast return magnitudes are
miscalibrated. Neither metric alone establishes after-cost economic usefulness.

## Portfolio interpretation and costs

Use shared sector-neutral constraints, execution timing and accounting across
the fitted models. The source adjusted closes remain synthetic total-return
units; no auction fills, capacity, impact or borrow availability are observed.
Costs are scenario assumptions and are not fit to outer returns.

A shared allocator does not imply identical gross exposure, concentration,
turnover, volatility or beta across models. Report those quantities and avoid
attributing their combined P&L differences entirely to predictive nonlinearity.
The paired forecast comparison supplies evidence about model quality separately
from the implemented portfolio comparison.

If using a cost-aware allocator, return forecasts and covariance risk must use
the same holding-period units. Decisions may depend only on prices, covariance
estimates and existing holdings known at close `t`. In particular, the optimizer
must not use the realized `t+1` prices or holdings drift to construct a decision
dated `t`. The cash ledger charges costs against actual execution-date notional;
forecast cost estimates and realized accounting charges need not coincide.

No validation net-return or net-Sharpe hurdle is added to this version's model
gate. Realized net economics, extra-session execution delay and the predeclared
cost grid remain diagnostics. Adding an economic gate later would create another
recorded trial, even if motivated by an apparent improvement in these results.

## Negative controls and failure investigations

Retain all feature families and log every candidate, gate, ablation and control.
A within-date-and-sector outcome permutation may break stock alignment while
preserving the revised label structure. Any revised control construction is
declared in the trial ledger. A few random controls are leakage diagnostics,
not a formal calibrated null distribution or a permutation p-value.

Hand-worked tests must establish the sector-label formula and sector-neutral
P&L identity, temporal independence across decision dates, constant and singleton
behavior, complete-observation requirements, score/target alignment and MSE
units. Keep the earlier feature-causality, maturity-purge and self-financing
ledger checks. Test that failed gates produce cash and cannot access outer labels.

## Data, reservation and claim limits

The fixed 30-survivor sample, absent membership history, unavailable delisting
returns, retrospective price adjustments and static classifications remain
unchanged. Improving target alignment cannot resolve these limitations.

Keep 2025 onward reserved and unconsumed by this development iteration. Never
fit, select, chart performance or evaluate these outcomes until a separately
authorized frozen evaluation. The prior course exposure still requires calling
that future release a reserved retrospective evaluation, not never-seen history.

Persist the protocol and configuration hashes, source snapshot, complete trial
ledger, all validation gate values, model artifacts, forecasts and accounting
outputs before drawing conclusions. Genuine method alignment is the motivation;
changed targets, selection, abstention or allocation are new research choices,
not retroactive repairs of V1 performance.
