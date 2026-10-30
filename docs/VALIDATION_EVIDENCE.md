# Validation evidence and its scope

These checks establish properties of the implemented experiment, not expected
financial profitability. The real-data findings, synthetic engineering controls,
and public-data qualification are separate evidence streams.

## Engineering recovery and abstention

The first frozen synthetic benchmark used seed 1729, 12 synthetic names, two
synthetic sectors, 1,500 business sessions and predeclared innovation scales.
The planted idiosyncratic log-return process has AR coefficient 0.85; the null
coefficient is zero. The same price-only feature builder, delayed labels, purged
folds, model families and cash rule are used. Neither generator parameters nor
seeds were adjusted after results.

| Artificial case | Model | Forecast MSE skill versus zero | Mean sector Rank IC | Deployment gates |
| --- | --- | ---: | ---: | --- |
| Planted persistent signal | Ridge | +31.24% | 0.5325 | Active in both folds |
| Planted persistent signal | Boosting | +31.05% | 0.5300 | Active in both folds |
| Independent null | Ridge | -0.495% | -0.0168 | Cash in both folds |
| Independent null | Boosting | -0.511% | +0.0249 | Cash in both folds |

The positive null IC is particularly useful: a single finite sample can show
positive IC without reducing forecast MSE. The gate correctly rejects that case.
One fixed null realization is not a null distribution, significance calibration,
or guarantee that the gate will reject every false model.

Original artifacts are in `artifacts/synthetic-validation-v1/`: summary, generator
parameters, source/config hashes, both case forecasts, trial ledgers, split audits
and selected model bundles. The module is `equity_alpha.synthetic_control`.
These are artificial controls and must never be reported as financial returns,
stock-market prediction accuracy or an improvement to real-data findings.

## Temporal and model lifecycle checks

- Future-price perturbations cannot change earlier features or complete matured
  labels; actual nested model fits retain earlier selections and predictions.
- Each decision date's stocks share its partition. Training and refit labels must
  mature before the next boundary; labels are close t+1 to close t+6.
- Scalers fit training samples only. Gradient boosting does not create a random
  internal early-stopping validation split.
- Frozen model bundles reproduce gated historical decisions. As-of inference uses
  historical features without forward labels and rejects incompatible universe,
  feature schema, model version or training maturity.
- Pre-gate prediction errors remain available even when deployment is cash.

See `tests/test_features_splits.py`, `tests/test_v2_experiment.py`,
`tests/test_inference.py` and `tests/test_synthetic_control.py`.

## Economic accounting and controls

Hand-worked cases independently check cash and holdings, delayed execution,
drift, actual traded-notional costs, post-fee target sizing, calendar-day borrow,
sign flips and final liquidation. A flat-price book loses only costs. Target
neutrality and position caps are separate from exposure drift between trades.

The real-data report shows assumed trading/borrow costs, pre-gate forecasts,
primary paired prediction-error uncertainty, secondary trading outcomes, negative
controls and diagnostic ablations. No model is promoted by selecting the best
displayed outer-test ablation after observing it.

See `tests/test_portfolio.py`, `tests/test_evaluation.py`, and
`docs/FINDINGS_V1.md` / `docs/FINDINGS_V2.md`.

## Data and reproducibility checks

The supplied pilot's 65,820 comparable daily returns reconcile to its adjusted
price series. This is an internal consistency test, not a vendor certification.
The broader Qlib archive fails explicit quote-coverage and historical-membership
qualification; its failure is retained rather than bypassed.

Frozen real-data runs preserve code/tests/protocol snapshots and exact inventories.
`equity-alpha verify` checks every recorded artifact, source inventory,
configuration and optional original data. The locked local registry records
start/completion/failure transitions and protects single experiment holdout-release
semantics. Hashes support internal consistency; externally immutable records are
still needed to prove against deliberate wholesale replacement or truncation.

See `tests/test_data.py`, `tests/test_qlib_data.py`, `tests/test_integrity.py`,
`tests/test_registry.py` and `tests/test_cli.py`.

## What remains outside this evidence

The experiments do not establish a tradable edge, survivor-complete broad-stock
generalization, executable auction fills, calibrated impact/capacity, point-in-time
fundamentals, a live operational deployment or future profitability. The current
data contract and the unconsumed 2025-onward reservation remain substantive gates.

The repository includes an Ubuntu CI matrix for Python 3.11 and 3.14. Configuration
of that workflow is not a claim that remote CI or both interpreter environments
have already passed; local validation is recorded separately.
