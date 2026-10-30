# Findings V2: aligned targets and explicit abstention

Study: `sector_alpha_v2`. Run: `artifacts/sector-alpha-v2-development`.
V2 was declared after observing V1 development results. It is a new research trial,
not a retrospective improvement to V1. The source remains the fixed 30-name pilot;
2025 onward was not evaluated. All figures here are development findings.

## What the experiment tests

The label is the five-session forward stock return minus its same-date sector
mean, with close-t features, assumed close-t+1 entry and close-t+6 label maturity.
This matches the subspace traded by the sector-neutral allocator. Ridge and
boosting predict returns in decimal units. Inner validation minimizes date-averaged
sector-relative MSE; deployment additionally requires beating zero-forecast MSE
and positive sector Rank IC. The selected model is refitted only on mature labels.

All original feature groups and four fitted candidate specifications remain.
Feature deletions shown in the report are diagnostics, not newly selected models.
Ungated forecasts are retained to distinguish prediction from deployment policy.

## Primary predictive finding

There is no demonstrated nonlinear forecasting advantage in this run.

| Ungated family | Mean MSE, bp² | Zero-forecast MSE, bp² | Out-of-sample MSE skill vs zero |
| --- | ---: | ---: | ---: |
| Ridge | 41,150.5584 | 41,119.8726 | -0.0746% |
| Gradient boosting | 41,235.7829 | 41,119.8726 | -0.2819% |

MSE skill is `1 - model_MSE / zero_MSE`; it is predictive, not an in-sample
asset-pricing regression R². Negative skill means the model has higher squared
forecast error than the zero residual-return forecast.

The paired tree-versus-Ridge MSE reduction is **-85.2246 bp²**, with a 95% circular
20-session date-block bootstrap interval **[-250.1614, 71.1470] bp²** across 747
fully labeled decision dates. Positive reduction would favor trees. The interval
includes zero; it does not establish incremental nonlinear predictive value.

These are conditional development intervals, not corrections for prior research
iterations, survivor selection, or static sector-classification uncertainty.

## Deployment versus raw-model performance

Every family and the cross-family selected procedure failed its validation
deployment hurdle in all three outer years. The predeclared gated policies held
cash. Zero turnover and zero market P&L are the expected result of those decisions;
a reported Sharpe of zero is the software's degenerate-cash convention, not a
positive investment-performance claim.

The ungated models remain visible:

| Ungated family | Mean sector Rank IC | Net Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Ridge | about 0.0030 | -0.1882 | -7.24% |
| Gradient boosting | about 0.0078 | -0.5822 | -9.85% |

These values include 5 bp one-way assumed trading costs and 100 bp annual assumed
short borrow, with synthetic adjusted-close accounting. A slightly positive rank
correlation does not establish calibrated forecasts or positive net economics.
Cash gating must not conceal the failed raw-model evidence.

## Wider-data qualification

The public Qlib archive was downloaded and audited separately, not substituted
for this study. Its actual calendar is 1999-12-31 through 2020-11-10. Its file
named `sp500.txt` contains 745 historical symbol labels and 755 spans, but is not
a verified historical index-membership dataset.

Two explicit vendor punctuation mappings recover 10,500 member-date quotes.
Coverage rises from 80.5774% to 80.9714%; 87 labels still have no close data and
141 membership exits lack a next-session close. Missing dead/acquired companies
are not silently removed and missing exit values are not carried forward.

A primary-source check also establishes a concrete membership error: the archive
starts BRK.B membership in 1999, whereas Berkshire's January 2010 announcement
describes its new addition to the S&P 500. See
[issuer announcement](https://www.berkshirehathaway.com/news/jan2810.pdf) and
[public source investigation](PUBLIC_DATA_SOURCES.md).

The archive fails both quote-coverage and membership-quality qualification. The
audit command saves its evidence and exits with status 2. It is not accepted as
a survivor-complete training or backtest dataset. The complete local audit is
`artifacts/qlib-source-qualification.json`.

## Reproducibility and remaining evidence

The completed run's independent verifier checked 115 hashed artifacts, 49 frozen
source files, normalized configuration, and the original input bytes. The registry
contains separate `run_started` and `run_completed` events. Local hashes establish
internal consistency, not externally immutable provenance.

The study's configuration hash is
`fe79a0f01cdb43cbae5eadde9207812264298ee41c1a9357af231f9504968654`.
The original input hash is
`db470f1ecce2b3da70c837c6b10c227c3af40c637f939c8bac48c992bfffcedd`.
The artifact-inventory hash is
`3fd9324cd39339937920d017b1be898a2d95093f90b0c61ba242fa2d29ed0387`.

This findings document was written after the completed run and is not represented
as part of its frozen protocol snapshot. Synthetic positive/null controls are
separate engineering experiments and cannot turn these negative equity findings
into empirical alpha evidence.

The substantive next requirement is a credible broader historical stock panel
with stable security identifiers and terminal-return handling. More model
complexity on the current fixed 30-name sample does not resolve that limitation.
The required input contract is in [Data requirements](DATA_REQUIREMENTS.md).
