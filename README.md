# AxiomAlpha

**Cross-Sectional Equity ML & Signal Validation**

An auditable research engine for testing nonlinear equity forecasts against
simple factors, with leakage controls, missing-outcome diagnostics and preserved
negative findings. The Python package remains `equity_alpha` for compatibility.

**Research question:** Does nonlinear machine learning improve five-session
cross-sectional equity forecasts beyond momentum, reversal and a regularized
linear model under delayed labels? The broader study is prediction-only; the
earlier fixed-panel pilot separately explores turnover costs and short borrow.

This project extends an FE5108 factor-research study into a predictive research
workflow. It contains a fixed 30-stock pilot and a separately registered,
broader-security prediction study. Institutional-scale alpha, execution capacity
and production readiness must be earned through additional evidence.

Start with the [reviewer entrypoint](RESEARCH_REVIEW.html). The bounded research
deliverable is complete when its methods, negative findings, tests and evidence
are reproducible and inspectable—not when a favorable alpha statistic appears.

The earlier pilot, **Sector Alpha V2**, forecasts industry-relative
returns, selects return-unit models by validation MSE, and deploys cash unless
the winner beats zero-forecast MSE and has positive sector-relative validation
Rank IC. Ungated predictions remain in the primary forecast comparison. This is
a new, explicitly recorded hypothesis following V1 development results, not a
retroactive correction to the original study. See [V2 protocol](docs/PROTOCOL_V2.md).

## Latest evidence: broad-security ML development

**Wide ML V1 completed on 2026-10-30. No robust standalone ML advantage was
established.** The fixed HGB7 model improves daily Rank IC over a negative Ridge100
baseline, but its own confidence interval crosses zero, one shuffled-label
control has a higher mean, and the simple five-session reversal baseline is
stronger. This is a preserved research finding, not a profitable-strategy claim.

| Development forecast | Mean daily observed-case Rank IC | 95% calendar-block interval |
| --- | ---: | ---: |
| Fixed HGB7 | 0.00165 | [-0.03179, 0.03315] |
| Fixed Ridge100 | -0.00902 | [-0.04250, 0.02410] |
| Five-session reversal | 0.02729 | [0.00790, 0.04877] |
| Fixed HGB7 minus Ridge100 | 0.01067 | [0.00238, 0.01833] |

Read the [reviewed report](artifacts/wide-ml-v1-development/report.html),
[all model/control statistics](artifacts/wide-ml-v1-development/summary.json),
and [release interpretation](artifacts/wide-ml-v1-development/release_review.json).
The original frozen report is retained alongside the reviewed interpretation.

The separate [incremental-information diagnostics](artifacts/wide-incremental-diagnostics-v1/report.html)
use the same frozen forecasts, zero new model fits, and an explicitly post-hoc
[analysis protocol](configs/wide_diagnostics_v1.json). They compare HGB7 with
reversal, momentum and Ridge, measure descriptive partial rank correlation after
projecting both score and outcome ranks on reversal/momentum, and retain both
years, all liquidity/volatility conditions and fixed block sensitivities. These
are exploratory, observed-case statistics—not tradable residual forecasts or
selection/multiplicity-corrected confirmation. No subgroup or model is selected
for deployment.

The fixed HGB7-minus-reversal5 mean IC difference is **−0.02564** (95% block
interval [−0.05577, 0.00346]). Its descriptive partial rank correlation controlling
reversal/momentum is **−0.00407** ([−0.03093, 0.02092]); no reliable positive
increment was established. One pre-summary diagnostic attempt hit a pandas
datetime-resolution comparison error. A unit-normalization fix and regression
test preceded one explicitly logged same-protocol retry; both attempts remain
recorded and no prediction model was refitted.

Portable checks require no vendor account or data and never execute snapshots:

```bash
uv sync --frozen --extra dev
uv run pytest
uv run python -m equity_alpha.release_verify --release artifacts/wide-ml-v1-development
uv run python -m equity_alpha.release_verify --release artifacts/wide-incremental-diagnostics-v1
```

Only these explicitly reviewed aggregate releases are whitelisted in Git. Raw
data, row-level forecasts, caches, model binaries and account settings are not.
The [platform runtime provenance](artifacts/wide-incremental-diagnostics-v1/runtime_provenance.json)
records package versions and the earlier image digests. **Host `uv.lock` is the
offline test/development environment, not the frozen QuantRocket environment**;
bitwise model replay from host `uv sync` is not claimed. Native replay also needs
the same entitled bundle/cache vintage. Recorded trials must not be rerun to
search for a more favorable outcome.

The distinct [wide protocol](configs/wide_ml_v1.json) uses 11,220 security IDs as
the source population, a past-only liquidity/price screen capped at 2,500 names
per day, 12 fixed price/volume features, and 1,530,082 development decision rows.
It fits four candidates and three within-date label-permutation controls over
two annual folds: **14 fits**, with **820,028 forecast rows per specification**.
All forecasts are retained, including unavailable future labels. The 731
non-boundary missing outcomes are reported, not filled with zero; tie-aware
Kendall completion bounds are separate from observed-case Spearman intervals.

Features use information through close t, with a return-proxy label from close
t+1 to t+6. Training labels must mature strictly before each forecast year.
Per-date target ranks precede deterministic sampling; date weights are equal.
The paired bootstrap uses 20-session calendar blocks and retains missing-date
positions. The historical common-stock universe, terminal payouts, vendor
revision times and executable returns remain unqualified. This broad panel is
not advertised as survivor-complete ordinary stocks.

**2009–2010 are development, not independent confirmation. 2011 prediction
performance remains disabled.** Its earlier coverage audit is disclosed. No
portfolio/PnL, trading costs or executable Sharpe is computed for this study;
the older pilot's accounting does not transfer automatically to this dataset.

Raw bars, feature/target caches, row-level forecasts and model binaries stay
inside the local QuantRocket platform. The repository release contains only
aggregate diagnostics, code snapshots and provenance. The full native run is
`/codeload/equity-alpha-wide/run-wide-v1-20261030`. Verify it without loading
serialized models:

```bash
docker compose \
  -f infra/quantrocket/compose.research.yml exec -T \
  -e PYTHONPATH=/codeload/equity-alpha-src jupyter \
  python -B -m equity_alpha.wide_verify \
  --run /codeload/equity-alpha-wide/run-wide-v1-20261030 \
  --cache /codeload/equity-alpha-wide/data-wide-v1-20261030-final \
  --registry /codeload/equity-alpha-wide/registry.jsonl
```

The dedicated wide verifier checks the exact 32-artifact inventory, six frozen
source files, eight bound cache files, 14 fit/forecast lifecycles and local registry
bindings. Checksums establish internal consistency, not scientific validity or
external immutability. The older `equity-alpha verify` command is for pilot runs,
not this schema. After the run, the trial-budget guard was hardened to reject
missing prior lifecycle records; frozen predictions and statistics were not rerun.
The recorded fit budget is exhausted: do not reset the registry or rerun the same
protocol merely to obtain a better result.

## Reproduce the fixed-panel pilot

Python 3.11+ and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run equity-alpha audit --data /absolute/path/universe30_daily_2018_2026.csv
uv run equity-alpha run --data /absolute/path/universe30_daily_2018_2026.csv
uv run equity-alpha run --config configs/sector_alpha_v2.json \
  --data /absolute/path/universe30_daily_2018_2026.csv
uv run equity-alpha verify --run artifacts/RUN --data /absolute/path/universe30_daily_2018_2026.csv
```

For the supplied workspace the source is
`../mid-term project/sources/universe30_daily_2018_2026.csv`.
Quote paths containing spaces. Source files are read-only and are not silently
downloaded, merged with other universes, or included in Git. Obtain redistribution
rights before publishing third-party data.

Each run creates a new directory under `artifacts/` with a self-contained HTML
report, scientific exhibits, predictions, exact execution schedule, daily holdings
and cash accounting, costs, attribution, trial ledger, split audit, data/config/code
hashes and frozen model bundles. Existing runs are never overwritten.
Runs also append start/completion/failure events to a locked local hash-chain
registry. `verify` checks every artifact, the exact frozen source inventory,
configuration and optional original data. Local checksums prove internal
consistency; they do not constitute externally immutable storage.

## Wider-universe qualification

The public Qlib US archive has a reconstructed historical S&P 500 membership list
and a much broader price panel. The reader inspects its ZIP directly, retains
missing names/dates, and audits price adjustment units and missing exit quotes.
An explicit symbol map can reconcile vendor punctuation; it cannot reconstruct
missing bankrupt-company prices or delisting payouts.

```bash
uv run equity-alpha audit-qlib --archive data/cache/qlib_us_v2.zip \
  --symbol-map configs/qlib_symbol_map.json --output artifacts/qlib_audit.json
```

The command exits 2 and still saves the audit if qualification fails. This source
does not automatically replace the clean pilot. A wide panel with unresolved
missing historical members is not labeled survivor-complete. Exact source,
download URL and limitations are in [Public data sources](docs/PUBLIC_DATA_SOURCES.md).

## Auxiliary public-data access

```bash
uv run python -m equity_alpha.public_data --output data/access-NEW-DIRECTORY
```

This bounded, no-account acquisition downloads official French daily five-factor
and momentum benchmarks plus the Sharadar public AAPL demonstration. It saves
raw hashes, HTTP provenance, the exact acquisition source and an explicit audit.
Only 2005–2024 risk benchmarks are emitted; current-vintage factors are not
certified point-in-time predictive inputs. The AAPL sample remains ineligible for
cross-sectional research, and truncated requested history is recorded rather
than silently accepted. No model is fitted. These auxiliary downloads do **not**
resolve the broader historical-stock-data requirement. Account-dependent routes,
scope differences and license constraints are recorded in
[`configs/data_access_routes.json`](configs/data_access_routes.json).

## Free in-platform learning data

The QuantRocket daily learning bundle was acquired on 2026-10-30 and audited
inside its local research environment. Its observed scope is **2007-01-03 through
2011-12-30**, **1,260 sessions**, **11,220 securities**, and **9,236,681 observed
closes**. This security count is not a count of verified ordinary common stocks.
There are 33 close observations without complete OHLCV. LEH, KKD and MON are
present as historical security records; current ticker/name metadata are not
treated as point-in-time selection attributes.

The research-only deployment is in `infra/quantrocket/compose.research.yml`.
It publishes only `127.0.0.1:1969`, has no broker/order services, and mounts project
source read-only. The raw bundle stays in QuantRocket's Docker volume; it is not
exported to this repository. Only an aggregate audit with limited identity
diagnostics is retained under
`artifacts/quantrocket-learning-20261030-v1/learning_bundle_audit.json`.

This resolves access to a broader historical learning panel. The subsequent
wide study validated causal split/dividend adjustment mechanics and explicitly
uses a vendor-adjusted return proxy, **not fully qualified terminal wealth**.
Historical common-stock classification and terminal/delisting returns remain
unqualified.
Any 2007–2011 forecasting study must have a separate protocol and must not be
described as evidence for current markets or the original 2025 reserved period.
Acquisition and qualification preceded the separate Wide ML V1 forecasting run.

## Fixed-panel pilot design decisions

- Features available at close **t**; assumed execution close **t+1**; five-session
  forward label ends at close **t+6**. Adjusted closes represent synthetic total-return
  units, not verified auction fills.
- Compact, predeclared price-only feature families: momentum, reversal, risk,
  sector/panel-relative behavior. Fundamentals without publication timestamps are excluded.
- Annual outer tests **2022, 2023, 2024**. Each uses the preceding year for inner
  validation, earlier years for training, and only matured labels for refitting.
  All names on a date share a partition; overlapping labels are purged at boundaries.
- Momentum, reversal, fixed composite and cash baselines; Ridge and shallow
  histogram gradient boosting. Hyperparameters and family selection use inner
  validation mean daily Rank IC, with deterministic ties in configuration order.
  No random internal early stopping is used.
- Shared sector-neutral allocation, 8% target position cap and gross exposure ≤1.
  Global scaling preserves neutrality; realized exposure drifts between trades.
- Self-financing cash/holding ledger, actual traded-notional costs, ACT/365 borrow,
  and final liquidation. Default one-way fees **5 bps**, annual short borrow **100 bps**.
  They are assumptions, not empirically calibrated quotes.
- Date-block confidence intervals and paired model comparisons; three label
  permutation diagnostics, feature-family ablations, cost and delay stress,
  horizon decay, score sorts, and name-level pre-cost attribution.

## Reserved evaluation and claims

**2025 onward remains reserved by default.** Development commands truncate source
prices before that boundary, before building features and labels. The existing
coursework previously examined the source history, so this is a frozen retrospective
evaluation for the new pipeline, not never-seen data. Broad point-in-time data and
prospectively logged predictions are required before stronger claims.

A high in-sample R² or Sharpe is not evidence of predictive alpha. A positive
development result is not automatically promoted, and a failed hypothesis remains
in the report and trial ledger. All candidate/control runs count as research trials.

## Fixed-panel pilot architecture

```text
CSV contract + price/return reconciliation
  → past-only features + delayed labels
  → date-grouped, maturity-purged nested folds
  → validation-only model choice + frozen model artifacts
  → execution schedule + cash/holding ledger
  → paired uncertainty, controls, attribution, report + manifest
```

`src/equity_alpha/` contains the implementation; `tests/` independently checks
temporal causality, label intervals, split boundaries and hand-worked accounting.

- [Research protocol](docs/RESEARCH_PROTOCOL.md)
- [Data card](docs/DATA_CARD.md)
- [Senior-review checklist](docs/REVIEW_CHECKLIST.md)
- [Original findings](docs/FINDINGS_V1.md)
- [V2 findings](docs/FINDINGS_V2.md)
- [Broader-data requirements](docs/DATA_REQUIREMENTS.md)
- [Validation evidence and limits](docs/VALIDATION_EVIDENCE.md)

## Engineering positive and null controls

The research runner is also tested on a fixed, artificial price process whose
predictability is known by construction, and a corresponding independent null.
This checks whether a valid signal can be recovered and whether deployment
abstention works. It is explicitly not stock-market alpha evidence.

```bash
uv run python -m equity_alpha.synthetic_control --output artifacts/synthetic-new-run
uv run python -m equity_alpha.source_report \
  --audit artifacts/qlib-source-qualification.json --output artifacts/source-audit-view
```

Synthetic runs refuse to overwrite existing artifacts. The first frozen seeded
benchmark, its generator parameters, both cases and model decisions are retained
in `artifacts/synthetic-validation-v1/summary.json`. The original engineering
control is a separate study; it does not alter or improve the recorded real-data
V2 findings.
- [Public data source assessment](docs/PUBLIC_DATA_SOURCES.md): Qlib US archive
  provenance, adjustment units, missing securities and historical membership limits.

Only load model bundles produced locally by a trusted run: joblib uses Python
serialization. Frozen-model prediction is a research scoring interface, with no
broker or exchange connection. A scoring example after a run:

```bash
uv run equity-alpha predict --data /path/to/data.csv \
  --bundle artifacts/RUN/models/FOLD_BUNDLE.joblib --asof YYYY-MM-DD
```

The actual bundle filenames are listed by each run. Predictions are not submitted
as orders. Package/version compatibility and feature-schema checks apply.
