# Project archive and data dictionary

Project: **Cross-Sectional Equity Ranking: ML vs. Momentum and Reversal**.
Repository: `OwenWang19/equity-ranking-ml-vs-factors`.
Archive date: **2026-10-31**. This is the only maintained local project folder.

## Future work is limited to CV/documentation

- Research is frozen. Inspect existing evidence and polish résumé wording; do
  not continue experiments or seek more favorable results by default.
- Do not download data, recreate Docker resources, rebuild a virtual environment,
  refit models or release reserved evaluation without a new explicit user request.
- Full vendor data, Qlib downloads, installers, private feature/label caches,
  per-security forecasts, fitted models and the old project runtime are removed.
- Retain code, tests, documentation, frozen aggregate evidence, small historical
  qualification records and the tiny synthetic `examples/` fixture.
- Preserve frozen `artifacts/` files and their hashes. Paths recorded there are
  historical provenance, not evidence that native data still exist locally.
- Portable aggregate verification still works. Native cache verification/model
  replay requires reacquiring the same entitled data vintage; do not claim the
  small example reproduces the reported market results.
- The Python package name remains `equity_alpha` for compatibility. Do not touch
  the separate `../mid-term project` course folder or any unrelated project.

## Retained example data: synthetic only

`examples/` has **3 invented assets × 140 weekday labels = 420 OHLCV rows**,
2008-01-02–2008-07-15. Dates are not a certified NYSE calendar. Values come from
deterministic formulas, not vendor samples or anonymized real observations.
These files were not used for any reported market performance and are too small
for the registered training protocol. The six CSVs together occupy about 34 KB.

| File | Rows | Exact fields |
| --- | ---: | --- |
| `daily_bars.csv` | 420 | `date, asset_id, open, high, low, close, volume` |
| `assets.csv` | 3 | `asset_id, sid, symbol` |
| `split_ratios.csv` | 1 | `sid, effective_date, ratio` |
| `dividend_ratios.csv` | 1 | `sid, effective_date, ratio` |
| `merger_ratios.csv` | 0 | `sid, effective_date, ratio` (header only) |
| `dividend_payouts.csv` | 1 | `sid, ex_date, amount` |

Types and units:

- `date`, `effective_date`, `ex_date`: ISO `YYYY-MM-DD` strings, interpreted as
  UTC-midnight daily labels. The original core also accepts integer epoch seconds
  for action dates; do not mistake seconds for milliseconds.
- `asset_id`: invented persistent string joining bars to `assets.csv`. Original
  persistent identity was `asset.real_sid`, distinct from numeric local `sid`.
- `sid`: integer local security key joining action tables to `assets.csv`.
- `symbol`: invented display string, not a point-in-time selection input.
- `open`, `high`, `low`, `close`: positive floating-point unadjusted USD/share.
  The native reader already converted its internal .001 storage units.
- `volume`: nonnegative integer shares traded, not dollar volume.
- `ratio`: finite positive adjustment factor; 0.5 denotes a 2-for-1 split.
  Dividend ratios use `1 - amount / previous_close`.
- `amount`: floating-point USD/share. The core matches payouts by `sid, ex_date`;
  this example amount demonstrates the dividend ratio convention.

`sample_manifest.json` records exact types, units, row counts, byte sizes and
SHA-256 hashes. `python examples/generate_sample.py --check` verifies deterministic
fixture regeneration without downloading anything.

## Retained research evidence is not training data

- `artifacts/wide-ml-v1-development/`: model/control summaries, aggregate daily
  IC and coverage, score-decile coverage, training counts, trial lifecycle,
  source snapshots, reports, figures and integrity records.
- `artifacts/wide-incremental-diagnostics-v1/`: aggregate date/scope diagnostics,
  five conditions, partial-rank analysis, fixed block sensitivities, runtime
  provenance, reports, source snapshots and retained failure/retry metadata.
- `archive/data_qualification/`: historical acquisition-stage counts, limited
  identity examples, file fingerprints and image/runtime metadata. Earlier
  `model_fitted: false` entries describe acquisition, not the later fitted study.

These files contain no per-security bars, targets, scores, residuals or serialized
models. Keep the small aggregate CSVs as supporting evidence for the CV claims.

## Deleted original wide-data contracts

The original bundle contained daily OHLCV for 11,220 security records over
2007-01-03–2011-12-30, not 11,220 verified ordinary common stocks. Development
raw inputs stopped before 2011; 2011 prediction performance was never evaluated.

Raw inputs were five arrays `open, high, low, close, volume`, each shaped
`[session, security]`, with UTC-midnight session labels. Original corporate-action
tables were `splits, dividends, mergers` with `sid, effective_date, ratio`;
cash-payout matching consumed `sid, ex_date`. Vendor declaration/payment times
were not qualified.

| Private cache field | Type | Shape / meaning |
| --- | --- | --- |
| `X.npy` | float32 | `[N,12]`, finite per-date cross-sectional feature ranks |
| `y.npy` | float32 | `[N]`, five-session adjusted-price return proxy; NaN if unavailable |
| `status.npy` | uint8 | `[N]`, label status/reason codes below |
| `date_index.npy` | int32 | `[N]`, positions in session dictionary |
| `asset_index.npy` | int32 | `[N]`, positions in persistent-ID dictionary |
| `dates.npy` | datetime64[D] | `[D]`, development session dictionary |
| `asset_ids.npy` | string | `[A]`, persistent security IDs |

The deleted `private_predictions.npz` contained exactly `predictions`
(float32 `[N_test,10]`), `date_index`, `asset_index`, `model_names`.
These arrays and fitted `.joblib` models were never published.

The fixed feature order is:
`rev1, rev5, mom21, mom63, mom126, vol21, vol63, range20, log_adv20,
volume_surprise, illiq21, close_location20`.
They cover reversal/momentum, log-return volatility, daily range, dollar volume,
split-causal volume surprise, illiquidity and close location. Their exact formulas
are in `src/equity_alpha/wide_data.py`; no missing bar is filled or bridged.
Daily feature ranks are formed before checking future-label availability.

Status codes: `0 complete`, `1 missing_entry`, `2 missing_exit`, `3 internal_gap`,
`4 right_boundary`, `5 unresolved_action`, `6 invalid_proxy`.

Features use information through close `t`; assumed entry is close `t+1`, exit
close `t+6`. The label compounds daily gross proxies ending `t+2…t+6`.
Only matured labels enter each annual training fold. This is a vendor-adjusted
price-return proxy, not qualified cash-inclusive terminal wealth.

The separate older 30-name pilot CSV contract was
`date, ticker, sector, adj_close, adj_return`: daily dates, security/sector strings,
positive adjusted prices and decimal returns consistent with consecutive prices.
That source is not retained inside this archive. Do not merge the pilot and wide
study findings or delete the separate original course folder.

## Supported CV claims and limits

- 1,530,082 development decision rows, 12 fixed features, maturity-purged annual
  walk-forward evaluation, Ridge and shallow histogram gradient boosting.
- Four candidates plus three training-label controls over two folds: 14 market-data
  fits; 820,028 forecasts per specification, including missing outcomes.
- Calendar-block uncertainty, permutation controls, coverage/completion bounds,
  and post-hoc incremental-signal diagnostics beyond momentum/reversal.
- Original completion suite: 540 tests; the archival sample adds tests. Cite the
  latest verified CI record when giving a total.
- No reliable positive standalone/incremental ML advantage was established.
  Do not claim profitable alpha, executable Sharpe, production readiness,
  survivor-complete ordinary stocks or fully point-in-time vendor data.
