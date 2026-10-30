# Research data card

## Pilot source

The daily CSV is an external local input. Its default source is the course
project's `sources/universe30_daily_2018_2026.csv`; raw observations are not copied
into this repository. The run audit records its absolute path and SHA-256 hash.

The audited snapshot contains 30 selected US stocks, 65,850 rows and 2,195 common
dates from 2018-01-02 to 2026-09-25. Fields are date, ticker, sector, adjusted
close, and adjusted return. Numeric returns use decimal units: 0.01 means 1%.
For every consecutive pair of available prices, supplied returns match price
ratios to floating-point precision. First supplied returns are not independently
checkable without the preceding price and are excluded from verification.

## Scope and provenance limits

- This is a fixed selected stock universe. Historical membership, securities
  removed from the universe, and the original selection date are not documented.
  Results are exploratory and subject to survivorship and selection bias.
- Adjusted-price arithmetic is internally consistent. Vendor provenance,
  corporate-action adjustment quality and exact point-in-time availability have
  not been independently verified. Adjustment does not make historical dollar
  prices suitable for measuring live execution or liquidity.
- There are no volume, raw OHLC, bid/ask, borrow or order observations. Trading
  costs are scenario assumptions; capacity and impact estimates are not measured.
- Static sector labels do not contain historical classification effective dates.
  Sector comparisons therefore inherit this data limitation.
- A common observed calendar is validated. An external exchange calendar is not
  supplied, so the loader does not certify that every exchange session is present.

## Excluded course files

The broad-universe `SP500_daily_returns_UPDATED.csv` dependency referenced by the
final course notebook is absent. Saved large-universe outputs are not a substitute
for reproducible raw input data.

The broad monthly workbook contains prices, a field called TRI Gross Divs, market
caps, book value and ROE, but no membership intervals or financial publication
timestamps. The TRI field equals PX_LAST in approximately 98.75% of jointly valid
rows; total-return semantics require independent clarification. This source is
excluded from the daily pilot.

Four monthly return observations in the course's 30-stock monthly file differ by
more than two percentage points from compounded daily returns. Labels and returns
in this project are computed from the single daily adjusted-price source, avoiding
an undocumented mixture of incompatible monthly and daily series.

## Requirements for broader research claims

A more general equity-alpha conclusion requires a wider universe with documented
historical selection, reliable corporate actions and delisting returns, effective
sector history, and dated data availability. Liquidity-sensitive claims additionally
require observed volume, execution prices, spreads and borrow data. These are data
acceptance gates, not defects that additional model complexity can resolve.
