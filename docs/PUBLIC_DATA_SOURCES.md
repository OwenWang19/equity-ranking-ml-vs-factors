# Public equity data: source selection and limitations

Verified on 2026-10-29. Public availability and software licensing do not imply
that upstream market-data redistribution rights have been granted.

## Selected expansion source: Qlib US daily sample

The [Microsoft Qlib loader](https://github.com/microsoft/qlib/blob/main/qlib/tests/data.py)
currently points to [this release archive](https://github.com/SunsetWolf/qlib_dataset/releases/download/v2/qlib_data_us_1d_latest.zip).
Release metadata gives **450,094,816 bytes**. The archive directory contains
8,994 feature directories, including non-common-stock securities and benchmark
indices; that count is not a count of investable common stocks. The calendar is
1999-12-31 to 2020-11-10, 5,250 sessions. Despite its filename, the archive is not
a current market-data feed. Its S&P 500 file has 755 historical membership records
covering 745 distinct ticker labels.

The audited archive SHA-256 is
`20e3283009784843dcfe690488bf3aa739e64159f1da51a9ec33dd3fb647187f`.
Of 2,665,190 member-date observations, 2,147,541 have observed closes (80.58%).
There are 89 labels with no close observations, including LEH and BS; some other
gaps are explicit symbol mismatches such as BRK.B/BF.B. The reader makes no
unverified alias substitutions. There are 141 membership exits with no observed
close on the immediately following session. These gaps are material: expanding
the name count does not solve survivorship bias or missing liquidation prices.

`configs/qlib_symbol_map.json` provides an **optional explicit** vendor-symbol map
for BRK.B → BRK-B and BF.B → BF-B. Both hyphenated feature directories are present
in the ZIP. The [Yahoo Berkshire Class B quote](https://finance.yahoo.com/quote/BRK-B/)
and [Yahoo Brown-Forman Class B quote](https://finance.yahoo.com/quote/BF-B/) verify
that convention; Brown-Forman's [issuer stock page](https://investors.brown-forman.com/investors/stock-information/stock-quote/default.aspx)
identifies BF.B as its Class B shares. This maps vendor punctuation for the same
share class; it does not map acquired/delisted companies to surviving issuers.
The default reader uses literal labels. Explicit mappings must have existing
quote files and one-to-one vendor IDs. Returned columns and membership labels
stay unchanged; the audit records the mapping, hash and before/after recovered
names and member-date coverage. These two recoveries cannot repair the remaining
missing historical securities or certify historical identity continuity.

**Known source membership error:** the archive assigns BRK.B the interval
1999-01-01 through 2099-12-31. Berkshire's own [2010-01-28 news release](https://www.berkshirehathaway.com/news/jan2810.pdf)
states that S&P announced its addition on 2010-01-26, with the effective date then
still to be announced. Thus its pre-2010 membership is demonstrably backdated.
Literal membership is retained for source fidelity; it must not be described as
correct historical S&P 500 membership, and a symbol-punctuation correction does
not repair this chronology. An independently sourced membership history is
required before making historical-index-universe claims.
The machine-readable audit therefore records `membership_quality.status` as
`known_inaccurate` for this exact archive SHA-256 and includes the issuer evidence.
This is a failed source qualification for historical-index-universe claims,
stronger than an absence of verification. Other archive hashes and synthetic test
fixtures remain `unverified`; the finding is not inferred merely from a filename.

The project reads only the calendar, reconstructed S&P 500 membership, and named
OHLC, volume and adjustment-factor arrays for those labels. All source bytes remain
unchanged. Missing labels and dates stay in the audit, with no forward fill,
backfill, future-availability screen, or implicit removal of a difficult exit.
Nonpositive/infinite price or factor values and negative/infinite volumes, if
present, are converted to NaN in returned research views and counted by name,
field and date. Existing NaNs remain missing. Raw archive bytes are unchanged.
Membership is inclusive as written in the source file. Literal intervals are not
proof of announcement-time availability, correct security identity, or complete
historical coverage.

Microsoft's [collector documentation](https://github.com/microsoft/qlib/blob/main/scripts/data_collector/yahoo/README.md)
says data come from Yahoo Finance, may contain abnormal adjusted prices, and are
not updated regularly. The [normalization implementation](https://github.com/microsoft/qlib/blob/main/scripts/data_collector/yahoo/collector.py)
multiplies raw OHLC by `adjclose / close`, divides volume by that factor, then
scales prices and the factor by the first adjusted close and inversely scales
volume. Consequently, using the archived values:

- Original Yahoo OHLC = normalized OHLC / normalized factor.
- Original Yahoo volume = normalized volume * normalized factor.
- Dollar turnover = normalized close * normalized volume.
- Normalized-price ratios preserve the provider's adjusted-return ratios.

These are transformations of vendor observations, not an independent corporate
action or total-return verification. Normalized volume is not a raw share count;
normalized close is not a dollar stock price. Do not apply dollar-price or raw
volume screens directly to those normalized fields.

The archive lacks audited security identifiers, terminal delisting payouts,
borrow availability, spread and executable close prices. Historical ticker reuse,
renames and unavailable dead tickers remain material limitations. Microsoft code
is MIT-licensed; its license does not relicense upstream Yahoo observations.
Third-party data remain outside Git. This source supports an explicitly limited
public-data research experiment; it cannot establish survivor-complete alpha.

## Alternatives investigated

### Gu, Kelly and Xiu author data

[Dacheng Xiu's author page](https://dachxiu.chicagobooth.edu/) links
[datashare.zip](https://dachxiu.chicagobooth.edu/download/datashare.zip), currently
1,555,896,289 bytes (3,836,202,174-byte CSV inside). The embedded README defines
DATE, CRSP permno, 94 lag-adjusted characteristics and SIC2. It explicitly states
that **CRSP returns are not included** and must be obtained through WRDS. Timing
lags are already applied and need careful matching to the README's response-date
convention. The author website restricts use to research/noncommercial purposes
and requests paper citations. Downloading this large file does not solve the
absence of licensed individual-stock return labels.

### Open Source Asset Pricing

The [official October 2026 data page](https://www.openassetpricing.com/data/)
links [OpenSourceAP/CrossSection](https://huggingface.co/datasets/OpenSourceAP/CrossSection),
release `Data-Release-2026.10`. The author dataset card specifies CC BY 4.0 and no
account requirement. It offers 209 public firm characteristics in a 1.9 GB
parquet file and factor portfolio returns, generally through December 2025.
Price, Size and STreversal require WRDS, and individual-stock future return
labels are not publicly supplied. Portfolio returns are useful factor controls;
they must not be substituted for individual-stock targets.

### Reconstructed historical S&P 500 lists

[fja05680/sp500](https://github.com/fja05680/sp500) provides MIT-licensed
reconstructed membership from 1996. The maintainer documents a Trading Evolved
base extended through Wikipedia and manual research; selected Wikipedia changes
are not complete. The repository provides neither security-level prices nor
delisting returns. Combining its membership with current Yahoo prices still
requires a missing-security and terminal-return audit before any survivor-free
claim.
