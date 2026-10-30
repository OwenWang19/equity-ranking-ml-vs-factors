# Data required for a credible broader equity study

The public-source audit is complete and failed qualification. This document gives
a provider-neutral export specification for the next experiment; it is not a
claim that an institutional source has been obtained or approved. Do not provide
account passwords or API secrets in a repository or chat.

## Daily security history

Use stable security identifiers rather than ticker alone. Include all securities
eligible under the historical universe rule, including names that later merged,
delisted, changed ticker, or failed. A useful first sample is US common equities
from at least 2005 onward, with 2025 onward still reserved for the prediction study.

Preferred fields:

| Field | Meaning |
| --- | --- |
| `security_id` | Stable provider identifier and its namespace |
| `date` | Exchange trading-session date |
| `ticker` | Historical display symbol, not the primary join key |
| `currency` | Quote/return currency |
| `close` | Raw closing-price convention and missing-value codes documented |
| `total_return` | Daily return including documented dividend/corporate-action treatment |
| `volume` | Shares traded, units and adjustment convention documented |
| `split_factor` | Adjustment convention, if available |
| `delisting_return` | Terminal return and explicit relationship to `total_return` |
| `is_terminal` | Last-observation/terminal-event flag with reason where available |
| `available_at` | Timestamp or documented availability convention |

If the vendor's return already incorporates a delisting event, do not compound
the terminal return twice. If it does not, retain both components and the vendor
definition. Missing terminal prices/payouts remain unresolved data cases, not
automatic zero returns or last-price exits. An audited return-index construction
can support the existing synthetic total-return-unit ledger after these semantics
are established; raw price ratios alone are insufficient.

## Universe and identity history

For an index-membership study, provide `security_id`, `effective_from`,
`effective_to`, and announcement/availability time where available, plus the source
definition. Include removals. Explain whether the bounds are inclusive, and how
non-trading effective dates map to the session calendar.

For an investable-universe rule instead, specify exchange/security-type eligibility
and evaluate size/liquidity screens from lagged observable inputs. Do not condition
historical eligibility on survival to the present, future data completeness, or a
future minimum sample length.

Ticker aliases require dated identity mappings. Punctuation conversion such as
BRK.B to BRK-B is a quote-provider convention; merger chains, ticker reuse and
new companies are separate identity decisions requiring evidence.

## Sector and fundamentals

Sector classifications need stable identifiers and historical effective dates.
Current GICS sectors backfilled over history are not automatically point-in-time.
If history is unavailable, label the study's grouping limitation or use a separately
declared target/risk design rather than inventing historical sectors.

Fundamentals require fiscal period, actual publication/filing timestamp, values,
and revision/vintage handling. A blanket month lag is not proof of availability.
They remain excluded until those requirements are met.

## Provenance and validation before training

Keep original exports immutable. Record provider/table, query, extraction date,
field definitions, currency, timezone, calendar, entitlement and redistribution
conditions, checksum and package/code version. Licensing determines whether raw
data can be shared; research code and derived summaries do not confer data rights.

Before the next fitted model, audit coverage by name/date and historical membership,
identifier uniqueness, adjustment/return consistency, corporate actions, terminal
events, publication timing, and price/volume scale. Retain rejection cases. Any
accepted missingness or exit policy must be explicit in the new protocol and
reported as sensitivity, not silently introduced to obtain a complete backtest.

## Access options verified during this iteration

- The authors' [JKP stock-data guide](https://jkpfactors.com/jkp-wrds-guide) and
  [CTF access guide](https://jkpfactors.com/ctf/dataset-access) use WRDS with the
  relevant CRSP/Compustat access. Institution access and permitted use must be
  checked by the account holder.
- Author-provided Gu–Kelly–Xiu characteristics do not include the CRSP return labels;
  obtaining a feature ZIP does not create a complete forecasting dataset.
- Public Qlib quotes and reconstructed index spans are useful for qualification
  and source-bias studies, but the audited release fails the current full-universe
  gate. Its `latest` filename is not a current date or quality guarantee.

The project can continue improving engineering and synthetic validation while
the data channel is resolved. Claims about broad-stock alpha remain conditional
on satisfying the empirical data requirements above.
