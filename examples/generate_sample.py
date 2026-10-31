"""Generate tiny, deterministic, entirely invented schema examples.

No network, vendor observations, fitted models or historical performance results
are used. Weekdays are merely convenient labels, NOT an NYSE session calendar.
The examples demonstrate field shapes and split/dividend conventions; they are
too small for the registered wide-study training requirements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from datetime import date, timedelta
from pathlib import Path


def _csv(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def sample_files():
    """Return the exact text fixture bytes as UTF-8 strings, without writing."""
    days = []
    candidate = date(2008, 1, 2)
    while len(days) < 140:
        if candidate.weekday() < 5:
            days.append(candidate.isoformat())
        candidate += timedelta(days=1)
    assets = [
        {"asset_id": f"SYNTHETIC-{j + 1:03d}", "sid": 900001 + j, "symbol": f"SYNTH_{chr(65 + j)}"}
        for j in range(3)
    ]
    rows = []
    last_close = {}
    split_day, dividend_day, dividend_ratio = 70, 100, 0.995
    dividend_amount = None
    for i, day in enumerate(days):
        for j, asset in enumerate(assets):
            scale = 0.5 if j == 0 and i >= split_day else 1.0
            if j == 1 and i >= dividend_day:
                scale *= dividend_ratio
            value = (28.0 + 22.0 * j) * math.exp(
                0.0003 * i + 0.004 * math.sin(0.37 * i + j) + 0.001 * math.cos(0.11 * i - 2 * j)
            )
            close = round(value * scale, 8)
            opening = round(close * (1 + 0.0005 * math.sin(i + 0.2 * j)), 8)
            high = round(max(opening, close) * 1.002, 8)
            low = round(min(opening, close) * 0.998, 8)
            volume = 1_100_000 + 170_000 * j + int(40_000 * math.sin(0.13 * i + j))
            if j == 0 and i >= split_day:
                volume *= 2
            if j == 1 and i == dividend_day:
                dividend_amount = round(last_close[j] * (1 - dividend_ratio), 12)
            rows.append(
                {
                    "date": day,
                    "asset_id": asset["asset_id"],
                    "open": f"{opening:.8f}",
                    "high": f"{high:.8f}",
                    "low": f"{low:.8f}",
                    "close": f"{close:.8f}",
                    "volume": volume,
                }
            )
            last_close[j] = close
    definitions = {
        "daily_bars.csv": {
            "rows": rows,
            "fields": [
                (
                    "date",
                    "ISO-8601 date string",
                    "synthetic weekday label; timezone-naive UTC midnight convention",
                ),
                ("asset_id", "string", "invented persistent security identity"),
                ("open", "float64", "synthetic USD per share, unadjusted"),
                ("high", "float64", "synthetic USD per share, unadjusted"),
                ("low", "float64", "synthetic USD per share, unadjusted"),
                ("close", "float64", "synthetic USD per share, unadjusted"),
                ("volume", "int64", "synthetic shares traded"),
            ],
        },
        "assets.csv": {
            "rows": assets,
            "fields": [
                ("asset_id", "string", "invented persistent identity; joins daily_bars.asset_id"),
                ("sid", "int64", "invented bundle-local identity; joins action tables"),
                ("symbol", "string", "invented display label, not an eligibility input"),
            ],
        },
        "split_ratios.csv": {
            "rows": [{"sid": 900001, "effective_date": days[split_day], "ratio": 0.5}],
            "fields": [
                ("sid", "int64", "local identity joining assets.sid"),
                ("effective_date", "ISO-8601 date string", "synthetic UTC-midnight effective date"),
                ("ratio", "float64", "positive adjustment factor; 0.5 denotes the sample 2-for-1 split"),
            ],
        },
        "dividend_ratios.csv": {
            "rows": [{"sid": 900002, "effective_date": days[dividend_day], "ratio": dividend_ratio}],
            "fields": [
                ("sid", "int64", "local identity joining assets.sid"),
                ("effective_date", "ISO-8601 date string", "synthetic UTC-midnight ex-date"),
                (
                    "ratio",
                    "float64",
                    "1 minus amount / previous synthetic close; price-return proxy convention",
                ),
            ],
        },
        "merger_ratios.csv": {
            "rows": [],
            "fields": [
                ("sid", "int64", "local identity joining assets.sid"),
                ("effective_date", "ISO-8601 date string", "synthetic UTC-midnight effective date"),
                ("ratio", "float64", "positive price-adjustment factor; no sample merger is invented"),
            ],
        },
        "dividend_payouts.csv": {
            "rows": [{"sid": 900002, "ex_date": days[dividend_day], "amount": dividend_amount}],
            "fields": [
                ("sid", "int64", "local identity joining assets.sid"),
                ("ex_date", "ISO-8601 date string", "synthetic UTC-midnight ex-date"),
                ("amount", "float64", "synthetic USD per share; matching core consumes sid/ex_date only"),
            ],
        },
    }
    files = {
        name: _csv([field[0] for field in definition["fields"]], definition["rows"])
        for name, definition in definitions.items()
    }
    manifest = {
        "schema_version": 1,
        "origin": "SYNTHETIC ONLY: deterministic formulas, not sampled from vendor or real security observations",
        "purpose": "Tiny field/adjustment-convention examples; not evidence for historical performance or alpha",
        "assets": 3,
        "date_labels": 140,
        "date_range": [days[0], days[-1]],
        "calendar": "Weekdays only; no NYSE holiday/session completeness or real-trading-calendar claim",
        "contains_vendor_data": False,
        "used_for_reported_market_results": False,
        "sufficient_for_registered_model_training": False,
        "sample_adjustments": {
            "split": "SYNTHETIC-001 raw price halves and raw volume doubles from the effective date",
            "dividend": "SYNTHETIC-002 raw price scale changes by 0.995; amount and prior close satisfy ratio = 1 - amount/previous_close",
            "cash_total_return_qualified": False,
        },
        "files": {
            name: {
                "rows": len(definition["rows"]),
                "fields": [
                    {"name": f[0], "type": f[1], "units_or_meaning": f[2]} for f in definition["fields"]
                ],
                "bytes": len(files[name].encode("utf-8")),
                "sha256": hashlib.sha256(files[name].encode("utf-8")).hexdigest(),
            }
            for name, definition in definitions.items()
        },
    }
    files["sample_manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true", help="Verify existing examples without writing")
    args = parser.parse_args(argv)
    files = sample_files()
    if args.check:
        for name, content in files.items():
            if (args.output / name).read_bytes() != content.encode("utf-8"):
                raise ValueError(f"Synthetic fixture differs from generator: {name}")
        print(f"Verified {len(files)} synthetic example files; no market data or models read")
    else:
        if any((args.output / name).exists() for name in files):
            raise FileExistsError("Refusing to overwrite existing examples; use --check to verify")
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            with (args.output / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
        print(f"Created {len(files)} synthetic example files")


if __name__ == "__main__":
    main()
