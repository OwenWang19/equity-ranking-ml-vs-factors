"""Date-grouped chronological splits purged using actual label maturity."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    year: int
    train: pd.Index
    validation: pd.Index
    refit: pd.Index
    test: pd.Index
    test_start: pd.Timestamp


def walk_forward_folds(panel, config):
    dates = panel.index.get_level_values("date")
    end = panel["label_end"]
    folds = []
    for year in config.test_years:
        validation_start = pd.Timestamp(f"{year - 1}-01-01")
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        train = panel.index[(dates < validation_start) & (end < validation_start)]
        validation = panel.index[(dates >= validation_start) & (dates < test_start) & (end < test_start)]
        refit = panel.index[(dates < test_start) & (end < test_start)]
        test = panel.index[(dates >= test_start) & (dates < test_end)]
        for label, rows, minimum in (
            ("train", train, config.min_train_dates),
            ("validation", validation, config.min_validation_dates),
        ):
            n_dates = rows.get_level_values("date").nunique()
            if n_dates < minimum:
                raise ValueError(f"Fold {year} {label} has {n_dates} dates, requires {minimum}")
        if len(test) == 0:
            raise ValueError(f"Fold {year} has no test rows")
        if end.loc[train].max() >= validation_start or end.loc[refit].max() >= test_start:
            raise AssertionError("Label maturity violates split boundary")
        folds.append(Fold(year, train, validation, refit, test, test_start))
    return folds
