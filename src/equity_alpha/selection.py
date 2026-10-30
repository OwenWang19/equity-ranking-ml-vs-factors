"""Predeclared selection and abstention decisions, using validation evidence only."""


def select_candidate(names, metrics, *, metric="rank_ic", cash_hurdle=False):
    if not names:
        raise ValueError("No eligible candidates")
    if metric == "rank_ic":
        chosen = max(names, key=lambda name: metrics[name]["rank_ic"])
    elif metric == "sector_mse":
        chosen = min(names, key=lambda name: metrics[name]["mse"])
    else:
        raise ValueError("Unknown selection metric")
    row = metrics[chosen]
    active = True
    reason = "no_abstention_in_protocol"
    if cash_hurdle:
        active = bool(row["mse"] < row["zero_mse"] and row["rank_ic"] > 0)
        reason = "beats_zero_mse_and_positive_sector_ic" if active else "validation_cash_hurdle_failed"
    return {"candidate": chosen, "active": active, "reason": reason}
