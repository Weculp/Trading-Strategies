"""
Evaluate trade lists: Sharpe, drawdown, hit rate, P&L statistics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def daily_pnl_series(trades: pd.DataFrame, col: str = "net_pnl") -> pd.Series:
    """Aggregate trade-level P&L to a business-daily series."""
    if len(trades) == 0:
        return pd.Series(dtype=float)
    eq = trades.groupby("entry_date")[col].sum().sort_index()
    full = pd.date_range(eq.index.min(), eq.index.max(), freq="B")
    return eq.reindex(full, fill_value=0.0)


def sharpe(daily: pd.Series, periods_per_year: int = 252) -> float:
    if len(daily) == 0 or daily.std() == 0:
        return float("nan")
    return float(daily.mean() / daily.std() * np.sqrt(periods_per_year))


def max_drawdown(daily: pd.Series) -> float:
    if len(daily) == 0:
        return float("nan")
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def summary(trades: pd.DataFrame, label: str = "") -> dict:
    """Return a dict of summary metrics for a trade list."""
    if len(trades) == 0:
        return {"label": label, "trades": 0}
    yrs = max((trades["entry_date"].max() - trades["entry_date"].min()).days / 365.25, 0.01)
    daily = daily_pnl_series(trades)
    net = trades["net_pnl"].values
    return {
        "label":     label,
        "trades":    len(trades),
        "years":     round(yrs, 2),
        "ann_net":   int(net.sum() / yrs),
        "sharpe":    round(sharpe(daily), 3),
        "hit_rate":  round((net > 0).mean(), 3),
        "max_dd":    int(max_drawdown(daily)),
        "mean_per_trade":   round(net.mean(), 2),
        "median_per_trade": round(np.median(net), 2),
    }


def yearly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-year P&L, Sharpe, hit rate."""
    if len(trades) == 0:
        return pd.DataFrame()
    out = []
    for yr, grp in trades.assign(year=trades["entry_date"].dt.year).groupby("year"):
        d = daily_pnl_series(grp)
        out.append({
            "year":    int(yr),
            "trades":  len(grp),
            "net_pnl": int(grp["net_pnl"].sum()),
            "sharpe":  round(sharpe(d), 2),
            "hit":     round((grp["net_pnl"] > 0).mean(), 3),
        })
    return pd.DataFrame(out)
