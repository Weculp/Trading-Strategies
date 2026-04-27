"""
Comprehensive performance statistics.

All functions take a daily-frequency P&L or return Series and return either a
scalar or a small DataFrame.

The institutional metric set:
  - CAGR
  - Annualised return (arithmetic)
  - Annualised vol
  - Sharpe (rf = 0 for simplicity; we report gross Sharpe)
  - Sortino (downside deviation only)
  - Calmar (CAGR / |max DD|)
  - Max drawdown ($ or %)
  - Average drawdown duration
  - Hit rate (frequency of positive periods)
  - Win/loss magnitude ratio
  - Profit factor (gross gains / gross losses)
  - Skew, kurtosis (return distribution shape)
  - Ulcer index (RMS of drawdown depth — a smoother DD penalty)
  - Best / worst day
  - VaR / CVaR at 95%
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def to_returns(daily_pnl: pd.Series, capital: float | None = None) -> pd.Series:
    """If a capital base is given, convert P&L to return %; else assume returns."""
    if capital is not None and capital > 0:
        return daily_pnl / capital
    return daily_pnl


def cagr(daily_pnl: pd.Series, capital: float) -> float:
    """Compound annual growth rate based on cumulative P&L on a capital base."""
    cum = daily_pnl.cumsum()
    end_value = capital + cum.iloc[-1]
    if end_value <= 0:
        return float("nan")
    yrs = len(daily_pnl) / PERIODS_PER_YEAR
    return (end_value / capital) ** (1 / yrs) - 1


def annualised_return(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.mean() * PERIODS_PER_YEAR)


def annualised_vol(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.std() * np.sqrt(PERIODS_PER_YEAR))


def sharpe(daily_pnl: pd.Series, rf: float = 0.0) -> float:
    excess = daily_pnl - rf / PERIODS_PER_YEAR
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR))


def sortino(daily_pnl: pd.Series, rf: float = 0.0) -> float:
    excess = daily_pnl - rf / PERIODS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(daily_pnl: pd.Series) -> float:
    cum = daily_pnl.cumsum()
    return float((cum - cum.cummax()).min())


def max_drawdown_pct(daily_returns: pd.Series) -> float:
    """Max drawdown for a *return* series (compounded)."""
    cum = (1 + daily_returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def calmar(daily_pnl: pd.Series, capital: float) -> float:
    """CAGR / |max drawdown %|."""
    dd_pct = max_drawdown(daily_pnl) / capital
    if dd_pct == 0:
        return float("nan")
    return cagr(daily_pnl, capital) / abs(dd_pct)


def ulcer_index(daily_pnl: pd.Series) -> float:
    """RMS of drawdown depth as % of peak — Martin (1989)."""
    cum = daily_pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    base = max(abs(peak.iloc[-1]), 1.0)
    return float(np.sqrt((dd ** 2).mean()) / base)


def hit_rate(daily_pnl: pd.Series) -> float:
    nonzero = daily_pnl[daily_pnl != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def win_loss_ratio(daily_pnl: pd.Series) -> float:
    wins = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    if len(wins) == 0 or len(losses) == 0 or losses.mean() == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(daily_pnl: pd.Series) -> float:
    wins = daily_pnl[daily_pnl > 0].sum()
    losses = -daily_pnl[daily_pnl < 0].sum()
    if losses == 0:
        return float("nan")
    return float(wins / losses)


def skew(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.skew())


def kurt(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.kurtosis())


def best_day(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.max())


def worst_day(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.min())


def var_95(daily_pnl: pd.Series) -> float:
    return float(np.percentile(daily_pnl, 5))


def cvar_95(daily_pnl: pd.Series) -> float:
    var = var_95(daily_pnl)
    tail = daily_pnl[daily_pnl <= var]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


def time_in_market(daily_pnl: pd.Series) -> float:
    """Fraction of days with non-zero P&L (a proxy for trading-days exposure)."""
    return float((daily_pnl != 0).mean())


def summary_table(
    daily_pnl: pd.Series,
    label: str,
    capital: float | None = None,
) -> dict:
    """One-row summary dict with every institutional metric."""
    out = {
        "Strategy":              label,
        "Ann. Return ($)":       annualised_return(daily_pnl),
        "Ann. Vol ($)":          annualised_vol(daily_pnl),
        "Sharpe":                sharpe(daily_pnl),
        "Sortino":               sortino(daily_pnl),
        "Max DD ($)":            max_drawdown(daily_pnl),
        "Ulcer Index":           ulcer_index(daily_pnl),
        "Hit Rate":              hit_rate(daily_pnl),
        "Win/Loss ratio":        win_loss_ratio(daily_pnl),
        "Profit Factor":         profit_factor(daily_pnl),
        "Skew":                  skew(daily_pnl),
        "Kurtosis":              kurt(daily_pnl),
        "Best Day ($)":          best_day(daily_pnl),
        "Worst Day ($)":         worst_day(daily_pnl),
        "VaR 95% ($)":           var_95(daily_pnl),
        "CVaR 95% ($)":          cvar_95(daily_pnl),
        "Time in Market":        time_in_market(daily_pnl),
    }
    if capital is not None and capital > 0:
        out["CAGR"]   = cagr(daily_pnl, capital)
        out["Calmar"] = calmar(daily_pnl, capital)
    return out


def yearly_table(daily_pnl: pd.Series) -> pd.DataFrame:
    """Per-calendar-year summary."""
    if len(daily_pnl) == 0:
        return pd.DataFrame()
    rows = []
    for yr, grp in daily_pnl.groupby(daily_pnl.index.year):
        rows.append({
            "Year":    int(yr),
            "Days":    len(grp),
            "P&L":     float(grp.sum()),
            "Vol":     annualised_vol(grp),
            "Sharpe":  sharpe(grp),
            "Max DD":  max_drawdown(grp),
            "Hit %":   hit_rate(grp) if (grp != 0).any() else 0,
        })
    return pd.DataFrame(rows)


def monthly_returns_matrix(daily_pnl: pd.Series) -> pd.DataFrame:
    """Pivot daily P&L into a year×month matrix of monthly P&L."""
    monthly = daily_pnl.resample("ME").sum()
    df = monthly.to_frame("pnl")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot(index="year", columns="month", values="pnl")


def drawdown_periods(daily_pnl: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Top-N drawdown periods with start, trough, recovery dates."""
    cum = daily_pnl.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    in_dd = dd < 0
    if not in_dd.any():
        return pd.DataFrame()
    # Identify contiguous DD runs
    groups = (in_dd != in_dd.shift()).cumsum()
    periods = []
    for gid, sub in dd.groupby(groups):
        if (sub < 0).any():
            start, trough_pos = sub.index[0], sub.idxmin()
            depth = sub.min()
            recover = sub.index[-1]
            periods.append({
                "Start":     start.date(),
                "Trough":    trough_pos.date(),
                "End":       recover.date(),
                "Days":      (recover - start).days,
                "Depth ($)": depth,
            })
    pdf = pd.DataFrame(periods).sort_values("Depth ($)").head(top_n)
    pdf.reset_index(drop=True, inplace=True)
    return pdf
