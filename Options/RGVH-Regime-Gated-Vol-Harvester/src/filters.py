"""
The three RGVH skip filters.

Given a per-trade DataFrame with columns:
    iv_rank                 (SPY F_iv_rank_252)
    vxn_excess_rank_252     (rolling rank of VXN - VIX)
    slope_2s10s_rank_252    (rolling rank of 10y - 2y Treasury slope)

`build_skip_mask` returns a boolean Series — True means SKIP this trade
(do NOT enter a short straddle today).

The three thresholds are deliberately set to NaN in `config.py`. To run a real
backtest, edit config.py with values you've validated on your own data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import IV_THR, VXN_THR, SLOPE_THR


def filter_iv_regime(trades: pd.DataFrame, threshold: float = IV_THR) -> pd.Series:
    """Skip when SPY's 30-day IV percentile rank is in the upper end."""
    return (trades["iv_rank"] > threshold).fillna(False)


def filter_cross_asset_stress(trades: pd.DataFrame, threshold: float = VXN_THR) -> pd.Series:
    """Skip when VXN-VIX spread (Nasdaq excess vol) is unusually elevated."""
    return (trades["vxn_excess_rank_252"] > threshold).fillna(False)


def filter_yield_curve_inversion(trades: pd.DataFrame, threshold: float = SLOPE_THR) -> pd.Series:
    """Skip when the 2s10s slope is in the bottom of its trailing range
    (i.e., curve inverted or near-inverted = recession-warning regime)."""
    return (trades["slope_2s10s_rank_252"] < threshold).fillna(False)


def build_skip_mask(
    trades: pd.DataFrame,
    iv_thr: float = IV_THR,
    vxn_thr: float = VXN_THR,
    slope_thr: float = SLOPE_THR,
) -> pd.Series:
    """
    Combine the three filter rules with OR semantics:
    skip a day if ANY of the three regime conditions fires.

    Returns:
        boolean Series aligned to `trades.index`; True = SKIP.
    """
    if any(np.isnan(t) for t in (iv_thr, vxn_thr, slope_thr)):
        raise ValueError(
            "RGVH thresholds are NaN by default — set IV_THR / VXN_THR / SLOPE_THR "
            "in config.py based on your own backtest. The original calibration is "
            "deliberately redacted from this open-source release."
        )
    return (
        filter_iv_regime(trades, iv_thr)
        | filter_cross_asset_stress(trades, vxn_thr)
        | filter_yield_curve_inversion(trades, slope_thr)
    )


# Convenience: classify which filter(s) caused a skip on each row
def skip_reason(
    trades: pd.DataFrame,
    iv_thr: float = IV_THR,
    vxn_thr: float = VXN_THR,
    slope_thr: float = SLOPE_THR,
) -> pd.Series:
    """Return a string Series labelling which filter(s) fired (or 'kept')."""
    iv  = filter_iv_regime(trades, iv_thr)
    vxn = filter_cross_asset_stress(trades, vxn_thr)
    slp = filter_yield_curve_inversion(trades, slope_thr)
    out = pd.Series("kept", index=trades.index, dtype=object)
    out[iv]  = "iv_regime"
    out[vxn] = out[vxn] + " | cross_asset"
    out[vxn & ~iv] = "cross_asset"
    out[slp & ~iv & ~vxn] = "curve_inversion"
    return out
