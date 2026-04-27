"""
Build the daily factor panel from the unified options parquet.

For RGVH we only need a small subset of the original 77-factor research panel:
    - F_iv_atm_30      (interpolated 30-DTE ATM IV from the daily surface)
    - F_iv_rank_252    (rolling 252-day percentile rank of F_iv_atm_30)

This module also produces ancillary daily series used by the filter rules:
    - vix, vxn, vxn_excess_rank_252  (from yfinance / the VIX index file)
    - dgs2, dgs10, slope_2s10s, slope_2s10s_rank_252  (from FRED / Treasury file)

Output:
    data/processed/factor_panel_daily.parquet
        tradeDate, S, F_iv_atm_30, F_iv_rank_252,
        vix, vxn_excess_rank_252,
        slope_2s10s, slope_2s10s_rank_252
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from scipy.interpolate import interp1d

from .config import RANK_LOOKBACK


def per_expiry_atm_iv(g: pd.DataFrame) -> float:
    """Interpolate the ATM IV (call-delta = 0.5) for one (date, expir) group."""
    g = g.sort_values("delta").drop_duplicates("delta", keep="first")
    if len(g) < 4:
        return float("nan")
    x, y = g["delta"].to_numpy(), g["iv_mid"].to_numpy()
    if not (x.min() <= 0.5 <= x.max()):
        return float("nan")
    return float(interp1d(x, y, kind="linear",
                          bounds_error=False, fill_value=np.nan)(0.5))


def per_date_iv30(g: pd.DataFrame) -> float:
    """Linearly interpolate ATM IV across expiries to 30-DTE."""
    g = g.dropna(subset=["atm_iv"]).sort_values("dte")
    g = g[g["dte"] > 0]
    if len(g) < 2:
        return float("nan")
    f = interp1d(g["dte"].to_numpy(), g["atm_iv"].to_numpy(),
                 kind="linear", bounds_error=False, fill_value=np.nan)
    return float(f(30))


def build_factor_panel(
    unified_path: Path | str,
    vix_path: Path | str,
    treasury_path: Path | str,
    output_path: Path | str,
    iv_clip: tuple[float, float] = (0.02, 3.0),
) -> pd.DataFrame:
    """
    Build the daily factor panel from:
      - the unified options parquet
      - a VIX/VXN parquet with columns ['date','vix','vxn']
      - a Treasury parquet with columns ['date','dgs2','dgs10']
    """
    unified_path  = Path(unified_path)
    vix_path      = Path(vix_path)
    treasury_path = Path(treasury_path)
    output_path   = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) ATM IV30 per day
    print("Computing per-(date, expir) ATM IV ...")
    df = pl.read_parquet(unified_path)
    clean = df.filter(
        pl.col("iv_mid").is_finite() & pl.col("iv_mid").is_between(*iv_clip)
        & pl.col("gamma").is_not_null() & (pl.col("gamma") >= 0) & (pl.col("gamma") < 2.0)
        & pl.col("vega").is_not_null() & (pl.col("vega") >= 0) & (pl.col("vega") < 500.0)
        & ((pl.col("callBidPrice") > 0) | (pl.col("putBidPrice") > 0))
    )
    sub = clean.select(["tradeDate", "expirDate", "dte",
                        "delta", "iv_mid", "spotPrice"]).to_pandas()

    atm_by = (sub.groupby(["tradeDate", "expirDate"])
                 .apply(per_expiry_atm_iv, include_groups=False)
                 .reset_index(name="atm_iv"))
    atm_by["dte"] = (pd.to_datetime(atm_by["expirDate"])
                     - pd.to_datetime(atm_by["tradeDate"])).dt.days

    print("Interpolating across DTE -> 30-DTE ATM IV ...")
    iv30 = (atm_by.groupby("tradeDate")
                  .apply(per_date_iv30, include_groups=False)
                  .reset_index(name="F_iv_atm_30"))

    spot_by_day = (clean.group_by("tradeDate")
                        .agg(pl.col("spotPrice").first().alias("S"))
                        .sort("tradeDate").to_pandas())

    panel = iv30.merge(spot_by_day, on="tradeDate", how="left")
    panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
    panel = panel.sort_values("tradeDate").reset_index(drop=True)
    panel["F_iv_rank_252"] = (panel["F_iv_atm_30"]
                              .rolling(RANK_LOOKBACK, min_periods=60)
                              .rank(pct=True))

    # 2) VIX features
    print("Merging VIX/VXN features ...")
    vix = pd.read_parquet(vix_path)
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vxn_vs_vix"] = vix["vxn"] - vix["vix"]
    vix["vxn_excess_rank_252"] = (vix["vxn_vs_vix"]
                                  .rolling(RANK_LOOKBACK, min_periods=60)
                                  .rank(pct=True))
    panel = panel.merge(
        vix[["date", "vix", "vxn_excess_rank_252"]].rename(columns={"date": "tradeDate"}),
        on="tradeDate", how="left",
    )

    # 3) Yield-curve features
    print("Merging Treasury / yield-curve features ...")
    tr = pd.read_parquet(treasury_path)
    tr["date"] = pd.to_datetime(tr["date"])
    tr = tr.sort_values("date").reset_index(drop=True)
    tr["slope_2s10s"] = tr["dgs10"] - tr["dgs2"]
    tr["slope_2s10s_rank_252"] = (tr["slope_2s10s"]
                                  .rolling(RANK_LOOKBACK, min_periods=60)
                                  .rank(pct=True))
    panel = panel.merge(
        tr[["date", "slope_2s10s", "slope_2s10s_rank_252"]].rename(columns={"date": "tradeDate"}),
        on="tradeDate", how="left",
    )

    panel.to_parquet(output_path, index=False)
    print(f"Wrote factor panel -> {output_path}  ({len(panel)} rows)")
    return panel


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--unified",  default="data/processed/spy_eod_unified.parquet")
    p.add_argument("--vix",      default="data/raw/vix_move.parquet")
    p.add_argument("--treasury", default="data/raw/treasury_rates.parquet")
    p.add_argument("--out",      default="data/processed/factor_panel_daily.parquet")
    args = p.parse_args()
    build_factor_panel(args.unified, args.vix, args.treasury, args.out)
