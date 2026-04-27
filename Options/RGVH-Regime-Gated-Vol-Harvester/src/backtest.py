"""
Short-vol straddle backtest with daily delta-hedging and realistic costs.

Pipeline per entry date:
  1. Find an ATM 22-DTE SPY straddle (call_delta ≈ 0.5)
  2. Sell the straddle at the bid (or use mid + half-spread debit on entry)
  3. Initial delta hedge in SPY
  4. Each subsequent business day until exit:
       - Mark-to-market the straddle
       - Mark-to-market the SPY hedge
       - Rebalance hedge if |drift| > HEDGE_DRIFT_THR
  5. Close the position after HOLD_DAYS or 1 day before expiry, whichever first
  6. Apply costs: bid-ask spread (entry + exit), commission per leg, hedge slippage

The simulator returns a DataFrame of trades with columns:
    entry_date, exit_date, size_contracts, entry_vega, entry_mid,
    opt_pnl, hedge_pnl, gross_pnl, total_cost, net_pnl

`run_baseline_short_short_vol` simulates an unfiltered always-short trade per OOF
date — its trade list is then filtered by `filters.build_skip_mask` for the
RGVH evaluation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .config import (
    TARGET_DTE, DTE_MIN, DTE_MAX, HOLD_DAYS, TARGET_VEGA_USD,
    HEDGE_DRIFT_THR, COMMISSION_PER_LEG_USD, SPY_HEDGE_SLIP_PER_SHARE,
    CONTRACT_MULTIPLIER,
)


# Dict-tuple positional layout for fast lookup
_C_MID, _C_DELTA, _C_VEGA, _C_HSC, _C_HSP, _C_SPOT, _C_DTE = range(7)


def _build_lookup(opts: pd.DataFrame) -> dict:
    """Build O(1) (date, expir, strike) -> contract array lookup."""
    td = opts["tradeDate"].tolist()
    ed = opts["expirDate"].tolist()
    sk = opts["strike"].astype(float).tolist()
    cols = ["straddle_mid", "straddle_delta", "straddle_vega",
            "half_spread_c", "half_spread_p", "spotPrice", "dte"]
    values = opts[cols].values
    return {(td[i], ed[i], sk[i]): values[i] for i in range(len(td))}


def _atm_picks(opts: pd.DataFrame) -> dict:
    """Per-day ATM 22-DTE pick: returns dict[date] -> (expir, strike)."""
    out: dict = {}
    for d, grp in opts.groupby("tradeDate"):
        sub = grp[(grp["dte"] >= DTE_MIN) & (grp["dte"] <= DTE_MAX)]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["dd"] = (sub["dte"] - TARGET_DTE).abs()
        be = sub.groupby("expirDate")["dd"].min().sort_values().index[0]
        sub2 = sub[sub["expirDate"] == be].copy()
        sub2["dlt"] = (sub2["delta"] - 0.5).abs()
        row = sub2.sort_values("dlt").iloc[0]
        out[pd.Timestamp(d)] = (pd.Timestamp(row["expirDate"]), float(row["strike"]))
    return out


def _prepare_options(opts: pd.DataFrame) -> pd.DataFrame:
    """Add convenience columns (mids, half-spreads, straddle quantities)."""
    opts = opts.copy()
    opts["tradeDate"] = pd.to_datetime(opts["tradeDate"])
    opts["expirDate"] = pd.to_datetime(opts["expirDate"])
    opts["call_mid"] = (opts["callBidPrice"] + opts["callAskPrice"]) / 2
    opts["put_mid"]  = (opts["putBidPrice"]  + opts["putAskPrice"])  / 2
    # Fall back to model value when bid is missing
    opts.loc[opts["callBidPrice"] <= 0, "call_mid"] = opts.loc[opts["callBidPrice"] <= 0, "callValue"]
    opts.loc[opts["putBidPrice"]  <= 0, "put_mid"]  = opts.loc[opts["putBidPrice"]  <= 0, "putValue"]
    opts["call_ask_eff"] = np.where(opts["callAskPrice"] > 0, opts["callAskPrice"], opts["callValue"] * 1.01)
    opts["call_bid_eff"] = np.where(opts["callBidPrice"] > 0, opts["callBidPrice"], opts["callValue"] * 0.99)
    opts["put_ask_eff"]  = np.where(opts["putAskPrice"]  > 0, opts["putAskPrice"],  opts["putValue"] * 1.01)
    opts["put_bid_eff"]  = np.where(opts["putBidPrice"]  > 0, opts["putBidPrice"],  opts["putValue"] * 0.99)
    opts["half_spread_c"] = np.maximum(0.0, (opts["call_ask_eff"] - opts["call_bid_eff"]) / 2)
    opts["half_spread_p"] = np.maximum(0.0, (opts["put_ask_eff"]  - opts["put_bid_eff"])  / 2)
    opts["straddle_mid"]   = opts["call_mid"] + opts["put_mid"]
    opts["straddle_delta"] = 2 * opts["delta"] - 1
    opts["straddle_vega"]  = 2 * opts["vega"]
    return opts


def run_baseline_short_vol(
    opts_path: Path | str,
    panel_path: Path | str,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Run the baseline (no-filter) always-short-vol simulation across every day
    in the factor panel that has a valid iv_rank value (i.e. after burn-in).

    Returns the trade DataFrame.
    """
    opts_path  = Path(opts_path)
    panel_path = Path(panel_path)

    opts = pd.read_parquet(opts_path, columns=[
        "tradeDate", "expirDate", "strike", "dte", "delta", "vega",
        "callBidPrice", "callAskPrice", "callValue",
        "putBidPrice",  "putAskPrice",  "putValue",  "spotPrice",
    ])
    opts = _prepare_options(opts)
    cdict = _build_lookup(opts)
    atm = _atm_picks(opts)

    spot_series = opts.groupby("tradeDate")["spotPrice"].first()
    trading_days = [pd.Timestamp(d) for d in spot_series.index]
    day_idx = {d: i for i, d in enumerate(trading_days)}

    panel = pd.read_parquet(panel_path)
    panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
    oof_dates = panel.dropna(subset=["F_iv_rank_252"])["tradeDate"]

    print(f"Simulating short-vol trades on {len(oof_dates)} OOF dates ...")
    trade_rows: list[dict] = []
    for d_pd in oof_dates:
        d = pd.Timestamp(d_pd)
        if d not in atm:
            continue
        expir, strike = atm[d]
        entry = cdict.get((d, expir, strike))
        if entry is None:
            continue
        entry_mid   = entry[_C_MID]
        entry_delta = entry[_C_DELTA]
        entry_vega  = entry[_C_VEGA]
        if not np.isfinite(entry_mid) or entry_mid <= 0:
            continue
        if not np.isfinite(entry_delta) or not np.isfinite(entry_vega):
            continue
        entry_hsc, entry_hsp = entry[_C_HSC], entry[_C_HSP]
        entry_spot = entry[_C_SPOT]
        size = max(TARGET_VEGA_USD / max(entry_vega * 100, 1.0), 0.05)
        if d not in day_idx:
            continue
        i_entry = day_idx[d]
        i_max = min(i_entry + HOLD_DAYS, len(trading_days) - 1)
        while i_max > i_entry and trading_days[i_max] >= expir:
            i_max -= 1

        direction = -1
        hedge_shares = -direction * entry_delta * CONTRACT_MULTIPLIER * size
        opt_pnl = hedge_pnl = 0.0
        hedge_cost = abs(hedge_shares) * SPY_HEDGE_SLIP_PER_SHARE
        prev_spot, prev_straddle = entry_spot, entry_mid
        last_key = (d, expir, strike)
        for i in range(i_entry + 1, i_max + 1):
            d_i = trading_days[i]
            c = cdict.get((d_i, expir, strike))
            if c is None:
                continue
            spot_d, straddle_d, new_delta = c[_C_SPOT], c[_C_MID], c[_C_DELTA]
            if not np.isfinite(straddle_d):
                continue
            opt_pnl   += direction * (straddle_d - prev_straddle) * CONTRACT_MULTIPLIER * size
            hedge_pnl += hedge_shares * (spot_d - prev_spot)
            desired = -direction * new_delta * CONTRACT_MULTIPLIER * size
            drift = desired - hedge_shares
            if abs(drift) > HEDGE_DRIFT_THR * CONTRACT_MULTIPLIER * size:
                hedge_cost += abs(drift) * SPY_HEDGE_SLIP_PER_SHARE
                hedge_shares = desired
            prev_spot, prev_straddle = spot_d, straddle_d
            last_key = (d_i, expir, strike)

        last_c = cdict.get(last_key)
        exit_hs = (last_c[_C_HSC] + last_c[_C_HSP]) if last_c is not None else (entry_hsc + entry_hsp)
        spread_cost = (entry_hsc + entry_hsp + exit_hs) * CONTRACT_MULTIPLIER * size
        commission = 4 * COMMISSION_PER_LEG_USD * size
        gross = opt_pnl + hedge_pnl
        total_cost = spread_cost + commission + hedge_cost
        trade_rows.append(dict(
            entry_date=d, exit_date=trading_days[i_max],
            size_contracts=size, entry_vega=entry_vega, entry_mid=entry_mid,
            opt_pnl=opt_pnl, hedge_pnl=hedge_pnl, gross_pnl=gross,
            total_cost=total_cost, net_pnl=gross - total_cost,
        ))

    trades = pd.DataFrame(trade_rows)
    print(f"  simulated {len(trades)} trades")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(output_path)
        print(f"  saved -> {output_path}")
    return trades


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--opts",  default="data/processed/spy_eod_unified.parquet")
    p.add_argument("--panel", default="data/processed/factor_panel_daily.parquet")
    p.add_argument("--out",   default="data/processed/baseline_trades.parquet")
    args = p.parse_args()
    run_baseline_short_vol(args.opts, args.panel, args.out)
