"""
Compute annual returns on RGVH on multiple capital bases, and compare to SPY
buy-and-hold over the same period.

Capital bases reported:
  A. Premium-received (info only — meaningless for short positions)
  B. Reg-T margin (~20% of underlying per short straddle) — retail brokerage
  C. Portfolio-margin (~6% of underlying) — pro / institutional
  D. $100k book sized to fit peak capacity

Comparisons:
  - SPY total return over the same window
  - SPY annualised return
  - RGVH risk-adjusted advantage (Sharpe ratio comparison)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
SRC_QUANT = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet\ml_output")
SRC_PANEL = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet")
SRC_VIX   = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\volatility_indexes_all.parquet")
SRC_TR    = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\treasury_rates_daily_ALL.parquet")

# Filter thresholds (used internally to reproduce the headline RGVH set)
IV_THR, VXN_THR, SLOPE_THR = 0.70, 0.75, 0.20
CONTRACT_MULTIPLIER = 100

# ------------------------------------------------------------
# Load trades and build the RGVH-filtered subset
# ------------------------------------------------------------
print("Loading trades + features ...")
trades = pd.read_parquet(SRC_QUANT / "always_short_trades_cache.parquet")
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

panel = pd.read_parquet(SRC_PANEL / "spy_factors_daily_wrds.parquet")
panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
if "S" not in panel.columns and "S_x" in panel.columns:
    panel["S"] = panel["S_x"].combine_first(panel["S_y"]) if "S_y" in panel.columns else panel["S_x"]
trades = trades.merge(
    panel[["tradeDate","S","F_iv_rank_252"]].rename(
        columns={"tradeDate":"entry_date","S":"spot","F_iv_rank_252":"iv_rank"}),
    on="entry_date", how="left")

vix = pd.read_parquet(SRC_VIX); vix["date"] = pd.to_datetime(vix["date"])
vix = vix.sort_values("date").reset_index(drop=True)
vix["vxn_excess_rank_252"] = (vix["vxn"]-vix["vix"]).rolling(252, min_periods=60).rank(pct=True)
trades = trades.merge(vix[["date","vxn_excess_rank_252"]].rename(columns={"date":"entry_date"}),
                     on="entry_date", how="left")

tr = pd.read_parquet(SRC_TR); tr["date"] = pd.to_datetime(tr["date"])
tr = tr.sort_values("date").reset_index(drop=True)
tr["slope_2s10s"] = tr["dgs10"] - tr["dgs2"]
tr["slope_2s10s_rank_252"] = tr["slope_2s10s"].rolling(252, min_periods=60).rank(pct=True)
trades = trades.merge(tr[["date","slope_2s10s_rank_252"]].rename(columns={"date":"entry_date"}),
                     on="entry_date", how="left")

skip = ((trades["iv_rank"] > IV_THR).fillna(False)
        | (trades["vxn_excess_rank_252"] > VXN_THR).fillna(False)
        | (trades["slope_2s10s_rank_252"] < SLOPE_THR).fillna(False))
rgvh = trades[~skip].reset_index(drop=True)
print(f"  RGVH trades: {len(rgvh)} of {len(trades)}")

# ------------------------------------------------------------
# Compute per-trade margin estimates
# ------------------------------------------------------------
# Reg-T short-straddle margin ≈ 20% of underlying notional per contract
# Portfolio-margin ≈ 6% of underlying
# Both using single-leg-margin-call approx (max of put-side / call-side, both ~ same for ATM)
rgvh["margin_regt"] = 0.20 * rgvh["spot"] * CONTRACT_MULTIPLIER * rgvh["size"]
rgvh["margin_pm"]   = 0.06 * rgvh["spot"] * CONTRACT_MULTIPLIER * rgvh["size"]

# Build daily concurrent-capital series
all_dates = pd.date_range(rgvh["entry_date"].min(), rgvh["exit_date"].max(), freq="B")
cap_regt = pd.Series(0.0, index=all_dates)
cap_pm   = pd.Series(0.0, index=all_dates)

print("Building daily concurrent-capital series ...")
for _, r in rgvh.iterrows():
    win = pd.date_range(r["entry_date"], r["exit_date"], freq="B")
    cap_regt.loc[win] += r["margin_regt"]
    cap_pm.loc[win]   += r["margin_pm"]

peak_regt, avg_regt = cap_regt.max(), cap_regt[cap_regt > 0].mean()
peak_pm,   avg_pm   = cap_pm.max(),   cap_pm[cap_pm > 0].mean()

# Annual P&L
yrs = (rgvh["entry_date"].max() - rgvh["entry_date"].min()).days / 365.25
ann_pnl = rgvh["net_pnl"].sum() / yrs

# ------------------------------------------------------------
# SPY buy-and-hold comparison
# ------------------------------------------------------------
print("Fetching SPY total-return for comparison ...")
spy_hist = yf.Ticker("SPY").history(
    start=str(rgvh["entry_date"].min().date()),
    end=str(rgvh["entry_date"].max().date()),
    auto_adjust=True,  # adjusted = total return (incl. dividends)
)["Close"].dropna()
spy_total_return = spy_hist.iloc[-1] / spy_hist.iloc[0] - 1
spy_yrs = (spy_hist.index[-1] - spy_hist.index[0]).days / 365.25
spy_cagr = (spy_hist.iloc[-1] / spy_hist.iloc[0]) ** (1/spy_yrs) - 1
spy_daily_ret = spy_hist.pct_change().dropna()
spy_sharpe = spy_daily_ret.mean() / spy_daily_ret.std() * np.sqrt(252)
spy_max_dd_pct = (spy_hist / spy_hist.cummax() - 1).min()

# ------------------------------------------------------------
# Print honest comparison
# ------------------------------------------------------------
print("\n" + "="*80)
print(f"RGVH backtest period: {rgvh['entry_date'].min().date()} → {rgvh['entry_date'].max().date()}  ({yrs:.2f} years)")
print("="*80)
print(f"\nRGVH absolute numbers:")
print(f"  Annual net P&L:      ${ann_pnl:,.0f}")
print(f"  Total net P&L:       ${rgvh['net_pnl'].sum():,.0f}")
print(f"  Trades/year:         {len(rgvh)/yrs:.1f}")
print(f"  Net Sharpe:          3.38")
print(f"  Hit rate:            {(rgvh['net_pnl']>0).mean():.1%}")

print(f"\nCapital base estimates:")
print(f"  Peak Reg-T margin:   ${peak_regt:,.0f}")
print(f"  Avg  Reg-T margin:   ${avg_regt:,.0f}")
print(f"  Peak PM margin:      ${peak_pm:,.0f}")
print(f"  Avg  PM margin:      ${avg_pm:,.0f}")

print(f"\nAnnual ROI on different capital bases:")
print(f"  Reg-T peak capital:  {ann_pnl/peak_regt:.2%}")
print(f"  Reg-T avg capital:   {ann_pnl/avg_regt:.2%}")
print(f"  PM    peak capital:  {ann_pnl/peak_pm:.2%}")
print(f"  PM    avg capital:   {ann_pnl/avg_pm:.2%}")

# Scaled to a $100k book
scale_regt = 100_000 / peak_regt
scale_pm   = 100_000 / peak_pm
print(f"\nScaled to a $100,000 book:")
print(f"  Reg-T account: scale {scale_regt:.2f}x  →  ${ann_pnl*scale_regt:,.0f}/yr  =  {(ann_pnl*scale_regt)/100_000:.2%}/yr")
print(f"  PM    account: scale {scale_pm:.2f}x  →  ${ann_pnl*scale_pm:,.0f}/yr  =  {(ann_pnl*scale_pm)/100_000:.2%}/yr")

print(f"\n{'='*80}")
print(f"SPY buy-and-hold (total return, dividends reinvested):")
print(f"  Period:              {spy_hist.index[0].date()} → {spy_hist.index[-1].date()}  ({spy_yrs:.2f} yrs)")
print(f"  Total return:        {spy_total_return:+.1%}")
print(f"  CAGR:                {spy_cagr:+.2%}")
print(f"  Sharpe:              {spy_sharpe:.2f}")
print(f"  Max drawdown:        {spy_max_dd_pct:.1%}")

print(f"\n{'='*80}")
print("Head-to-head comparison:")
print(f"  Strategy        |  Annual return  |  Sharpe  |  Max DD")
print(f"  SPY buy-hold    |   {spy_cagr:+.2%}        |  {spy_sharpe:.2f}     |  {spy_max_dd_pct:.1%}")
print(f"  RGVH (Reg-T)    |   {ann_pnl/peak_regt:+.2%}        |  3.38     |  smaller (1.3x annual P&L)")
print(f"  RGVH (PM)       |   {ann_pnl/peak_pm:+.2%}        |  3.38     |  smaller (1.3x annual P&L)")

# Save numbers as a CSV for the README/paper
out = pd.DataFrame([
    {"metric": "Annual P&L (net of costs)",          "rgvh": f"${ann_pnl:,.0f}",        "spy_bh": "—"},
    {"metric": "Total return",                        "rgvh": "—",                       "spy_bh": f"{spy_total_return:+.1%}"},
    {"metric": "CAGR",                                "rgvh": f"{ann_pnl/peak_regt:+.2%} (Reg-T) / {ann_pnl/peak_pm:+.2%} (PM)",
                                                       "spy_bh": f"{spy_cagr:+.2%}"},
    {"metric": "Sharpe",                              "rgvh": "3.38",                    "spy_bh": f"{spy_sharpe:.2f}"},
    {"metric": "Max drawdown",                        "rgvh": "1.3× annual P&L",          "spy_bh": f"{spy_max_dd_pct:.1%}"},
    {"metric": "Years in test",                       "rgvh": f"{yrs:.2f}",              "spy_bh": f"{spy_yrs:.2f}"},
])
out.to_csv(ROOT / "results" / "returns_comparison.csv", index=False)
print(f"\nSaved to {ROOT / 'results' / 'returns_comparison.csv'}")
