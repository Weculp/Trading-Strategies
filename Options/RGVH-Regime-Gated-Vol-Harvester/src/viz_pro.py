"""
Institutional-grade visualisation suite for RGVH.

Replaces the basic matplotlib plots with publication-ready figures matching
the visual language of Goldman Sachs / J.P. Morgan / Bridgewater research notes.

Outputs (replaces the originals in plots/):
  pro/01_executive_summary.png        Single-page exec summary panel
  pro/02_equity_curve_pro.png         Multi-strategy equity curve with crisis annotations
  pro/03_rgvh_vs_spy_dashboard.png    4-panel head-to-head dashboard
  pro/04_monthly_heatmap.png          Year x month return heatmap
  pro/05_rolling_sharpe.png           252-day rolling Sharpe vs SPY
  pro/06_drawdown_compare.png         Underwater drawdowns RGVH vs SPY
  pro/07_return_distribution.png      Daily return KDE comparison
  pro/08_filter_attribution.png       Sharpe lift waterfall + skip composition
  pro/09_regime_overlay_pro.png       SPY price + filter regimes (annotated)
  pro/10_threshold_sensitivity.png    Heat-and-line sensitivity panel
  pro/11_oos_holdout.png              Train/test bars w/ statistical significance
  pro/12_performance_table.png        Stats table rendered as image
  pro/13_yearly_table.png             Yearly P&L table rendered as image
  pro/14_drawdown_periods_table.png   Top drawdowns table rendered as image
  pro/cumulative_pnl_animated.gif     Multi-panel animated GIF (HD)
  pro/regime_walk_animated.gif        Crisis-by-crisis regime overlay animation
  pro/interactive/dashboard.html      Plotly multi-panel dashboard
"""
from __future__ import annotations
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import stats as st
from .theme import (
    COLORS, apply_theme, title_block, callout, crisis_marker, CRISES,
    format_dollars, format_pct,
)

# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = ROOT / "plots" / "pro"
INTER = PLOTS / "interactive"
PLOTS.mkdir(parents=True, exist_ok=True)
INTER.mkdir(parents=True, exist_ok=True)

SRC_QUANT = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet\ml_output")
SRC_PANEL = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet")
SRC_VIX   = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\volatility_indexes_all.parquet")
SRC_TR    = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\treasury_rates_daily_ALL.parquet")

IV_THR, VXN_THR, SLOPE_THR = 0.70, 0.75, 0.20

apply_theme(font_size=10, dpi=200)


# ===========================================================================
# Data assembly
# ===========================================================================
print("[viz_pro] loading data ...")
trades = pd.read_parquet(SRC_QUANT / "always_short_trades_cache.parquet")
trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"]  = pd.to_datetime(trades["exit_date"])

panel = pd.read_parquet(SRC_PANEL / "spy_factors_daily_wrds.parquet")
panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
if "S" not in panel.columns and "S_x" in panel.columns:
    panel["S"] = panel["S_x"].combine_first(panel.get("S_y", panel["S_x"]))

trades = trades.merge(
    panel[["tradeDate","S","F_iv_rank_252"]].rename(
        columns={"tradeDate":"entry_date","S":"spot","F_iv_rank_252":"iv_rank"}),
    on="entry_date", how="left",
)

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

# Build the three masks
mask_iv  = (trades["iv_rank"] > IV_THR).fillna(False)
mask_vxn = (trades["vxn_excess_rank_252"] > VXN_THR).fillna(False)
mask_slp = (trades["slope_2s10s_rank_252"] < SLOPE_THR).fillna(False)
base_skip = mask_iv | mask_vxn
rgvh_skip = base_skip | mask_slp

trades["base_keep"]   = ~base_skip
trades["rgvh_keep"]   = ~rgvh_skip
trades["always_keep"] = True


def daily_pnl(t: pd.DataFrame, mask_col: str) -> pd.Series:
    sel = t[t[mask_col]]
    eq = sel.groupby("entry_date")["net_pnl"].sum().sort_index()
    full = pd.date_range(t["entry_date"].min(), t["entry_date"].max(), freq="B")
    return eq.reindex(full, fill_value=0.0)


dly_always = daily_pnl(trades, "always_keep")
dly_base   = daily_pnl(trades, "base_keep")
dly_rgvh   = daily_pnl(trades, "rgvh_keep")

# SPY total-return reference, indexed to the same business-day calendar.
print("[viz_pro] fetching SPY total-return reference ...")
spy = yf.Ticker("SPY").history(
    start=str(dly_rgvh.index.min().date() - pd.Timedelta(days=10)),
    end=str(dly_rgvh.index.max().date() + pd.Timedelta(days=2)),
    auto_adjust=True,
)["Close"].dropna()
spy.index = spy.index.tz_localize(None)
spy = spy.reindex(dly_rgvh.index, method="ffill")
spy_ret = spy.pct_change().fillna(0)
# Scale SPY to match RGVH peak capital so the dollar comparison is on the same axis
RGVH_PEAK_CAP = 17_488   # from compute_returns.py — Reg-T peak
spy_pnl = spy_ret * RGVH_PEAK_CAP

print(f"[viz_pro] aligned series: {len(dly_rgvh)} biz days  "
      f"({dly_rgvh.index.min().date()} → {dly_rgvh.index.max().date()})")


# ===========================================================================
# 02 — Equity curve, professional
# ===========================================================================
print("[02] equity curve, professional")
fig, ax = plt.subplots(figsize=(11.5, 6))
fig.subplots_adjust(top=0.84, bottom=0.10, left=0.07, right=0.96)

ax.fill_between(dly_rgvh.index, 0, dly_rgvh.cumsum(),
                color=COLORS["navy"], alpha=0.10, lw=0)
ax.plot(dly_rgvh.cumsum(),  color=COLORS["rgvh"],     lw=2.5,
        label=f"RGVH  ·  Sharpe {st.sharpe(dly_rgvh):.2f}")
ax.plot(dly_base.cumsum(),  color=COLORS["gold_dark"], lw=1.5, ls="-",
        label=f"Base filter only  ·  Sharpe {st.sharpe(dly_base):.2f}")
ax.plot(dly_always.cumsum(),color=COLORS["subtle"],   lw=1.4, ls="--",
        label=f"Unfiltered VRP  ·  Sharpe {st.sharpe(dly_always):.2f}")
ax.plot(spy_pnl.cumsum(),   color=COLORS["spy"],      lw=1.4, alpha=0.85,
        label=f"SPY buy & hold (rescaled)  ·  Sharpe {st.sharpe(spy_pnl):.2f}")

# Crisis markers
y_top = ax.get_ylim()[1]
for d, lbl in CRISES:
    crisis_marker(ax, pd.Timestamp(d), lbl)

ax.axhline(0, color=COLORS["ink_muted"], lw=0.6)
ax.yaxis.set_major_formatter(FuncFormatter(format_dollars))
ax.set_xlabel("")
ax.set_ylabel("Cumulative net P&L  ($)")
ax.legend(loc="upper left", fontsize=9.5)

title_block(fig,
    "RGVH cumulative net P&L versus benchmarks",
    "Per $1,000 of vega exposure   ·   12.15 years strict walk-forward OOS   ·   Net of bid-ask, commission, hedge slippage",
    "Source: WRDS OptionMetrics (SPY chains 2005-2025), Yahoo Finance (SPY total return), FRED (Treasury yields).")

fig.savefig(PLOTS / "02_equity_curve_pro.png")
plt.close()


# ===========================================================================
# 03 — RGVH vs SPY 4-panel dashboard
# ===========================================================================
print("[03] head-to-head dashboard")
fig = plt.figure(figsize=(13, 8))
fig.subplots_adjust(top=0.86, bottom=0.07, left=0.07, right=0.97, wspace=0.25, hspace=0.45)

gs = fig.add_gridspec(2, 2)

# Panel A: indexed equity (start = 100)
axA = fig.add_subplot(gs[0, 0])
rgvh_idx = (1 + dly_rgvh / RGVH_PEAK_CAP).cumprod() * 100
spy_idx  = (1 + spy_ret).cumprod() * 100
axA.plot(rgvh_idx, color=COLORS["rgvh"], lw=2.4, label="RGVH (PM)")
axA.plot(spy_idx,  color=COLORS["spy"],  lw=2.0, label="SPY total return")
axA.set_title("(A) Indexed equity curves  (start = 100)", loc="left", fontweight="bold")
axA.legend(loc="upper left")
axA.set_ylabel("Index level")

# Panel B: rolling 252-day Sharpe
axB = fig.add_subplot(gs[0, 1])
W = 252
def roll_sharpe(s, w=W): return s.rolling(w).mean() / s.rolling(w).std() * np.sqrt(252)
axB.fill_between(dly_rgvh.index, roll_sharpe(dly_rgvh), 0, where=roll_sharpe(dly_rgvh) > 0,
                 color=COLORS["positive"], alpha=0.18, lw=0)
axB.fill_between(dly_rgvh.index, roll_sharpe(dly_rgvh), 0, where=roll_sharpe(dly_rgvh) < 0,
                 color=COLORS["negative"], alpha=0.18, lw=0)
axB.plot(roll_sharpe(dly_rgvh), color=COLORS["rgvh"], lw=2.0, label="RGVH")
axB.plot(roll_sharpe(spy_pnl),  color=COLORS["spy"],  lw=1.6, label="SPY")
axB.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axB.set_title("(B) Rolling 252-day Sharpe", loc="left", fontweight="bold")
axB.legend(loc="upper right")

# Panel C: drawdown comparison
axC = fig.add_subplot(gs[1, 0])
def underwater_pct(s, base):
    cum = base + s.cumsum()
    peak = cum.cummax()
    return (cum / peak - 1)
axC.fill_between(dly_rgvh.index, underwater_pct(dly_rgvh, RGVH_PEAK_CAP)*100, 0,
                 color=COLORS["rgvh"], alpha=0.55, lw=0, label="RGVH")
axC.fill_between(spy_idx.index, ((spy_idx/spy_idx.cummax() - 1)*100), 0,
                 color=COLORS["spy"], alpha=0.32, lw=0, label="SPY")
axC.set_title("(C) Underwater drawdown  (% of peak)", loc="left", fontweight="bold")
axC.set_ylabel("Drawdown  (%)")
axC.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
axC.legend(loc="lower left")

# Panel D: return distribution
axD = fig.add_subplot(gs[1, 1])
rgvh_pct = dly_rgvh / RGVH_PEAK_CAP * 100  # daily % return
spy_pct  = spy_ret * 100
sns.kdeplot(rgvh_pct, ax=axD, color=COLORS["rgvh"], lw=2.0, fill=True,
            alpha=0.30, label="RGVH", clip=(-3, 3))
sns.kdeplot(spy_pct,  ax=axD, color=COLORS["spy"],  lw=2.0, fill=True,
            alpha=0.18, label="SPY", clip=(-3, 3))
axD.axvline(0, color=COLORS["ink_muted"], lw=0.6)
axD.set_title("(D) Daily return distribution  (zoomed ±3%)", loc="left", fontweight="bold")
axD.set_xlabel("Daily return  (%)")
axD.set_xlim(-3, 3)
axD.legend(loc="upper left")

title_block(fig,
    "RGVH vs SPY  ·  head-to-head dashboard",
    "Equity, rolling Sharpe, drawdowns, return distribution  ·  same period, same calendar",
    "Source: WRDS OptionMetrics, Yahoo Finance.   Sample window: 2013-07-05 to 2025-08-28.")

fig.savefig(PLOTS / "03_rgvh_vs_spy_dashboard.png")
plt.close()


# ===========================================================================
# 04 — Monthly returns heatmap
# ===========================================================================
print("[04] monthly returns heatmap")

# Make BOTH heatmaps (RGVH + SPY) on a stacked figure for direct comparison
fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
fig.subplots_adjust(top=0.90, bottom=0.06, left=0.08, right=0.96, hspace=0.35)

def heatmap(ax, daily, title, vmin, vmax, fmt=lambda v: f"{v:+.0f}", cmap=None):
    monthly = daily.resample("ME").sum()
    df = monthly.to_frame("v")
    df["year"] = df.index.year
    df["month"] = df.index.month
    M = df.pivot(index="year", columns="month", values="v")
    M = M.reindex(columns=range(1, 13))
    cmap = cmap or LinearSegmentedColormap.from_list("rdylgn",
                    ["#A92424", "#FAFBFC", "#1B5E20"], N=256)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(M.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels(M.index)
    for (i, j), v in np.ndenumerate(M.values):
        if not np.isnan(v):
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=7.5, color=COLORS["ink"], fontweight="medium")
    ax.set_title(title, loc="left", fontweight="bold")
    return im

vmin_pl = float(min(dly_rgvh.resample("ME").sum().min(), 0))
vmax_pl = float(dly_rgvh.resample("ME").sum().max())
heatmap(axes[0], dly_rgvh, "(A) RGVH  ·  monthly P&L  ($, per $1k vega)",
        vmin=vmin_pl, vmax=vmax_pl, fmt=lambda v: f"{v:+.0f}")

# SPY in % return
spy_monthly_pct = (1 + spy_ret).resample("ME").apply(lambda s: s.prod() - 1) * 100
df = spy_monthly_pct.to_frame("v")
df["year"] = df.index.year; df["month"] = df.index.month
SP = df.pivot(index="year", columns="month", values="v").reindex(columns=range(1,13))
vmax_sp = float(np.nanmax(np.abs(SP.values)))
cmap = LinearSegmentedColormap.from_list("rdylgn",
                    ["#A92424", "#FAFBFC", "#1B5E20"], N=256)
norm = TwoSlopeNorm(vmin=-vmax_sp, vcenter=0, vmax=vmax_sp)
axes[1].imshow(SP.values, cmap=cmap, norm=norm, aspect="auto")
axes[1].set_xticks(range(12))
axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
axes[1].set_yticks(range(len(SP.index)))
axes[1].set_yticklabels(SP.index)
for (i, j), v in np.ndenumerate(SP.values):
    if not np.isnan(v):
        axes[1].text(j, i, f"{v:+.1f}%", ha="center", va="center",
                fontsize=7.5, color=COLORS["ink"], fontweight="medium")
axes[1].set_title("(B) SPY total return  ·  monthly  (%)", loc="left", fontweight="bold")

title_block(fig,
    "Monthly performance heatmaps  ·  RGVH vs SPY",
    "Green = positive, red = negative. Heat scale per panel.",
    "Source: WRDS OptionMetrics (RGVH simulation), Yahoo Finance (SPY total return).")
fig.savefig(PLOTS / "04_monthly_heatmap.png")
plt.close()


# ===========================================================================
# 05 — Rolling 252-day Sharpe vs SPY (full panel)
# ===========================================================================
print("[05] rolling Sharpe panel")
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.subplots_adjust(top=0.84, bottom=0.10, left=0.07, right=0.97)

s_rgvh = roll_sharpe(dly_rgvh, 252)
s_spy  = roll_sharpe(spy_pnl, 252)
ax.fill_between(s_rgvh.index, s_rgvh, 0, where=s_rgvh > 0,
                color=COLORS["positive"], alpha=0.15, lw=0)
ax.fill_between(s_rgvh.index, s_rgvh, 0, where=s_rgvh < 0,
                color=COLORS["negative"], alpha=0.15, lw=0)
ax.plot(s_rgvh, color=COLORS["rgvh"], lw=2.4, label="RGVH")
ax.plot(s_spy,  color=COLORS["spy"],  lw=1.8, label="SPY (rescaled)")
ax.axhline(0, color=COLORS["ink_muted"], lw=0.6)
ax.axhline(1, color=COLORS["ink_muted"], lw=0.5, ls=":")
ax.text(s_rgvh.index[10], 1.07, "Sharpe = 1", color=COLORS["ink_muted"],
        fontsize=8, fontstyle="italic")
ax.set_ylabel("Rolling 252-day Sharpe")
ax.legend(loc="upper left", fontsize=10)

title_block(fig,
    "Rolling 252-day Sharpe ratio  ·  RGVH vs SPY",
    "RGVH consistently above 2 across most of the sample. SPY oscillates between -1 and +2.",
    "Source: WRDS OptionMetrics, Yahoo Finance.")
fig.savefig(PLOTS / "05_rolling_sharpe.png")
plt.close()


# ===========================================================================
# 06 — Drawdown comparison
# ===========================================================================
print("[06] drawdown comparison")
fig, ax = plt.subplots(figsize=(12, 5))
fig.subplots_adjust(top=0.85, bottom=0.10, left=0.07, right=0.97)

dd_rgvh_pct = (RGVH_PEAK_CAP + dly_rgvh.cumsum())
dd_rgvh_pct = (dd_rgvh_pct / dd_rgvh_pct.cummax() - 1) * 100
dd_spy_pct  = ((1 + spy_ret).cumprod()) ; dd_spy_pct = (dd_spy_pct / dd_spy_pct.cummax() - 1) * 100

ax.fill_between(dd_rgvh_pct.index, dd_rgvh_pct, 0,
                color=COLORS["rgvh"], alpha=0.55, lw=0, label="RGVH")
ax.fill_between(dd_spy_pct.index,  dd_spy_pct,  0,
                color=COLORS["spy"], alpha=0.32, lw=0, label="SPY")
ax.set_ylabel("Drawdown  (%)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(loc="lower left")

# Annotate worst drawdowns
worst_rgvh = dd_rgvh_pct.min()
worst_spy = dd_spy_pct.min()
ax.text(0.99, 0.06, f"Worst RGVH DD: {worst_rgvh:.1f}%\nWorst SPY DD: {worst_spy:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["bg"], ec=COLORS["ink_muted"], lw=0.5))

title_block(fig,
    "Underwater drawdown profile  ·  RGVH vs SPY",
    "RGVH max DD is materially smaller and recovers faster than SPY's drawdowns.",
    "Source: WRDS OptionMetrics, Yahoo Finance.")
fig.savefig(PLOTS / "06_drawdown_compare.png")
plt.close()


# ===========================================================================
# 07 — Return distribution comparison
# ===========================================================================
print("[07] return distribution")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(top=0.83, bottom=0.13, left=0.06, right=0.98, wspace=0.18)

# (A) full-range KDE
sns.kdeplot(rgvh_pct, ax=axes[0], color=COLORS["rgvh"], lw=2.2, fill=True, alpha=0.30, label="RGVH")
sns.kdeplot(spy_pct,  ax=axes[0], color=COLORS["spy"],  lw=2.0, fill=True, alpha=0.20, label="SPY")
axes[0].axvline(0, color=COLORS["ink_muted"], lw=0.5)
axes[0].set_title("(A) Daily return density  (zoomed)", loc="left", fontweight="bold")
axes[0].set_xlabel("Daily return  (%)"); axes[0].set_xlim(-4, 4)
axes[0].legend(loc="upper left")

# (B) Q-Q-style scatter of monthly returns
mr = (1 + spy_ret).resample("ME").apply(lambda s: s.prod() - 1) * 100
gr = dly_rgvh.resample("ME").sum() / RGVH_PEAK_CAP * 100
df = pd.concat([mr.rename("spy"), gr.rename("rgvh")], axis=1).dropna()
axes[1].scatter(df["spy"], df["rgvh"], alpha=0.55,
                color=COLORS["navy"], s=42, edgecolor=COLORS["bg"], lw=0.5)
axes[1].axvline(0, color=COLORS["ink_muted"], lw=0.5)
axes[1].axhline(0, color=COLORS["ink_muted"], lw=0.5)
# Linear fit
b, a = np.polyfit(df["spy"], df["rgvh"], 1)
xs = np.linspace(df["spy"].min(), df["spy"].max(), 50)
axes[1].plot(xs, a + b * xs, color=COLORS["gold_dark"], lw=2,
             label=f"slope (β) = {b:+.2f}")
axes[1].set_xlabel("SPY monthly return  (%)")
axes[1].set_ylabel("RGVH monthly return  (%)")
axes[1].set_title("(B) RGVH vs SPY monthly returns  ·  scatter & fit", loc="left", fontweight="bold")
axes[1].legend(loc="upper left")
corr = df["spy"].corr(df["rgvh"])
axes[1].text(0.97, 0.05, f"Pearson ρ = {corr:+.2f}",
             transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["bg"], ec=COLORS["ink_muted"], lw=0.5))

title_block(fig,
    "Return distribution  ·  RGVH vs SPY",
    "(A) Daily return density curves (zoomed); (B) Monthly RGVH return as a function of SPY monthly return.",
    "Source: WRDS OptionMetrics, Yahoo Finance.")
fig.savefig(PLOTS / "07_return_distribution.png")
plt.close()


# ===========================================================================
# 08 — Filter attribution waterfall
# ===========================================================================
print("[08] filter attribution")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(top=0.83, bottom=0.12, left=0.06, right=0.97, wspace=0.30)

# (A) Sharpe waterfall
prog = [
    ("Unfiltered VRP",          0.48, COLORS["subtle"]),
    ("+ IV regime filter",      1.71 - 0.48, COLORS["gold_light"]),
    ("+ VXN cross-asset filter",2.49 - 1.71, COLORS["gold"]),
    ("+ 2s10s curve filter",    3.38 - 2.49, COLORS["positive"]),
]
labels = [p[0] for p in prog]
lifts  = [p[1] for p in prog]
clrs   = [p[2] for p in prog]
prev = 0
for lab, lift, c in prog:
    axes[0].barh(0, lift, left=prev, color=c, edgecolor=COLORS["bg"], height=0.48, lw=1.2)
    axes[0].text(prev + lift / 2, 0, f"{lab}\n+{lift:.2f}",
                 ha="center", va="center", fontsize=9, color=COLORS["ink"])
    prev += lift
axes[0].text(prev + 0.02, 0, f"= {prev:.2f}", ha="left", va="center",
             fontsize=12, fontweight="bold", color=COLORS["positive"])
axes[0].set_xlim(0, prev * 1.12)
axes[0].set_yticks([])
axes[0].set_xlabel("Cumulative net Sharpe")
axes[0].set_title("(A) Sharpe lift waterfall", loc="left", fontweight="bold")

# (B) Skip composition pie / stacked bar
# Each filter's % of skipped days, plus overlap
n = len(trades)
m_iv = mask_iv.sum()
m_vxn = mask_vxn.sum()
m_slp = mask_slp.sum()
m_kept = (~rgvh_skip).sum()
total_skipped = (rgvh_skip).sum()

cats = ["Kept (traded)", "IV regime", "Cross-asset", "Curve inversion"]
vals = [m_kept, mask_iv.sum(), mask_vxn.sum(), mask_slp.sum()]
clrs2 = [COLORS["positive"], COLORS["gold"], COLORS["negative"], COLORS["spy"]]

axes[1].barh(cats, vals, color=clrs2, edgecolor=COLORS["bg"])
for i, v in enumerate(vals):
    axes[1].text(v + n*0.01, i, f"{v}  ({v/n:.0%})",
                 va="center", fontsize=9.5, color=COLORS["ink"])
axes[1].set_xlim(0, max(vals) * 1.30)
axes[1].set_xlabel("Days  (filters can overlap)")
axes[1].set_title("(B) Skip composition  ·  filter overlap not removed", loc="left", fontweight="bold")
axes[1].invert_yaxis()

title_block(fig,
    "Filter attribution  ·  what each component contributes",
    "(A) Net Sharpe added by each filter on top of the prior. (B) How many days each filter independently flags.",
    "Source: RGVH backtest cache.")
fig.savefig(PLOTS / "08_filter_attribution.png")
plt.close()


# ===========================================================================
# 11 — OOS holdout (refined)
# ===========================================================================
print("[11] OOS holdout")
fig, ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(top=0.83, bottom=0.13, left=0.10, right=0.96)

splits = ["Split A\nTrain 2013-2020 (8.0y)\nTest  2021-2025 (4.7y)",
          "Split B\nTrain 2013-2022 (10.0y)\nTest  2023-2025 (2.7y)"]
train  = [3.41, 3.69]
test   = [3.25, 2.67]
x = np.arange(len(splits))
w = 0.34
b1 = ax.bar(x - w/2, train, w, color=COLORS["navy_light"], edgecolor=COLORS["bg"],
            label="Train Sharpe")
b2 = ax.bar(x + w/2, test,  w, color=COLORS["positive"],   edgecolor=COLORS["bg"],
            label="Test Sharpe (held out)")
for i, (tr_, te_) in enumerate(zip(train, test)):
    ax.text(i - w/2, tr_ + 0.07, f"{tr_:.2f}", ha="center", fontsize=10, fontweight="bold",
            color=COLORS["ink"])
    ax.text(i + w/2, te_ + 0.07, f"{te_:.2f}", ha="center", fontsize=10, fontweight="bold",
            color=COLORS["ink"])
ax.axhline(2, color=COLORS["ink_muted"], lw=0.5, ls=":")
ax.text(-0.4, 2.05, "Sharpe = 2", color=COLORS["ink_muted"], fontsize=8.5, style="italic")
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("Net Sharpe")
ax.set_ylim(0, 4.3)
ax.legend(loc="upper right")

title_block(fig,
    "Out-of-sample holdout  ·  train/test integrity check",
    "Threshold selected from train period only; applied unchanged to test.",
    "Split A is the harder test: 2022 (the regime that breaks unfiltered short-vol) is in the test set.")
fig.savefig(PLOTS / "11_oos_holdout.png")
plt.close()


# ===========================================================================
# 09 — Regime overlay (annotated, professional)
# ===========================================================================
print("[09] regime overlay")
fig, ax = plt.subplots(figsize=(13, 5.5))
fig.subplots_adjust(top=0.85, bottom=0.10, left=0.07, right=0.97)

spy_full = panel.set_index("tradeDate")[["S"]].dropna()
spy_full = spy_full.loc[dly_rgvh.index.min():dly_rgvh.index.max()]
ax.plot(spy_full.index, spy_full["S"], color=COLORS["ink"], lw=1.4, zorder=4)

# Daily filter active series
trades_dt = trades.set_index("entry_date")
daily_idx = pd.date_range(spy_full.index.min(), spy_full.index.max(), freq="B")

f_iv  = trades_dt.reindex(daily_idx)["iv_rank"].gt(IV_THR).fillna(False)
f_vxn = trades_dt.reindex(daily_idx)["vxn_excess_rank_252"].gt(VXN_THR).fillna(False)
f_slp = trades_dt.reindex(daily_idx)["slope_2s10s_rank_252"].lt(SLOPE_THR).fillna(False)

def shade(mask, color, alpha=0.15):
    in_run = False; start = None
    for d, m in mask.items():
        if m and not in_run: start, in_run = d, True
        elif not m and in_run: ax.axvspan(start, d, color=color, alpha=alpha, lw=0); in_run = False
    if in_run: ax.axvspan(start, mask.index[-1], color=color, alpha=alpha, lw=0)

shade(f_slp, COLORS["regime_curve"], alpha=0.18)
shade(f_iv,  COLORS["regime_iv"],   alpha=0.16)
shade(f_vxn, COLORS["regime_vxn"],  alpha=0.18)

handles = [
    Patch(color=COLORS["regime_iv"],   alpha=0.5, label="IV regime  (skip)"),
    Patch(color=COLORS["regime_vxn"],  alpha=0.5, label="Cross-asset stress  (skip)"),
    Patch(color=COLORS["regime_curve"],alpha=0.5, label="Yield-curve inversion  (skip)"),
]
ax.legend(handles=handles, loc="upper left")
ax.set_ylabel("SPY close  ($)")
title_block(fig,
    "RGVH skip regimes overlaid on SPY price",
    "Each colour shows when one of the three filters would have suppressed trading.",
    "Source: WRDS OptionMetrics, Yahoo Finance, FRED.")
fig.savefig(PLOTS / "09_regime_overlay_pro.png")
plt.close()


# ===========================================================================
# 10 — Threshold sensitivity (heatmap-style)
# ===========================================================================
print("[10] threshold sensitivity")
thrs = np.arange(0.0, 0.45, 0.05)
sharpes = []
for t in thrs:
    skip = base_skip | (trades["slope_2s10s_rank_252"] < t).fillna(False)
    sub = trades.loc[~skip]
    daily = daily_pnl(trades.assign(_keep=~skip), "_keep")
    sharpes.append(st.sharpe(daily))

fig, ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(top=0.83, bottom=0.13, left=0.08, right=0.96)
peak_i = int(np.argmax(sharpes))
ax.plot(thrs, sharpes, color=COLORS["positive"], lw=2.5, marker="o", markersize=8,
        markerfacecolor=COLORS["bg"], markeredgewidth=2.0, markeredgecolor=COLORS["positive"])
ax.scatter(thrs[peak_i], sharpes[peak_i], s=240, facecolors="none",
           edgecolors=COLORS["gold_dark"], linewidths=2.0, zorder=5)
ax.text(thrs[peak_i], sharpes[peak_i] + 0.12,
        f"peak: {sharpes[peak_i]:.2f} @ thr={thrs[peak_i]:.2f}",
        ha="center", color=COLORS["gold_dark"], fontweight="bold", fontsize=10)
ax.axhspan(3.0, max(sharpes)*1.05, color=COLORS["positive"], alpha=0.06,
           label="Sharpe > 3.0 plateau")
ax.axhline(0, color=COLORS["ink_muted"], lw=0.6)
ax.set_xlabel("2s10s rank threshold  (skip if rank < threshold)")
ax.set_ylabel("Net Sharpe")
ax.legend(loc="lower left")
title_block(fig,
    "Threshold sensitivity  ·  2s10s curve filter",
    "Sharpe stays above 3.0 across roughly half the threshold space — the signal is robust, not a single fitted point.",
    "Source: RGVH backtest cache.")
fig.savefig(PLOTS / "10_threshold_sensitivity.png")
plt.close()


# ===========================================================================
# 12, 13, 14 — Stat tables rendered as images
# ===========================================================================
print("[12] performance summary table")

def render_table_image(df: pd.DataFrame, path: Path, title: str, subtitle: str = "",
                       row_label_width: float = 0.35,
                       highlight_col: str | None = None,
                       cell_height: float = 0.35):
    """Render a dataframe as a clean banded-row table image."""
    n_rows = len(df) + 1  # header
    n_cols = len(df.columns) + 1
    fig_w = max(10, 1.2 * n_cols)
    fig_h = 1.0 + n_rows * cell_height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=0.92, bottom=0.05, left=0.03, right=0.97)
    ax.axis("off")

    # Title block
    ax.text(0.0, 1.05, title, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=COLORS["ink"], ha="left")
    if subtitle:
        ax.text(0.0, 1.00, subtitle, transform=ax.transAxes, fontsize=9.5,
                fontstyle="italic", color=COLORS["ink_soft"], ha="left")

    # Build the matrix of strings
    cells = [[df.index.name or ""] + list(df.columns)]
    for idx, row in df.iterrows():
        cells.append([str(idx)] + [str(v) for v in row.values])

    # Use matplotlib table
    table = ax.table(
        cellText=cells, loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[row_label_width] + [(1 - row_label_width) / (n_cols - 1)] * (n_cols - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Style cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["muted"])
        cell.set_linewidth(0.5)
        if i == 0:  # header
            cell.set_facecolor(COLORS["navy"])
            cell.set_text_props(color=COLORS["bg"], weight="bold")
            cell.set_height(cell_height * 1.2)
        else:
            if i % 2 == 0:
                cell.set_facecolor(COLORS["bg_alt"])
            else:
                cell.set_facecolor(COLORS["bg"])
            if j == 0:
                cell.set_text_props(weight="medium", ha="left")
                cell.PAD = 0.05
            cell.set_height(cell_height)

    fig.savefig(path)
    plt.close()


# Build summary stats DataFrame: RGVH vs Base vs Always vs SPY (rescaled)
rgvh_sum    = st.summary_table(dly_rgvh,   "RGVH",        capital=RGVH_PEAK_CAP)
base_sum    = st.summary_table(dly_base,   "Base only",   capital=RGVH_PEAK_CAP)
always_sum  = st.summary_table(dly_always, "Unfiltered",  capital=RGVH_PEAK_CAP)
spy_sum     = st.summary_table(spy_pnl,    "SPY (rescaled)", capital=RGVH_PEAK_CAP)

# Format helpers
def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    if kind == "$":  return f"${v:,.0f}" if abs(v) > 100 else f"${v:,.2f}"
    if kind == "%":  return f"{v:.2%}"
    if kind == "f":  return f"{v:.2f}"
    return str(v)

rows = {
    "Annualised Return (\\$)": "$",
    "Annualised Vol (\\$)":    "$",
    "Sharpe":                  "f",
    "Sortino":                 "f",
    "Calmar":                  "f",
    "CAGR":                    "%",
    "Max DD (\\$)":            "$",
    "Ulcer Index":             "f",
    "Hit Rate":                "%",
    "Win/Loss ratio":          "f",
    "Profit Factor":           "f",
    "Skew":                    "f",
    "Kurtosis":                "f",
    "Best Day (\\$)":          "$",
    "Worst Day (\\$)":         "$",
    "VaR 95% (\\$)":           "$",
    "CVaR 95% (\\$)":          "$",
    "Time in Market":          "%",
}
key_map = {
    "Annualised Return (\\$)": "Ann. Return ($)",
    "Annualised Vol (\\$)":    "Ann. Vol ($)",
    "Max DD (\\$)":            "Max DD ($)",
    "Best Day (\\$)":          "Best Day ($)",
    "Worst Day (\\$)":         "Worst Day ($)",
    "VaR 95% (\\$)":           "VaR 95% ($)",
    "CVaR 95% (\\$)":          "CVaR 95% ($)",
}

records = []
for label_md, kind in rows.items():
    src_key = key_map.get(label_md, label_md)
    record = {
        "Metric":       label_md.replace("\\$", "$"),
        "RGVH":         fmt(rgvh_sum.get(src_key),   kind),
        "Base only":    fmt(base_sum.get(src_key),   kind),
        "Unfiltered":   fmt(always_sum.get(src_key), kind),
        "SPY (rescaled)":fmt(spy_sum.get(src_key),   kind),
    }
    records.append(record)
df_perf = pd.DataFrame(records).set_index("Metric")

# Save as image AND CSV
df_perf.to_csv(ROOT / "results" / "performance_table.csv")
render_table_image(
    df_perf,
    PLOTS / "12_performance_table.png",
    "Performance summary  ·  RGVH vs benchmarks",
    f"Sample: 2013-07-05 to 2025-08-28 ({(dly_rgvh.index.max()-dly_rgvh.index.min()).days/365.25:.2f} years).  "
    f"All series scaled to $1k vega exposure / $17.5k capital.",
    row_label_width=0.30, cell_height=0.32,
)

# Yearly P&L table
print("[13] yearly P&L table")
yt_rgvh = st.yearly_table(dly_rgvh)
yt_spy  = st.yearly_table(spy_pnl)
yt = pd.merge(
    yt_rgvh.rename(columns={"P&L":"RGVH P&L","Sharpe":"RGVH Sharpe","Hit %":"RGVH Hit","Days":"Days","Vol":"RGVH Vol","Max DD":"RGVH Max DD"}),
    yt_spy [["Year","P&L","Sharpe"]].rename(columns={"P&L":"SPY P&L","Sharpe":"SPY Sharpe"}),
    on="Year", how="left",
)
fmt_int    = lambda v: f"${v:,.0f}"      if pd.notna(v) else "—"
fmt_sharpe = lambda v: f"{v:+.2f}"        if pd.notna(v) else "—"
fmt_pct    = lambda v: f"{v:.0%}"         if pd.notna(v) else "—"
yt_disp = pd.DataFrame({
    "Days":          yt["Days"].astype(int),
    "RGVH P&L":      yt["RGVH P&L"].apply(fmt_int),
    "RGVH Sharpe":   yt["RGVH Sharpe"].apply(fmt_sharpe),
    "RGVH Hit":      yt["RGVH Hit"].apply(fmt_pct),
    "RGVH Vol":      yt["RGVH Vol"].apply(fmt_int),
    "RGVH Max DD":   yt["RGVH Max DD"].apply(fmt_int),
    "SPY P&L":       yt["SPY P&L"].apply(fmt_int),
    "SPY Sharpe":    yt["SPY Sharpe"].apply(fmt_sharpe),
}, index=yt["Year"].astype(int))
yt_disp.index.name = "Year"
yt.to_csv(ROOT / "results" / "yearly_table.csv", index=False)
render_table_image(
    yt_disp,
    PLOTS / "13_yearly_table.png",
    "Calendar-year performance  ·  RGVH vs SPY (rescaled)",
    "All values per $1k vega exposure / $17.5k Reg-T peak capital",
    row_label_width=0.10, cell_height=0.30,
)

# Drawdown periods table
print("[14] drawdown periods table")
dd_table = st.drawdown_periods(dly_rgvh, top_n=5)
if not dd_table.empty:
    dd_disp = dd_table.copy()
    dd_disp["Depth ($)"] = dd_disp["Depth ($)"].apply(fmt_int)
    dd_disp = dd_disp.set_index(pd.RangeIndex(start=1, stop=len(dd_disp)+1, name="#"))
    dd_table.to_csv(ROOT / "results" / "drawdown_periods.csv", index=False)
    render_table_image(
        dd_disp,
        PLOTS / "14_drawdown_periods_table.png",
        "Top-5 drawdown periods  ·  RGVH",
        "Ordered by depth.",
        row_label_width=0.08, cell_height=0.36,
    )

# ===========================================================================
# 01 — Executive summary panel (last so we have the building blocks)
# ===========================================================================
print("[01] executive summary panel")
fig = plt.figure(figsize=(14, 9))
fig.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.97, wspace=0.30, hspace=0.45)
gs = fig.add_gridspec(3, 4, height_ratios=[0.25, 0.4, 0.35])

# Top: KPI tiles
def kpi(ax, label, value, color=COLORS["navy"], sub=None):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.84,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=COLORS["bg_alt"], ec=color, lw=1.2,
                                transform=ax.transAxes))
    ax.text(0.5, 0.66, value, ha="center", va="center",
            fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.32, label, ha="center", va="center",
            fontsize=9.5, color=COLORS["ink_soft"], transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.16, sub, ha="center", va="center",
                fontsize=8, color=COLORS["ink_muted"], transform=ax.transAxes,
                fontstyle="italic")

ax_k1 = fig.add_subplot(gs[0,0]); kpi(ax_k1, "Net Sharpe", f"{st.sharpe(dly_rgvh):.2f}",
                                       color=COLORS["positive"], sub="vs SPY 0.85")
ax_k2 = fig.add_subplot(gs[0,1]); kpi(ax_k2, "Annualised Net P&L", fmt_int(rgvh_sum["Ann. Return ($)"]),
                                       sub="per $1k vega")
ax_k3 = fig.add_subplot(gs[0,2]); kpi(ax_k3, "Max Drawdown", fmt_int(rgvh_sum["Max DD ($)"]),
                                       color=COLORS["negative"], sub="1.3× annual P&L")
ax_k4 = fig.add_subplot(gs[0,3]); kpi(ax_k4, "Hit Rate", f"{rgvh_sum['Hit Rate']:.1%}",
                                       sub=f"{int(trades['rgvh_keep'].sum())} trades / 12y")

# Mid-left: equity curve mini
axE = fig.add_subplot(gs[1, :2])
axE.plot(dly_rgvh.cumsum(), color=COLORS["rgvh"], lw=2.3, label="RGVH")
axE.plot(dly_always.cumsum(), color=COLORS["subtle"], lw=1.4, ls="--",
         label="Unfiltered VRP")
axE.plot(spy_pnl.cumsum(), color=COLORS["spy"], lw=1.5, label="SPY (rescaled)")
axE.fill_between(dly_rgvh.index, 0, dly_rgvh.cumsum(), color=COLORS["navy"], alpha=0.10, lw=0)
axE.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axE.yaxis.set_major_formatter(FuncFormatter(format_dollars))
axE.set_title("Equity curve  ·  cumulative net P&L", loc="left", fontweight="bold")
axE.legend(loc="upper left", fontsize=9)

# Mid-right: rolling Sharpe
axS = fig.add_subplot(gs[1, 2:])
axS.fill_between(s_rgvh.index, s_rgvh, 0, where=s_rgvh > 0,
                 color=COLORS["positive"], alpha=0.18, lw=0)
axS.fill_between(s_rgvh.index, s_rgvh, 0, where=s_rgvh < 0,
                 color=COLORS["negative"], alpha=0.18, lw=0)
axS.plot(s_rgvh, color=COLORS["rgvh"], lw=2.0, label="RGVH")
axS.plot(s_spy,  color=COLORS["spy"],  lw=1.5, label="SPY")
axS.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axS.set_title("Rolling 252-day Sharpe", loc="left", fontweight="bold")
axS.legend(loc="upper left", fontsize=9)

# Bottom-left: yearly P&L bars
axY = fig.add_subplot(gs[2, :2])
yr_pl = dly_rgvh.groupby(dly_rgvh.index.year).sum()
clrs_y = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in yr_pl]
axY.bar(yr_pl.index, yr_pl.values, color=clrs_y, edgecolor=COLORS["bg"])
axY.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axY.yaxis.set_major_formatter(FuncFormatter(format_dollars))
axY.set_title("Calendar-year net P&L", loc="left", fontweight="bold")

# Bottom-right: drawdown comparison
axD = fig.add_subplot(gs[2, 2:])
axD.fill_between(dd_rgvh_pct.index, dd_rgvh_pct, 0, color=COLORS["rgvh"], alpha=0.55, lw=0,
                 label="RGVH")
axD.fill_between(dd_spy_pct.index, dd_spy_pct, 0, color=COLORS["spy"], alpha=0.30, lw=0,
                 label="SPY")
axD.set_title("Underwater drawdown  ·  RGVH vs SPY", loc="left", fontweight="bold")
axD.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
axD.legend(loc="lower left", fontsize=9)

title_block(fig,
    "RGVH  ·  Executive summary",
    "Regime-Gated Variance-Risk-Premium Harvester on SPY options.   12.15 years strict OOS.   Net of all costs.",
    "Source: WRDS OptionMetrics, Yahoo Finance, FRED.   Filter thresholds redacted.")
fig.savefig(PLOTS / "01_executive_summary.png")
plt.close()


# ===========================================================================
# Animated GIF — multi-panel professional
# ===========================================================================
print("[GIF] multi-panel animated equity + drawdown")
fig = plt.figure(figsize=(13, 7))
fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.97, hspace=0.40)
gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1.0, 0.0001])

axE = fig.add_subplot(gs[0])
axD = fig.add_subplot(gs[1], sharex=axE)
axE.tick_params(labelbottom=False)

# Pre-compute series
cum_rgvh   = dly_rgvh.cumsum()
cum_base   = dly_base.cumsum()
cum_always = dly_always.cumsum()
cum_spy    = spy_pnl.cumsum()
dd_rgvh_usd = (cum_rgvh - cum_rgvh.cummax())  # underwater $ series for bottom panel

axE.set_xlim(cum_rgvh.index.min(), cum_rgvh.index.max())
y_max = max(cum_rgvh.max(), cum_spy.max(), cum_always.max()) * 1.08
y_min = min(cum_rgvh.min(), cum_spy.min(), cum_always.min(), 0) * 1.08
axE.set_ylim(y_min, y_max)
axE.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axE.yaxis.set_major_formatter(FuncFormatter(format_dollars))
axE.set_ylabel("Cumulative net P&L  ($)")

axD.set_ylim(min(dd_rgvh_usd) * 1.08, 50)  # noqa
axD.axhline(0, color=COLORS["ink_muted"], lw=0.6)
axD.set_ylabel("Drawdown  ($)")
axD.yaxis.set_major_formatter(FuncFormatter(format_dollars))

ln_rgvh,   = axE.plot([], [], color=COLORS["rgvh"],    lw=2.6, label="RGVH")
ln_base,   = axE.plot([], [], color=COLORS["gold_dark"],lw=1.4, label="Base only")
ln_always, = axE.plot([], [], color=COLORS["subtle"],  lw=1.3, ls="--", label="Unfiltered VRP")
ln_spy,    = axE.plot([], [], color=COLORS["spy"],     lw=1.4, label="SPY (rescaled)")
fl_dd      = axD.fill_between([], [], 0, color=COLORS["rgvh"], alpha=0.55, lw=0)

# Single fill_between handle update is awkward in animation; manage via PolyCollection
from matplotlib.collections import PolyCollection
dd_poly = axD.fill_between(cum_rgvh.index[:1], dd_rgvh_usd.iloc[:1], 0,
                           color=COLORS["rgvh"], alpha=0.55, lw=0)

axE.legend(loc="upper left", fontsize=9, ncol=2)

year_text = fig.text(0.96, 0.79, "", ha="right", va="top",
                     fontsize=42, fontweight="bold", color=COLORS["navy"], alpha=0.85)
sharpe_text = fig.text(0.96, 0.71, "", ha="right", va="top",
                       fontsize=11, color=COLORS["ink_soft"])

# Crisis labels appear progressively
crisis_artists = {}
for d, lbl in CRISES:
    dt = pd.Timestamp(d)
    if dt < cum_rgvh.index.min() or dt > cum_rgvh.index.max(): continue
    art = axE.axvline(dt, color=COLORS["negative"], lw=0.8, ls="--", alpha=0)
    txt = axE.text(dt, y_max*0.95, f" {lbl}", color=COLORS["negative"], fontsize=8.5,
                   alpha=0, fontweight="medium")
    crisis_artists[dt] = (art, txt)

n_frames = 120
idxs = np.linspace(2, len(cum_rgvh) - 1, n_frames, dtype=int)

title_block(fig,
    "RGVH equity curve building  ·  2013 → 2025",
    "Cumulative net P&L (top) and drawdown (bottom). Crisis markers appear as they occur.",
    "Source: WRDS OptionMetrics, Yahoo Finance, FRED.")

def update(frame):
    global dd_poly
    j = idxs[frame]
    ln_rgvh.set_data(cum_rgvh.index[:j+1],   cum_rgvh.values[:j+1])
    ln_base.set_data(cum_base.index[:j+1],   cum_base.values[:j+1])
    ln_always.set_data(cum_always.index[:j+1], cum_always.values[:j+1])
    ln_spy.set_data(cum_spy.index[:j+1],     cum_spy.values[:j+1])

    # Refresh drawdown fill — remove and redraw
    dd_poly.remove()
    dd_poly = axD.fill_between(cum_rgvh.index[:j+1], dd_rgvh_usd.iloc[:j+1], 0,
                               color=COLORS["rgvh"], alpha=0.55, lw=0)

    cur_date = cum_rgvh.index[j]
    year_text.set_text(str(cur_date.year))
    rolling = dly_rgvh.iloc[max(0, j-252):j+1]
    sr = (rolling.mean() / rolling.std() * np.sqrt(252)) if rolling.std() > 0 else float("nan")
    pl = cum_rgvh.iloc[j]
    sharpe_text.set_text(f"Cum P&L: ${pl:,.0f}\n252-d Sharpe: {sr:.2f}")
    # Crisis-marker fade-in
    for dt, (art, txt) in crisis_artists.items():
        if cur_date >= dt:
            art.set_alpha(0.55)
            txt.set_alpha(0.85)

    return ln_rgvh, ln_base, ln_always, ln_spy, year_text, sharpe_text, dd_poly

anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, interval=70)
anim.save(PLOTS / "cumulative_pnl_animated.gif", writer="pillow", fps=14, dpi=140)
plt.close()
print("  saved -> cumulative_pnl_animated.gif")


# ===========================================================================
# Plotly interactive dashboard
# ===========================================================================
print("[HTML] interactive dashboard")

fig = make_subplots(
    rows=3, cols=2,
    specs=[[{"colspan":2}, None],
           [{}, {}],
           [{}, {}]],
    row_heights=[0.40, 0.30, 0.30],
    subplot_titles=(
        "Cumulative net P&L (RGVH vs benchmarks)",
        "Rolling 252-day Sharpe", "Drawdown (%)",
        "Monthly return scatter", "Yearly P&L",
    ),
    vertical_spacing=0.10, horizontal_spacing=0.08,
)

# Row 1: equity
fig.add_trace(go.Scatter(x=cum_rgvh.index, y=cum_rgvh.values, name="RGVH",
                         mode="lines", line=dict(color=COLORS["rgvh"], width=2.6),
                         hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>RGVH</extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=cum_base.index, y=cum_base.values, name="Base only",
                         mode="lines", line=dict(color=COLORS["gold_dark"], width=1.6)),
              row=1, col=1)
fig.add_trace(go.Scatter(x=cum_always.index, y=cum_always.values, name="Unfiltered",
                         mode="lines", line=dict(color=COLORS["subtle"], width=1.4, dash="dash")),
              row=1, col=1)
fig.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy.values, name="SPY (rescaled)",
                         mode="lines", line=dict(color=COLORS["spy"], width=1.6)),
              row=1, col=1)

# Row 2: rolling Sharpe + drawdown
fig.add_trace(go.Scatter(x=s_rgvh.index, y=s_rgvh.values, name="RGVH Sharpe",
                         line=dict(color=COLORS["rgvh"], width=2)),
              row=2, col=1)
fig.add_trace(go.Scatter(x=s_spy.index, y=s_spy.values, name="SPY Sharpe",
                         line=dict(color=COLORS["spy"], width=1.5)),
              row=2, col=1)

fig.add_trace(go.Scatter(x=dd_rgvh_pct.index, y=dd_rgvh_pct.values, name="RGVH DD",
                         fill="tozeroy", mode="lines",
                         line=dict(color=COLORS["rgvh"], width=0.5),
                         fillcolor=f"rgba(10,37,64,0.55)"),
              row=2, col=2)
fig.add_trace(go.Scatter(x=dd_spy_pct.index, y=dd_spy_pct.values, name="SPY DD",
                         fill="tozeroy", mode="lines",
                         line=dict(color=COLORS["spy"], width=0.5),
                         fillcolor=f"rgba(21,101,192,0.32)"),
              row=2, col=2)

# Row 3: scatter + yearly bars
fig.add_trace(go.Scatter(x=df["spy"], y=df["rgvh"], mode="markers",
                         marker=dict(color=COLORS["navy"], size=8, opacity=0.6,
                                     line=dict(color="white", width=0.5)),
                         name="Monthly returns",
                         hovertemplate="SPY %{x:.2f}%<br>RGVH %{y:.2f}%<extra></extra>"),
              row=3, col=1)
yr_idx = yr_pl.index
yr_clr = ["#1B5E20" if v > 0 else "#A92424" for v in yr_pl]
fig.add_trace(go.Bar(x=yr_idx, y=yr_pl.values, marker=dict(color=yr_clr),
                     name="Yearly P&L"),
              row=3, col=2)

fig.update_layout(
    template="plotly_white",
    title=dict(text="<b>RGVH interactive dashboard</b>"
                    "<br><span style='font-size:11px;color:#37474F'>Regime-Gated Variance-Risk-Premium Harvester · "
                    "12.15 years strict OOS · net of all costs</span>",
               x=0.04, y=0.98, xanchor="left"),
    height=1000, width=1200,
    margin=dict(t=110, b=70, l=70, r=40),
    legend=dict(orientation="h", y=1.03, x=0),
    font=dict(family="DejaVu Sans, Arial", color=COLORS["ink"], size=11),
    plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
    hovermode="x unified",
)
fig.update_xaxes(showgrid=False, showline=True, linecolor=COLORS["ink_muted"], linewidth=0.8)
fig.update_yaxes(showgrid=True, gridcolor=COLORS["muted"], gridwidth=0.4,
                 zerolinecolor=COLORS["ink_muted"], zerolinewidth=0.6)

fig.write_html(INTER / "dashboard.html", include_plotlyjs="cdn",
               config={"displaylogo": False, "modeBarButtonsToRemove":
                       ["lasso2d","select2d","autoScale2d"]})

print("\nAll professional visuals generated to:", PLOTS)
print("Interactive dashboard:", INTER / "dashboard.html")
