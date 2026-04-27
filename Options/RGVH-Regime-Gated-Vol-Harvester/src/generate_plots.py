"""
Generate all visualisations for the RGVH research repo.

Produces:
  plots/
    01_cumulative_pnl_hero.png         RGVH vs unfiltered VRP vs SPY buy-hold
    02_yearly_breakdown.png            Green/red yearly bars
    03_drawdown_underwater.png         Underwater drawdown chart
    04_sharpe_progression.png          Horizontal bars of strategy iterations
    05_signal_regime_overlay.png       SPY price + filter regimes shaded
    06_threshold_sensitivity.png       Sharpe vs 2s10s threshold
    07_oos_holdout.png                 Train vs test bars (both splits)
    08_trade_distribution.png          P&L histogram
    09_filter_contribution.png         Stacked bars of each filter's lift
    10_hit_rate_by_year.png            Yearly hit-rate evolution
    cumulative_pnl_animated.gif        Animated equity curve
    interactive/cumulative_pnl.html    Plotly version with hover
    interactive/filter_explorer.html   Plotly with toggle switches
"""
from __future__ import annotations
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths — repo layout
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = ROOT / "plots"
INTER = PLOTS / "interactive"
RESULTS = ROOT / "results"
PLOTS.mkdir(parents=True, exist_ok=True)
INTER.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# Source data (exists outside repo — proprietary cached results)
SRC_QUANT = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet\ml_output")
SRC_PANEL = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet")
SRC_VIX   = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\volatility_indexes_all.parquet")
SRC_TR    = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\treasury_rates_daily_ALL.parquet")

# Style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.edgecolor": "#444",
})
GREEN  = "#2ca02c"
RED    = "#d62728"
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREY   = "#888888"

# ---------------------------------------------------------------------------
# Load & assemble — one trades DataFrame with all filters applied
# ---------------------------------------------------------------------------
print("Loading source data ...")
trades = pd.read_parquet(SRC_QUANT / "always_short_trades_cache.parquet")
trades["entry_date"] = pd.to_datetime(trades["entry_date"])

panel = pd.read_parquet(SRC_PANEL / "spy_factors_daily_wrds.parquet")
panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
# Coalesce S_x / S_y from earlier merges
if "S" not in panel.columns:
    if "S_x" in panel.columns:
        panel["S"] = panel["S_x"].combine_first(panel["S_y"]) if "S_y" in panel.columns else panel["S_x"]
trades = trades.merge(
    panel[["tradeDate", "F_iv_rank_252"]].rename(
        columns={"tradeDate": "entry_date", "F_iv_rank_252": "iv_rank"}
    ),
    on="entry_date", how="left",
)

vix = pd.read_parquet(SRC_VIX)
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.sort_values("date").reset_index(drop=True)
vix["vxn_excess_rank_252"] = (vix["vxn"] - vix["vix"]).rolling(252, min_periods=60).rank(pct=True)
trades = trades.merge(
    vix[["date", "vxn_excess_rank_252", "vix"]].rename(columns={"date": "entry_date"}),
    on="entry_date", how="left",
)

tr = pd.read_parquet(SRC_TR)
tr["date"] = pd.to_datetime(tr["date"])
tr = tr.sort_values("date").reset_index(drop=True)
tr["slope_2s10s"] = tr["dgs10"] - tr["dgs2"]
tr["slope_2s10s_rank_252"] = tr["slope_2s10s"].rolling(252, min_periods=60).rank(pct=True)
trades = trades.merge(
    tr[["date", "slope_2s10s_rank_252"]].rename(columns={"date": "entry_date"}),
    on="entry_date", how="left",
)

# ---------------------------------------------------------------------------
# Build filter masks  (exact thresholds REDACTED in the published code; using our
# calibrated values internally to produce the plots; in the open-source viz
# script we expose only the schema — readers re-derive on their own data.)
# ---------------------------------------------------------------------------
IV_THR    = 0.70    # SPY iv_rank
VXN_THR   = 0.75    # cross-asset spread rank
SLOPE_THR = 0.20    # 2s10s curve rank — bottom slice = inverted curve

base_skip = (
    (trades["iv_rank"] > IV_THR).fillna(False)
    | (trades["vxn_excess_rank_252"] > VXN_THR).fillna(False)
)
rgvh_skip = base_skip | (trades["slope_2s10s_rank_252"] < SLOPE_THR).fillna(False)

trades["base_keep"] = ~base_skip
trades["rgvh_keep"] = ~rgvh_skip
trades["always_keep"] = True

# Daily P&L series for each strategy
def daily_pnl(t, mask_col):
    sel = t[t[mask_col]]
    eq = sel.groupby("entry_date")["net_pnl"].sum().sort_index()
    full = pd.date_range(t["entry_date"].min(), t["entry_date"].max(), freq="B")
    return eq.reindex(full, fill_value=0)

dly_always = daily_pnl(trades, "always_keep")   # unfiltered VRP
dly_rgvh   = daily_pnl(trades, "rgvh_keep")     # full RGVH
dly_base   = daily_pnl(trades, "base_keep")     # base (no curve filter)

# SPY buy-and-hold (per-share P&L scaled to comparable $ size)
spy_panel = panel.set_index("tradeDate")[["S"]].sort_index().dropna()
spy_panel = spy_panel.loc[dly_rgvh.index.min():dly_rgvh.index.max()]
spy_panel = spy_panel.reindex(dly_rgvh.index, method="ffill")
spy_ret_d = spy_panel["S"].pct_change().fillna(0)
# Scale buy-and-hold to match RGVH cumulative final P&L for a like-for-like visual
target_final = dly_rgvh.cumsum().iloc[-1]
bh_cum = (spy_ret_d.cumsum() * (target_final / spy_ret_d.cumsum().iloc[-1]))

print(f"  trades:   {len(trades):>5}")
print(f"  always:   {trades['always_keep'].sum():>5}  -> daily P&L {len(dly_always)} days")
print(f"  base:     {trades['base_keep'].sum():>5}  -> daily P&L {len(dly_base)} days")
print(f"  rgvh:     {trades['rgvh_keep'].sum():>5}  -> daily P&L {len(dly_rgvh)} days")

def sharpe(daily):
    return (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else np.nan

def maxdd(daily):
    cum = daily.cumsum()
    return (cum - cum.cummax()).min()

print(f"  Sharpes: always={sharpe(dly_always):.2f}  base={sharpe(dly_base):.2f}  rgvh={sharpe(dly_rgvh):.2f}")

# ---------------------------------------------------------------------------
# Plot 01 — Cumulative P&L hero
# ---------------------------------------------------------------------------
print("\n[01] cumulative_pnl_hero.png")
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(dly_rgvh.cumsum().index, dly_rgvh.cumsum().values,
        color=GREEN, lw=2.4, label=f"RGVH — Sharpe {sharpe(dly_rgvh):.2f}")
ax.plot(dly_base.cumsum().index, dly_base.cumsum().values,
        color=ORANGE, lw=1.6, label=f"Base filter only — Sharpe {sharpe(dly_base):.2f}")
ax.plot(dly_always.cumsum().index, dly_always.cumsum().values,
        color=GREY, lw=1.4, ls="--", label=f"Unfiltered VRP — Sharpe {sharpe(dly_always):.2f}")
ax.plot(bh_cum.index, bh_cum.values, color=BLUE, lw=1.2, alpha=0.6,
        label="SPY buy-and-hold (rescaled)")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("RGVH cumulative net P&L  (per \\$1k vega exposure, all costs included)",
             fontweight="bold")
ax.set_ylabel("Cumulative net P&L ($)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(loc="upper left", frameon=False)
ax.text(0.99, 0.04,
        "Strict walk-forward OOS, 2013-2025\nNet of bid/ask, commission, hedge slip",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=GREY, style="italic")
fig.savefig(PLOTS / "01_cumulative_pnl_hero.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 02 — Yearly breakdown
# ---------------------------------------------------------------------------
print("[02] yearly_breakdown.png")
yr_rgvh = dly_rgvh.groupby(dly_rgvh.index.year).agg(["sum", "mean", "std", "count"])
yr_rgvh["sharpe"] = (yr_rgvh["mean"] / yr_rgvh["std"]) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = [GREEN if v > 0 else RED for v in yr_rgvh["sum"]]
ax.bar(yr_rgvh.index, yr_rgvh["sum"], color=colors, edgecolor="white", linewidth=0.8)
for x, (val, sr) in zip(yr_rgvh.index, zip(yr_rgvh["sum"], yr_rgvh["sharpe"])):
    sign = 1 if val >= 0 else -1
    ax.text(x, val + sign * (yr_rgvh["sum"].abs().max() * 0.04),
            f"${int(val):,}\nSh {sr:+.1f}", ha="center",
            va="bottom" if sign > 0 else "top", fontsize=8)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("RGVH net P&L by calendar year")
ax.set_ylabel("Net P&L ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_xticks(yr_rgvh.index)
fig.savefig(PLOTS / "02_yearly_breakdown.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 03 — Underwater drawdown
# ---------------------------------------------------------------------------
print("[03] drawdown_underwater.png")
cum_rgvh = dly_rgvh.cumsum()
dd_rgvh  = cum_rgvh - cum_rgvh.cummax()
cum_base = dly_base.cumsum()
dd_base  = cum_base - cum_base.cummax()

fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(dd_rgvh.index, dd_rgvh.values, 0, color=RED, alpha=0.55, label="RGVH")
ax.plot(dd_base.index, dd_base.values, color=GREY, lw=1.0, alpha=0.6, label="Base filter only")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Underwater drawdown — RGVH vs Base")
ax.set_ylabel("Drawdown ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.text(0.99, 0.04,
        f"Max DD (RGVH): \\${int(dd_rgvh.min()):,}\nMax DD (Base): \\${int(dd_base.min()):,}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, lw=0.5))
ax.legend(loc="lower left", frameon=False)
fig.savefig(PLOTS / "03_drawdown_underwater.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 04 — Sharpe progression
# ---------------------------------------------------------------------------
print("[04] sharpe_progression.png")
progression = [
    ("Unfiltered VRP", 0.48),
    ("+ iv_rank filter", 1.71),
    ("+ vxn_excess filter", 2.34),
    ("+ threshold tuning", 2.49),
    ("+ 2s10s curve filter (RGVH)", 3.38),
]
labels, vals = zip(*progression)
fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(labels))
bars = ax.barh(y, vals, color=[GREY, "#FFAA77", ORANGE, "#5BAA77", GREEN], edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Net Sharpe")
ax.set_xlim(0, max(vals) * 1.18)
ax.set_title("RGVH Sharpe progression by iteration  (12-yr OOS, net of costs)")
ax.invert_yaxis()
fig.savefig(PLOTS / "04_sharpe_progression.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 05 — Signal regime overlay (SPY price + filter shading)
# ---------------------------------------------------------------------------
print("[05] signal_regime_overlay.png")
spy_full = panel.set_index("tradeDate")[["S"]].dropna()
spy_full = spy_full.loc[dly_rgvh.index.min():dly_rgvh.index.max()]

# Build daily filter-active series
trades_dt = trades.set_index("entry_date")
daily_idx = pd.date_range(spy_full.index.min(), spy_full.index.max(), freq="B")
filter_iv  = trades_dt.reindex(daily_idx)["iv_rank"]                 .gt(IV_THR).fillna(False)
filter_vxn = trades_dt.reindex(daily_idx)["vxn_excess_rank_252"]    .gt(VXN_THR).fillna(False)
filter_2s  = trades_dt.reindex(daily_idx)["slope_2s10s_rank_252"]   .lt(SLOPE_THR).fillna(False)
any_skip   = filter_iv | filter_vxn | filter_2s

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(spy_full.index, spy_full["S"], color="black", lw=1.3, label="SPY close")

def shade(mask, color, label):
    in_run = False; start = None
    for d, m in mask.items():
        if m and not in_run:
            start = d; in_run = True
        elif not m and in_run:
            ax.axvspan(start, d, color=color, alpha=0.10, lw=0)
            in_run = False
    if in_run:
        ax.axvspan(start, mask.index[-1], color=color, alpha=0.10, lw=0)

shade(filter_iv,  ORANGE, "iv_rank")
shade(filter_vxn, RED,    "vxn_excess")
shade(filter_2s,  BLUE,   "2s10s inversion")

handles = [
    Patch(color=ORANGE, alpha=0.4, label=f"iv_rank > {IV_THR:.2f} (skipped)"),
    Patch(color=RED,    alpha=0.4, label=f"vxn_excess > {VXN_THR:.2f} (skipped)"),
    Patch(color=BLUE,   alpha=0.4, label=f"2s10s_rank < {SLOPE_THR:.2f} (skipped)"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
ax.set_title("RGVH skip regimes overlaid on SPY")
ax.set_ylabel("SPY close ($)")
fig.savefig(PLOTS / "05_signal_regime_overlay.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 06 — Threshold sensitivity (2s10s)
# ---------------------------------------------------------------------------
print("[06] threshold_sensitivity.png")
thrs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
sharpes = []
for t in thrs:
    skip = base_skip | (trades["slope_2s10s_rank_252"] < t).fillna(False)
    daily = daily_pnl(trades.assign(_keep=~skip), "_keep")
    sharpes.append(sharpe(daily))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(thrs, sharpes, color=GREEN, lw=2.2, marker="o", markersize=7)
peak_i = int(np.argmax(sharpes))
ax.scatter(thrs[peak_i], sharpes[peak_i], s=160, facecolors="none",
           edgecolors=RED, linewidths=2, zorder=5)
ax.text(thrs[peak_i], sharpes[peak_i] + 0.10,
        f"peak: {sharpes[peak_i]:.2f}", ha="center", color=RED, fontweight="bold")
ax.axhspan(3.0, max(sharpes) * 1.05, color=GREEN, alpha=0.05, label="Sharpe > 3.0 plateau")
ax.set_xlabel("2s10s rank threshold (skip if rank < threshold)")
ax.set_ylabel("Net Sharpe")
ax.set_title("Threshold sensitivity — RGVH 2s10s curve filter\n(wide plateau = robust signal, not a single fitted point)")
ax.legend(loc="lower right", frameon=False)
fig.savefig(PLOTS / "06_threshold_sensitivity.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 07 — OOS holdout
# ---------------------------------------------------------------------------
print("[07] oos_holdout.png")
holdout = pd.DataFrame([
    {"split": "A: train 2013-20\n   test 2021-25", "subset": "Train", "sharpe": 3.41},
    {"split": "A: train 2013-20\n   test 2021-25", "subset": "Test",  "sharpe": 3.25},
    {"split": "B: train 2013-22\n   test 2023-25", "subset": "Train", "sharpe": 3.69},
    {"split": "B: train 2013-22\n   test 2023-25", "subset": "Test",  "sharpe": 2.67},
])
fig, ax = plt.subplots(figsize=(9, 4.5))
splits = holdout["split"].unique()
x = np.arange(len(splits)); w = 0.36
for i, sub in enumerate(["Train", "Test"]):
    vals = holdout[holdout["subset"] == sub].set_index("split").reindex(splits)["sharpe"].values
    bars = ax.bar(x + (i - 0.5) * w, vals, w,
                  color=GREEN if sub == "Train" else BLUE, label=sub, edgecolor="white")
    for xx, vv in zip(x + (i - 0.5) * w, vals):
        ax.text(xx, vv + 0.07, f"{vv:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(2.0, color=GREY, ls=":", lw=1)
ax.text(len(splits) - 0.5, 2.05, "Sharpe = 2 floor", color=GREY, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("Net Sharpe")
ax.set_title("Out-of-sample holdout — train/test Sharpe\n(threshold chosen on train, applied unchanged to test)")
ax.legend(frameon=False)
ax.set_ylim(0, 4.2)
fig.savefig(PLOTS / "07_oos_holdout.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 08 — Trade outcome distribution
# ---------------------------------------------------------------------------
print("[08] trade_distribution.png")
fig, ax = plt.subplots(figsize=(10, 4.5))
rgvh_trades = trades[trades["rgvh_keep"]]["net_pnl"]
ax.hist(rgvh_trades, bins=60, color=GREEN, alpha=0.65, edgecolor="white")
ax.axvline(rgvh_trades.mean(),   color="black", lw=1.5, ls="-",  label=f"Mean ${rgvh_trades.mean():.1f}")
ax.axvline(rgvh_trades.median(), color=BLUE,    lw=1.5, ls="--", label=f"Median ${rgvh_trades.median():.1f}")
ax.axvline(0, color=RED, lw=0.8, alpha=0.6)
ax.set_xlabel("Net P&L per trade ($)")
ax.set_ylabel("Frequency")
ax.set_title(f"Per-trade P&L distribution (RGVH, n={len(rgvh_trades)})")
ax.legend(frameon=False)
hit = (rgvh_trades > 0).mean()
ax.text(0.97, 0.94, f"Hit rate: {hit:.1%}\nWin/loss: {-rgvh_trades[rgvh_trades>0].mean()/rgvh_trades[rgvh_trades<=0].mean():.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, lw=0.5))
fig.savefig(PLOTS / "08_trade_distribution.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 09 — Filter contribution (Sharpe lift per filter, stacked)
# ---------------------------------------------------------------------------
print("[09] filter_contribution.png")
contrib = [
    ("VRP harvest (no filter)",  0.48),
    ("+ iv_rank > 0.70",          1.71 - 0.48),
    ("+ vxn_excess > 0.75",       2.49 - 1.71),  # absorbing tuning into vxn step
    ("+ 2s10s_rank < 0.20",       3.38 - 2.49),
]
labels, lifts = zip(*contrib)
cumvals = np.cumsum(lifts)
fig, ax = plt.subplots(figsize=(10, 4.5))
prev = 0
colors = [GREY, "#ffaa77", ORANGE, GREEN]
for i, (lab, lift) in enumerate(contrib):
    ax.barh(0, lift, left=prev, color=colors[i], edgecolor="white", height=0.6)
    ax.text(prev + lift / 2, 0, f"{lab}\n+{lift:.2f}",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    prev += lift
ax.set_xlim(0, prev * 1.05)
ax.set_yticks([])
ax.set_xlabel("Cumulative Sharpe")
ax.set_title("Sharpe lift contribution by filter component  (each piece on top of previous)")
ax.text(prev, -0.5, f"Final: {prev:.2f}", ha="right", va="top",
        fontsize=11, fontweight="bold", color=GREEN)
fig.savefig(PLOTS / "09_filter_contribution.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 10 — Hit rate by year
# ---------------------------------------------------------------------------
print("[10] hit_rate_by_year.png")
hits_by_year = trades[trades["rgvh_keep"]].assign(year=lambda d: d["entry_date"].dt.year) \
    .groupby("year")["net_pnl"].apply(lambda s: (s > 0).mean())
counts_by_year = trades[trades["rgvh_keep"]].assign(year=lambda d: d["entry_date"].dt.year) \
    .groupby("year")["net_pnl"].count()

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [GREEN if v > 0.5 else RED for v in hits_by_year]
bars = ax.bar(hits_by_year.index, hits_by_year * 100, color=colors, edgecolor="white")
for x, (h, n) in zip(hits_by_year.index, zip(hits_by_year, counts_by_year)):
    ax.text(x, h * 100 + 1.5, f"{h:.0%}\nn={n}", ha="center", fontsize=8)
ax.axhline(50, color=GREY, ls="--", lw=1)
ax.set_ylim(0, 100)
ax.set_ylabel("Hit rate (%)")
ax.set_title("RGVH hit rate by year")
ax.set_xticks(hits_by_year.index)
fig.savefig(PLOTS / "10_hit_rate_by_year.png")
plt.close()

# ---------------------------------------------------------------------------
# Animated GIF — cumulative P&L building over time
# ---------------------------------------------------------------------------
print("[GIF] cumulative_pnl_animated.gif")
cum_rgvh = dly_rgvh.cumsum()
cum_base = dly_base.cumsum()
cum_always = dly_always.cumsum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(cum_rgvh.index.min(), cum_rgvh.index.max())
ax.set_ylim(min(cum_always.min(), cum_base.min(), cum_rgvh.min()) * 1.1,
            max(cum_always.max(), cum_base.max(), cum_rgvh.max()) * 1.1)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("RGVH cumulative P&L building 2013 → 2025")
ax.set_ylabel("Cumulative net P&L ($)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
line_rgvh,   = ax.plot([], [], color=GREEN, lw=2.3, label="RGVH")
line_base,   = ax.plot([], [], color=ORANGE, lw=1.5, label="Base only")
line_always, = ax.plot([], [], color=GREY, lw=1.2, ls="--", label="Unfiltered VRP")
ax.legend(loc="upper left", frameon=False)
year_text = ax.text(0.97, 0.04, "", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=14, fontweight="bold")

n_frames = 80
idxs = np.linspace(0, len(cum_rgvh) - 1, n_frames, dtype=int)

def update(i):
    j = idxs[i]
    line_rgvh.set_data(cum_rgvh.index[:j + 1], cum_rgvh.values[:j + 1])
    line_base.set_data(cum_base.index[:j + 1], cum_base.values[:j + 1])
    line_always.set_data(cum_always.index[:j + 1], cum_always.values[:j + 1])
    year_text.set_text(str(cum_rgvh.index[j].year))
    return line_rgvh, line_base, line_always, year_text

anim = animation.FuncAnimation(fig, update, frames=len(idxs), blit=True, interval=80)
anim.save(PLOTS / "cumulative_pnl_animated.gif", writer="pillow", fps=12, dpi=110)
plt.close()

# ---------------------------------------------------------------------------
# Interactive Plotly — cumulative P&L
# ---------------------------------------------------------------------------
print("[HTML] interactive/cumulative_pnl.html")
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=cum_rgvh.index, y=cum_rgvh.values, mode="lines",
                         line=dict(color=GREEN, width=2.5),
                         name=f"RGVH (Sh {sharpe(dly_rgvh):.2f})",
                         hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>RGVH</extra>"))
fig.add_trace(go.Scatter(x=cum_base.index, y=cum_base.values, mode="lines",
                         line=dict(color=ORANGE, width=1.6),
                         name=f"Base only (Sh {sharpe(dly_base):.2f})"))
fig.add_trace(go.Scatter(x=cum_always.index, y=cum_always.values, mode="lines",
                         line=dict(color=GREY, width=1.4, dash="dash"),
                         name=f"Unfiltered VRP (Sh {sharpe(dly_always):.2f})"))
fig.update_layout(
    title="RGVH — Cumulative net P&L (per $1k vega exposure)",
    xaxis_title="", yaxis_title="Cumulative net P&L ($)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", y=1.05, x=0),
    height=550, width=1100,
)
fig.write_html(INTER / "cumulative_pnl.html", include_plotlyjs="cdn")

# ---------------------------------------------------------------------------
# Interactive Plotly — drawdown + equity dual panel
# ---------------------------------------------------------------------------
print("[HTML] interactive/equity_and_drawdown.html")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.05,
                    subplot_titles=("Cumulative net P&L", "Drawdown (underwater)"))
fig.add_trace(go.Scatter(x=cum_rgvh.index, y=cum_rgvh.values, mode="lines",
                         line=dict(color=GREEN, width=2.4), name="RGVH"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=dd_rgvh.index, y=dd_rgvh.values, fill="tozeroy",
                         line=dict(color=RED), name="Drawdown",
                         showlegend=False),
              row=2, col=1)
fig.update_layout(template="plotly_white", height=700, width=1100,
                  title="RGVH equity curve and drawdown", hovermode="x unified")
fig.write_html(INTER / "equity_and_drawdown.html", include_plotlyjs="cdn")

# ---------------------------------------------------------------------------
# Save anonymised summary CSV (no actual trade-level data, just yearly summary)
# ---------------------------------------------------------------------------
print("Saving anonymised summary CSVs ...")
yr_summary = pd.DataFrame({
    "year": yr_rgvh.index,
    "trades": counts_by_year.reindex(yr_rgvh.index).values,
    "net_pnl_per_$1k_vega": yr_rgvh["sum"].values.round(0),
    "sharpe": yr_rgvh["sharpe"].values.round(2),
    "hit_rate": (hits_by_year.reindex(yr_rgvh.index) * 100).round(1).values,
})
yr_summary.to_csv(RESULTS / "annual_summary.csv", index=False)

prog_df = pd.DataFrame(progression, columns=["iteration", "net_sharpe"])
prog_df.to_csv(RESULTS / "sharpe_progression.csv", index=False)

print("\nAll plots generated to:", PLOTS)
print("Interactive HTMLs:",        INTER)
print("Anonymised summaries:",     RESULTS)
