"""
LinkedIn-friendly animated GIF: '$100,000 — three strategies, 12 years'.

Designed for autoplay in LinkedIn / X / Threads feeds. Single canvas, year
counter, three equity curves drawing in over ~10 seconds, end-value labels
landing dramatically at the close.

Output:
    plots/pro/100k_three_strategies.gif    (~1200x700, 15fps, ~10s, ~5MB)
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
import matplotlib.animation as animation
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

from .theme import COLORS, apply_theme

apply_theme(font_size=11, dpi=110)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT  = ROOT / "plots" / "pro" / "100k_three_strategies.gif"
OUT.parent.mkdir(parents=True, exist_ok=True)

SRC_QUANT = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet\ml_output")
SRC_PANEL = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet")
SRC_VIX   = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\volatility_indexes_all.parquet")
SRC_TR    = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\treasury_rates_daily_ALL.parquet")

IV_THR, VXN_THR, SLOPE_THR = 0.70, 0.75, 0.20
RGVH_PEAK_REGT = 17_488.0
RGVH_PEAK_PM   =  5_246.0
START_CAPITAL  = 100_000.0


def tlog(m): print(f"[gif] {m}", flush=True)


# ---------------------------------------------------------------------------
# Data assembly (same as interactive_comparison.py)
# ---------------------------------------------------------------------------
tlog("loading data ...")
trades = pd.read_parquet(SRC_QUANT / "always_short_trades_cache.parquet")
trades["entry_date"] = pd.to_datetime(trades["entry_date"])

panel = pd.read_parquet(SRC_PANEL / "spy_factors_daily_wrds.parquet")
panel["tradeDate"] = pd.to_datetime(panel["tradeDate"])
trades = trades.merge(
    panel[["tradeDate","F_iv_rank_252"]].rename(
        columns={"tradeDate":"entry_date","F_iv_rank_252":"iv_rank"}),
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

skip = ((trades["iv_rank"] > IV_THR).fillna(False)
        | (trades["vxn_excess_rank_252"] > VXN_THR).fillna(False)
        | (trades["slope_2s10s_rank_252"] < SLOPE_THR).fillna(False))
rgvh_trades = trades[~skip]

eq = rgvh_trades.groupby("entry_date")["net_pnl"].sum().sort_index()
full_idx = pd.date_range(rgvh_trades["entry_date"].min(),
                         rgvh_trades["entry_date"].max(), freq="B")
dly_rgvh = eq.reindex(full_idx, fill_value=0.0)

tlog("fetching SPY total-return ...")
spy = yf.Ticker("SPY").history(
    start=str(dly_rgvh.index.min().date() - pd.Timedelta(days=10)),
    end=str(dly_rgvh.index.max().date() + pd.Timedelta(days=2)),
    auto_adjust=True,
)["Close"].dropna()
spy.index = spy.index.tz_localize(None)
spy = spy.reindex(dly_rgvh.index, method="ffill")
spy_ret = spy.pct_change().fillna(0)

# Compounded equity curves on $100k start
def compound(rets, start):
    return start * (1 + rets).cumprod()

eq_spy   = compound(spy_ret,                       START_CAPITAL)
eq_regt  = compound(dly_rgvh / RGVH_PEAK_REGT,     START_CAPITAL)
eq_pm    = compound(dly_rgvh / RGVH_PEAK_PM,       START_CAPITAL)

end_spy   = eq_spy.iloc[-1]
end_regt  = eq_regt.iloc[-1]
end_pm    = eq_pm.iloc[-1]

# CAGR for the labels
yrs = len(eq_spy) / 252
cagr = lambda end: (end / START_CAPITAL) ** (1/yrs) - 1
sharpe = lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else float("nan")

stats_text = {
    "spy":  f"S&P 500\n${end_spy:,.0f}\nCAGR {cagr(end_spy):.1%}\nSharpe {sharpe(spy_ret):.2f}",
    "regt": f"RGVH Reg-T\n${end_regt:,.0f}\nCAGR {cagr(end_regt):.1%}\nSharpe {sharpe(dly_rgvh/RGVH_PEAK_REGT):.2f}",
    "pm":   f"RGVH PM\n${end_pm:,.0f}\nCAGR {cagr(end_pm):.1%}\nSharpe {sharpe(dly_rgvh/RGVH_PEAK_PM):.2f}",
}

print(f"  end SPY  = ${end_spy:,.0f}   CAGR {cagr(end_spy):.2%}")
print(f"  end RegT = ${end_regt:,.0f}  CAGR {cagr(end_regt):.2%}")
print(f"  end PM   = ${end_pm:,.0f}    CAGR {cagr(end_pm):.2%}")


# ---------------------------------------------------------------------------
# Figure setup — sized for LinkedIn feeds (16:9-ish)
# ---------------------------------------------------------------------------
W_INCH, H_INCH = 13.0, 7.0   # ~1430x770 at dpi=110 — extra width for end-tags
fig = plt.figure(figsize=(W_INCH, H_INCH), facecolor=COLORS["bg"])
fig.subplots_adjust(top=0.78, bottom=0.10, left=0.06, right=0.82)  # leave 18% on the right
ax = fig.add_subplot(111)

# Y-limits — generous headroom for the line tags
y_top = max(end_spy, end_regt, end_pm) * 1.18
ax.set_ylim(0, y_top)
ax.set_xlim(eq_spy.index.min(), eq_spy.index.max() + pd.Timedelta(days=120))
ax.set_facecolor(COLORS["bg"])
ax.grid(True, axis="y", color=COLORS["muted"], alpha=0.55, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(COLORS["ink_muted"])
ax.spines["bottom"].set_linewidth(0.7)
ax.tick_params(axis="x", colors=COLORS["ink_soft"], labelsize=10)
ax.tick_params(axis="y", colors=COLORS["ink_soft"], labelsize=10, length=0)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:.0f}k" if v < 1e6 else f"${v/1e6:.1f}M"))

# Reference line at the starting capital for visual anchor
ax.axhline(START_CAPITAL, color=COLORS["ink_muted"], lw=0.8, ls=":", alpha=0.7, zorder=1)
ax.text(eq_spy.index.min(), START_CAPITAL * 1.06, f" starting capital  ${START_CAPITAL:,.0f}",
        fontsize=9, color=COLORS["ink_muted"], style="italic", va="bottom")

# Three line objects (start empty) — order matters for layering
line_spy,  = ax.plot([], [], color=COLORS["spy"],      lw=2.6, zorder=4,
                     solid_capstyle="round", label="S&P 500")
line_regt, = ax.plot([], [], color=COLORS["rgvh"],     lw=2.6, zorder=5,
                     solid_capstyle="round", label="RGVH (Reg-T)")
line_pm,   = ax.plot([], [], color=COLORS["gold_dark"], lw=3.4, zorder=6,
                     solid_capstyle="round", label="RGVH (Portfolio Margin)")

# Big year counter (top-right) and a subtle equity readout below it
year_text  = fig.text(0.97, 0.86, "", ha="right", va="bottom",
                      fontsize=42, fontweight="bold", color=COLORS["navy"])
read_text  = fig.text(0.97, 0.815, "", ha="right", va="bottom",
                      fontsize=10.5, color=COLORS["ink_soft"])

# Title block — left-aligned, two-tier
fig.text(0.07, 0.94, "What does $100,000 become?",
         ha="left", va="top", fontsize=22, fontweight="bold",
         color=COLORS["ink"])
fig.text(0.07, 0.885,
         "Three strategies on identical starting capital, 12 years compounded  "
         "(2013-2025, OOS, net of all costs)",
         ha="left", va="top", fontsize=11.5, color=COLORS["ink_soft"], style="italic")

# Static stats footer (always visible, scale-invariant)
fig.text(0.06, 0.045,
         f"Backtest period: 2013-07 to 2025-08  ·  {len(eq_spy):,} trading days  ·  filter thresholds redacted",
         ha="left", va="bottom", fontsize=8.5, color=COLORS["ink_muted"], style="italic")
fig.text(0.97, 0.045,
         "Source: WRDS OptionMetrics, Yahoo Finance, FRED  ·  github.com/Weculp/Trading-Strategies",
         ha="right", va="bottom", fontsize=8.5, color=COLORS["ink_muted"], style="italic")

# End-value tag artists (built once, made visible at the end)
def make_end_tag(color):
    return ax.annotate(
        "", xy=(eq_spy.index[-1], 0), xytext=(12, 0),
        textcoords="offset points",
        ha="left", va="center",
        fontsize=10.0, color=COLORS["ink"], fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.45", fc=COLORS["bg"], ec=color, lw=1.4),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0),
        annotation_clip=False,
    )

tag_spy  = make_end_tag(COLORS["spy"])
tag_regt = make_end_tag(COLORS["rgvh"])
tag_pm   = make_end_tag(COLORS["gold_dark"])
for t in (tag_spy, tag_regt, tag_pm):
    t.set_visible(False)

# Crisis flash markers (dotted vertical lines + faint label) — pop up briefly
CRISIS_EVENTS = [
    (pd.Timestamp("2018-02-05"), "Volmageddon"),
    (pd.Timestamp("2020-03-12"), "COVID crash"),
    (pd.Timestamp("2022-06-13"), "Fed hike cycle"),
]
crisis_lines = []
crisis_labels = []
for d, lbl in CRISIS_EVENTS:
    line = ax.axvline(d, color=COLORS["negative"], ls="--", lw=0.9, alpha=0)
    text = ax.text(d, y_top * 0.97, f" {lbl}", color=COLORS["negative"],
                   fontsize=9, alpha=0, fontweight="medium", va="top")
    crisis_lines.append(line)
    crisis_labels.append(text)

# Legend — minimalist
ax.legend(loc="upper left", frameon=False, fontsize=10.5,
          handlelength=1.6, borderpad=0.0)

ax.set_ylabel("Account value", color=COLORS["ink_soft"], fontsize=11)


# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------
FPS         = 18
LINE_FRAMES = 165   # seconds drawing the equity curves (~9.2s)
HOLD_FRAMES = 30    # final hold (~1.7s) before loop
N_FRAMES    = LINE_FRAMES + HOLD_FRAMES
n_pts       = len(eq_spy)


def ease_out(t):
    """Smooth ease-out for the line draw (1 - (1-t)^2)."""
    return 1 - (1 - t) ** 2


def update(frame):
    if frame < LINE_FRAMES:
        progress = ease_out(frame / max(LINE_FRAMES - 1, 1))
        j = int(progress * (n_pts - 1))
    else:
        j = n_pts - 1

    # Update lines
    line_spy .set_data(eq_spy.index[:j + 1],   eq_spy.values[:j + 1])
    line_regt.set_data(eq_regt.index[:j + 1],  eq_regt.values[:j + 1])
    line_pm  .set_data(eq_pm.index[:j + 1],    eq_pm.values[:j + 1])

    # Year counter + small readout under it
    cur_date = eq_spy.index[j]
    year_text.set_text(str(cur_date.year))
    cur_pm   = eq_pm.iloc[j]
    cur_spy  = eq_spy.iloc[j]
    cur_regt = eq_regt.iloc[j]
    read_text.set_text(
        f"PM:    ${cur_pm:>9,.0f}\n"
        f"SPY:   ${cur_spy:>9,.0f}\n"
        f"Reg-T: ${cur_regt:>9,.0f}"
    )

    # Crisis markers fade in as we cross them, fade out after 30 frames
    for (d, _), line, txt in zip(CRISIS_EVENTS, crisis_lines, crisis_labels):
        cur_alpha = 0.0
        if cur_date >= d:
            # Highlight for ~25 frames after the event passes
            frames_since = j - eq_spy.index.get_indexer([d], method="nearest")[0]
            if 0 <= frames_since <= 60:
                cur_alpha = 0.85 * (1 - frames_since / 60)
        line.set_alpha(cur_alpha)
        txt .set_alpha(cur_alpha)

    # End-value tags — appear in the last LINE_FRAMES segment + during hold
    if frame >= LINE_FRAMES - 8:
        for tag, eq_series, color, label in [
            (tag_pm,   eq_pm,   COLORS["gold_dark"], stats_text["pm"]),
            (tag_spy,  eq_spy,  COLORS["spy"],       stats_text["spy"]),
            (tag_regt, eq_regt, COLORS["rgvh"],      stats_text["regt"]),
        ]:
            tag.xy = (eq_series.index[-1], eq_series.iloc[-1])
            tag.set_text(label)
            tag.set_visible(True)
    else:
        for tag in (tag_pm, tag_spy, tag_regt):
            tag.set_visible(False)

    return (line_spy, line_regt, line_pm, year_text, read_text,
            tag_spy, tag_regt, tag_pm, *crisis_lines, *crisis_labels)


tlog(f"rendering {N_FRAMES} frames @ {FPS} fps  →  ~{N_FRAMES/FPS:.1f} seconds")
anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False, interval=1000/FPS)
anim.save(OUT, writer="pillow", fps=FPS, dpi=110)
plt.close()

size_mb = OUT.stat().st_size / 1024 / 1024
tlog(f"saved {OUT}  ({size_mb:.1f} MB)")
print(f"\nLinkedIn-ready GIF:")
print(f"  Path: {OUT}")
print(f"  Size: {size_mb:.1f} MB  (LinkedIn limit 100 MB; recommend <8 MB for fast feed loading)")
print(f"  Duration: {N_FRAMES/FPS:.1f}s @ {FPS}fps")
print(f"  Resolution: {int(W_INCH*110)}x{int(H_INCH*110)} px")
