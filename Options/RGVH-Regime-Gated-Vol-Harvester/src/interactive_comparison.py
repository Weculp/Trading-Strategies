"""
Apples-to-apples interactive comparison: RGVH vs SPY buy-and-hold on
identical starting capital, with returns COMPOUNDED on both sides.

Why compounding matters
-----------------------
SPY buy-and-hold naturally compounds: dividends reinvest, capital grows, the
next day's percentage move is applied to a larger base.

RGVH in the original backtest uses fixed vega-$ sizing — every trade sized to
$1,000 of vega regardless of account equity. That under-states the strategy
relative to SPY because it ignores the natural growth of the trading account.

For an honest comparison we treat RGVH's daily P&L as a percentage return on
the relevant capital base (Reg-T peak ≈ $17,488 or PM peak ≈ $5,246), then
compound that return stream just like SPY's. This is the way the strategy
would actually grow if the trader scaled their position size with their
account equity.

Outputs
-------
plots/pro/interactive/spy_vs_rgvh_compounded.html
    Multi-trace Plotly figure with toggle buttons for the four equity curves,
    a slider for starting capital, hover tooltips, and an inset stats panel.
"""
from __future__ import annotations
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import COLORS

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INTER = ROOT / "plots" / "pro" / "interactive"
INTER.mkdir(parents=True, exist_ok=True)

SRC_QUANT = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet\ml_output")
SRC_PANEL = Path(r"C:\Users\vikal\Claude_quant\options-data\parquet")
SRC_VIX   = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\volatility_indexes_all.parquet")
SRC_TR    = Path(r"C:\Users\vikal\Claude_quant\options-data\wrds\treasury_rates_daily_ALL.parquet")

# Calibration (redacted in published config)
IV_THR, VXN_THR, SLOPE_THR = 0.70, 0.75, 0.20
RGVH_PEAK_REGT = 17_488.0
RGVH_PEAK_PM   =  5_246.0


def tlog(m): print(f"[interactive] {m}", flush=True)


# ---------------------------------------------------------------------------
# Build daily P&L series
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

# SPY total return with auto-adjusted close = dividends reinvested
tlog("fetching SPY total-return ...")
spy = yf.Ticker("SPY").history(
    start=str(dly_rgvh.index.min().date() - pd.Timedelta(days=10)),
    end=str(dly_rgvh.index.max().date() + pd.Timedelta(days=2)),
    auto_adjust=True,
)["Close"].dropna()
spy.index = spy.index.tz_localize(None)
spy = spy.reindex(dly_rgvh.index, method="ffill")
spy_ret = spy.pct_change().fillna(0)

# ---------------------------------------------------------------------------
# Compounded equity curves on a normalised $100 base
# ---------------------------------------------------------------------------
def compound_curve(daily_returns: pd.Series, start: float = 100.0) -> pd.Series:
    """Standard geometric compounding."""
    return start * (1 + daily_returns).cumprod()

# Convert RGVH daily $ P&L to daily % return on each margin basis
rgvh_pct_regt = dly_rgvh / RGVH_PEAK_REGT
rgvh_pct_pm   = dly_rgvh / RGVH_PEAK_PM

eq_spy   = compound_curve(spy_ret,        100.0)
eq_regt  = compound_curve(rgvh_pct_regt,  100.0)
eq_pm    = compound_curve(rgvh_pct_pm,    100.0)

# Statistics
def stats_block(eq: pd.Series, ret: pd.Series, label: str):
    n_days = len(eq)
    yrs = n_days / 252
    cagr = (eq.iloc[-1] / 100) ** (1/yrs) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    return dict(label=label, end=eq.iloc[-1], cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd)

stats = [
    stats_block(eq_spy,  spy_ret,        "SPY total return"),
    stats_block(eq_regt, rgvh_pct_regt,  "RGVH (Reg-T 20% margin)"),
    stats_block(eq_pm,   rgvh_pct_pm,    "RGVH (Portfolio Margin 6%)"),
]
for s in stats:
    print(f"  {s['label']:35s}  ending={s['end']:7.1f}  CAGR={s['cagr']:6.2%}  "
          f"Sharpe={s['sharpe']:.2f}  max DD={s['max_dd']:.1%}")

# ---------------------------------------------------------------------------
# Build the interactive Plotly figure
# ---------------------------------------------------------------------------
tlog("building interactive figure ...")

# Pre-compute curves at common starting-capital levels so the slider just
# selects from a discrete set (smoother UX than continuous re-multiplication).
START_CAPITALS = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
DEFAULT_IDX = 3   # $100k

def scale(eq, start_cap):
    return eq * (start_cap / 100.0)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.72, 0.28], vertical_spacing=0.06,
    subplot_titles=(
        "Compounded equity on identical starting capital  ·  apples-to-apples",
        "Drawdown  (% of peak)",
    ),
)

# Add traces for each starting capital.
# Strategy idx in trace order:
#   0..n_caps-1   = SPY
#   n_caps..2n-1  = RGVH Reg-T
#   2n..3n-1      = RGVH PM
n_caps = len(START_CAPITALS)
trace_idx = {"spy": [], "regt": [], "pm": []}

for i, cap in enumerate(START_CAPITALS):
    visible = (i == DEFAULT_IDX)
    fig.add_trace(go.Scatter(
        x=eq_spy.index, y=scale(eq_spy, cap),
        mode="lines", name="SPY total return", legendgroup="spy",
        line=dict(color=COLORS["spy"], width=2.4),
        visible=visible, showlegend=visible,
        hovertemplate="<b>SPY</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    trace_idx["spy"].append(len(fig.data) - 1)

    fig.add_trace(go.Scatter(
        x=eq_regt.index, y=scale(eq_regt, cap),
        mode="lines", name="RGVH (Reg-T 20% margin)", legendgroup="regt",
        line=dict(color=COLORS["rgvh"], width=2.6),
        visible=visible, showlegend=visible,
        hovertemplate="<b>RGVH (Reg-T)</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    trace_idx["regt"].append(len(fig.data) - 1)

    fig.add_trace(go.Scatter(
        x=eq_pm.index, y=scale(eq_pm, cap),
        mode="lines", name="RGVH (Portfolio Margin 6%)", legendgroup="pm",
        line=dict(color=COLORS["positive"], width=2.6, dash="dash"),
        visible=visible, showlegend=visible,
        hovertemplate="<b>RGVH (PM)</b><br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    trace_idx["pm"].append(len(fig.data) - 1)

# Drawdown traces (in % of peak — same regardless of starting capital,
# so we add them once and they stay visible across slider positions)
def dd_pct(eq):
    return (eq / eq.cummax() - 1) * 100

fig.add_trace(go.Scatter(
    x=eq_spy.index, y=dd_pct(eq_spy), name="SPY DD",
    line=dict(color=COLORS["spy"], width=1), fill="tozeroy",
    fillcolor="rgba(21,101,192,0.18)",
    hovertemplate="SPY DD<br>%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=eq_regt.index, y=dd_pct(eq_regt), name="RGVH (Reg-T) DD",
    line=dict(color=COLORS["rgvh"], width=1), fill="tozeroy",
    fillcolor="rgba(10,37,64,0.30)",
    hovertemplate="RGVH (Reg-T) DD<br>%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=eq_pm.index, y=dd_pct(eq_pm), name="RGVH (PM) DD",
    line=dict(color=COLORS["positive"], width=1, dash="dash"), fill="tozeroy",
    fillcolor="rgba(27,94,32,0.18)",
    hovertemplate="RGVH (PM) DD<br>%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    showlegend=False,
), row=2, col=1)

# ---------------------------------------------------------------------------
# Slider — starting capital
# ---------------------------------------------------------------------------
n_total = len(fig.data)

def visibility_for_capital(active_i):
    """Return a boolean list for each trace: True if it should be visible."""
    vis = [False] * n_total
    for k, idxs in trace_idx.items():
        vis[idxs[active_i]] = True
    # The 3 drawdown traces are always visible
    vis[-3:] = [True, True, True]
    return vis

slider_steps = []
for i, cap in enumerate(START_CAPITALS):
    vis = visibility_for_capital(i)
    # Compute end-state stats for the annotation
    end_spy = scale(eq_spy,  cap).iloc[-1]
    end_rgt = scale(eq_regt, cap).iloc[-1]
    end_pm  = scale(eq_pm,   cap).iloc[-1]
    annotation_text = (
        f"<b>Starting capital: ${cap:,}</b>  →  After 12.15 years:<br>"
        f"<span style='color:{COLORS['spy']}'>SPY total return:</span> ${end_spy:,.0f}  "
        f"(+{(end_spy/cap-1):.0%})<br>"
        f"<span style='color:{COLORS['rgvh']}'>RGVH on Reg-T:</span> ${end_rgt:,.0f}  "
        f"(+{(end_rgt/cap-1):.0%})<br>"
        f"<span style='color:{COLORS['positive']}'>RGVH on PM:</span> ${end_pm:,.0f}  "
        f"(+{(end_pm/cap-1):.0%})"
    )
    slider_steps.append(dict(
        method="update",
        label=f"${cap/1000:.0f}k" if cap < 1_000_000 else f"${cap/1_000_000:.0f}M",
        args=[
            {"visible": vis,
             "showlegend": [vis[t] and t in (trace_idx["spy"][i], trace_idx["regt"][i], trace_idx["pm"][i])
                           for t in range(n_total)]},
            {"annotations[0].text": annotation_text},
        ],
    ))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
default_cap = START_CAPITALS[DEFAULT_IDX]
default_end_spy = scale(eq_spy,  default_cap).iloc[-1]
default_end_rgt = scale(eq_regt, default_cap).iloc[-1]
default_end_pm  = scale(eq_pm,   default_cap).iloc[-1]
default_annotation = (
    f"<b>Starting capital: ${default_cap:,}</b>  →  After 12.15 years:<br>"
    f"<span style='color:{COLORS['spy']}'>SPY total return:</span> ${default_end_spy:,.0f}  "
    f"(+{(default_end_spy/default_cap-1):.0%})<br>"
    f"<span style='color:{COLORS['rgvh']}'>RGVH on Reg-T:</span> ${default_end_rgt:,.0f}  "
    f"(+{(default_end_rgt/default_cap-1):.0%})<br>"
    f"<span style='color:{COLORS['positive']}'>RGVH on PM:</span> ${default_end_pm:,.0f}  "
    f"(+{(default_end_pm/default_cap-1):.0%})"
)

fig.update_layout(
    template="plotly_white",
    title=dict(
        text=("<b>Apples-to-apples</b>: same starting capital, returns compounded both sides<br>"
              "<span style='font-size:11px;color:#37474F'>"
              "RGVH daily P&L treated as a return on margin capital (Reg-T peak $17,488 or PM peak $5,246) "
              "and compounded with the same mechanics as SPY's daily total return."
              "</span>"),
        x=0.04, y=0.98, xanchor="left",
    ),
    height=820, width=1200,
    margin=dict(t=130, b=110, l=70, r=40),
    legend=dict(orientation="h", y=1.04, x=0.0, xanchor="left",
                bgcolor="rgba(0,0,0,0)"),
    font=dict(family="DejaVu Sans, Arial, sans-serif",
              color=COLORS["ink"], size=11),
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor=COLORS["bg"],
    hovermode="x unified",
    sliders=[dict(
        active=DEFAULT_IDX,
        pad=dict(t=20, b=10),
        currentvalue=dict(prefix="Starting capital: ", font=dict(size=12)),
        steps=slider_steps,
        bgcolor=COLORS["muted"],
        activebgcolor=COLORS["navy"],
        font=dict(size=11),
    )],
    annotations=[
        dict(
            text=default_annotation,
            showarrow=False,
            xref="paper", yref="paper",
            x=0.99, y=0.98,
            xanchor="right", yanchor="top",
            align="right",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=COLORS["ink_muted"],
            borderwidth=0.5,
            borderpad=8,
            font=dict(size=10),
        ),
    ],
)
fig.update_xaxes(showgrid=False, showline=True, linecolor=COLORS["ink_muted"], linewidth=0.6)
fig.update_yaxes(
    row=1, col=1, title_text="Account value ($)",
    showgrid=True, gridcolor=COLORS["muted"], gridwidth=0.4,
    zerolinecolor=COLORS["ink_muted"], tickprefix="$", separatethousands=True,
)
fig.update_yaxes(
    row=2, col=1, title_text="Drawdown (%)",
    showgrid=True, gridcolor=COLORS["muted"], gridwidth=0.4,
    zerolinecolor=COLORS["ink_muted"], ticksuffix="%",
)

# Static stats block at the bottom of the figure (always visible — the slider
# only multiplies dollars; CAGR/Sharpe/DD% are scale-invariant).
spy_st, rgt_st, pm_st = stats
stats_text = (
    f"<b>Scale-invariant statistics</b>  (period: 2013-07-05 to 2025-08-28, 12.15 years)<br>"
    f"<span style='color:{COLORS['spy']}'>SPY total return:</span> "
    f"CAGR <b>{spy_st['cagr']:.2%}</b>  ·  Vol {spy_st['vol']:.1%}  ·  "
    f"Sharpe <b>{spy_st['sharpe']:.2f}</b>  ·  Max DD {spy_st['max_dd']:.1%}<br>"
    f"<span style='color:{COLORS['rgvh']}'>RGVH (Reg-T):</span> "
    f"CAGR <b>{rgt_st['cagr']:.2%}</b>  ·  Vol {rgt_st['vol']:.1%}  ·  "
    f"Sharpe <b>{rgt_st['sharpe']:.2f}</b>  ·  Max DD {rgt_st['max_dd']:.1%}<br>"
    f"<span style='color:{COLORS['positive']}'>RGVH (PM):</span> "
    f"CAGR <b>{pm_st['cagr']:.2%}</b>  ·  Vol {pm_st['vol']:.1%}  ·  "
    f"Sharpe <b>{pm_st['sharpe']:.2f}</b>  ·  Max DD {pm_st['max_dd']:.1%}"
)
fig.add_annotation(
    text=stats_text, showarrow=False,
    xref="paper", yref="paper",
    x=0.0, y=-0.18,
    xanchor="left", yanchor="top",
    align="left",
    font=dict(size=10),
)

out = INTER / "spy_vs_rgvh_compounded.html"
fig.write_html(out, include_plotlyjs="cdn",
               config={"displaylogo": False,
                       "modeBarButtonsToRemove": ["lasso2d","select2d","autoScale2d"]})

tlog(f"saved {out}")
print(f"\nFile size: {out.stat().st_size/1024:.1f} KB")
