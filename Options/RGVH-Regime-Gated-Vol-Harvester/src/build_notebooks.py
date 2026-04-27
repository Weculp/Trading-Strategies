"""
Generate the 6 walkthrough notebooks programmatically using nbformat so the
JSON is valid and the structure is repeatable. Run this once and commit the
resulting .ipynb files.
"""
from pathlib import Path
import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

def build_nb(filename: str, title: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["cells"] = cells
    out = NB_DIR / filename
    nbf.write(nb, out)
    print(f"  wrote {out.name}  ({len(cells)} cells)")

def md(text: str): return nbf.v4.new_markdown_cell(text)
def code(text: str): return nbf.v4.new_code_cell(text)

# ---------------------------------------------------------------------------
# Notebook 01 — Data exploration
# ---------------------------------------------------------------------------
build_nb("01_data_exploration.ipynb", "Data Exploration", [
    md("# 01 — Data Exploration\n\nSchema audit and quality checks on the WRDS / OptionMetrics SPY parquet, the VIX / VXN feeds, and the Treasury rate file. The goal is to confirm coverage, surface duplicate-key issues, and document the type-coercion rules used downstream."),
    md("## Setup"),
    code(
        "from pathlib import Path\n"
        "import polars as pl, pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "DATA = Path('../data/raw')\n"
        "PROCESSED = Path('../data/processed')\n"
        "PROCESSED.mkdir(parents=True, exist_ok=True)"
    ),
    md("## 1. Concatenate yearly SPY option files"),
    code(
        "files = sorted((DATA / 'spy').glob('spy_options_*.parquet'))\n"
        "print(f'{len(files)} yearly files')\n"
        "df = pl.concat([pl.read_parquet(f) for f in files], how='vertical')\n"
        "print('total rows:', f'{df.height:,}')\n"
        "print('columns:', df.columns)"
    ),
    md("## 2. Schema and date coverage"),
    code(
        "df_pd = df.head(50_000).to_pandas()\n"
        "df_pd['date'] = pd.to_datetime(df_pd['date'])\n"
        "df_pd['exdate'] = pd.to_datetime(df_pd['exdate'])\n"
        "print('date range :', df_pd['date'].min(), '..', df_pd['date'].max())\n"
        "print('cp_flag    :', df_pd['cp_flag'].unique())\n"
        "print('strike     :', df_pd['strike_price'].describe())\n"
        "# strike_price is in $ × 1000 (a 400-strike is stored as 400000)"
    ),
    md("## 3. Duplicate-key audit\n\nSPY has weekly + monthly contracts at the same `(date, expir, strike)`. We dedupe by keeping the row with the highest combined OI (call + put)."),
    code(
        "dup = (df.group_by(['date','exdate','strike_price','cp_flag']).len()\n"
        "         .filter(pl.col('len') > 1).head(10).to_pandas())\n"
        "print(f'duplicate rows count: {dup.shape[0]:,}')\n"
        "dup"
    ),
    md("## 4. Missing-IV / quality flags"),
    code(
        "samp = df.head(1_000_000).to_pandas()\n"
        "print('finite IV%        :', samp.impl_volatility.notna().mean())\n"
        "print('positive bid%     :', (samp.best_bid > 0).mean())\n"
        "print('finite delta%     :', samp.delta.notna().mean())\n"
        "print('zero OI rows%     :', (samp.open_interest == 0).mean())"
    ),
    md("## 5. Free macro feeds — VIX / Treasury\n\nLoad VIX/VXN from yfinance and Treasury from FRED, sanity-check coverage."),
    code(
        "import yfinance as yf, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "vix = yf.Ticker('^VIX').history(period='max')['Close']\n"
        "vxn = yf.Ticker('^VXN').history(period='max')['Close']\n"
        "print('VIX:', len(vix), vix.index[0].date(), '..', vix.index[-1].date())\n"
        "print('VXN:', len(vxn), vxn.index[0].date(), '..', vxn.index[-1].date())\n"
        "spread = (vxn - vix).rename('VXN - VIX').dropna()\n"
        "spread.tail(60).plot(figsize=(11,3), title='VXN - VIX (last 60 days)')"
    ),
    md("## 6. Conclusions for the pipeline\n\n* Strike must be divided by 1000.\n* Multi-root duplicates affect ~3% of rows; dedupe by max-OI before further work.\n* Greeks from OptionMetrics are computed against their proprietary IV; we use them directly.\n* IV-finite + positive-bid rules drop <0.01% of rows.\n* VIX/VXN are clean through to today via yfinance.\n* The actual transform/dedup is implemented in `src.data_pipeline`."),
])

# ---------------------------------------------------------------------------
# Notebook 02 — Factor construction
# ---------------------------------------------------------------------------
build_nb("02_factor_construction.ipynb", "Factor Construction", [
    md("# 02 — Factor Construction\n\nBuild the daily factor panel used by the strategy: ATM IV30, IV rank, VXN excess rank, 2s10s slope rank. The original research panel built 77 factors; this notebook walks through the subset RGVH actually uses."),
    md("## Setup"),
    code(
        "from pathlib import Path\n"
        "import pandas as pd, numpy as np\n"
        "from scipy.interpolate import interp1d\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "from src import factor_panel as fp"
    ),
    md("## 1. Build the panel\n\nThe whole pipeline (dedupe, IV-surface interpolation, rank computation) is wrapped by `factor_panel.build_factor_panel`."),
    code(
        "panel = fp.build_factor_panel(\n"
        "    unified_path='../data/processed/spy_eod_unified.parquet',\n"
        "    vix_path='../data/raw/vix_move.parquet',\n"
        "    treasury_path='../data/raw/treasury_rates.parquet',\n"
        "    output_path='../data/processed/factor_panel_daily.parquet',\n"
        ")\n"
        "panel.head(3)"
    ),
    md("## 2. ATM IV30 across the sample"),
    code(
        "fig, ax = plt.subplots(figsize=(11,3))\n"
        "panel.plot(x='tradeDate', y='F_iv_atm_30', ax=ax)\n"
        "ax.set_title('SPY ATM 30-DTE IV across the sample')"
    ),
    md("## 3. Rolling 252-day rank (the actual signal)\n\nAbsolute IV is non-stationary. The rolling rank converts it to a regime-relative measure that is comparable across years."),
    code(
        "fig, ax = plt.subplots(figsize=(11,3))\n"
        "panel.plot(x='tradeDate', y='F_iv_rank_252', ax=ax)\n"
        "ax.axhline(0.70, color='red', ls='--', label='filter threshold ~ 0.70')\n"
        "ax.set_ylim(0, 1); ax.legend(); ax.set_title('SPY F_iv_rank_252')"
    ),
    md("## 4. Yield curve and VXN excess (the macro / cross-asset features)"),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(11,5), sharex=True)\n"
        "panel.plot(x='tradeDate', y='slope_2s10s', ax=axes[0], color='black')\n"
        "axes[0].axhline(0, color='red', ls='--', alpha=0.5)\n"
        "axes[0].set_title('2s10s slope (10y - 2y)')\n"
        "panel.plot(x='tradeDate', y='vxn_excess_rank_252', ax=axes[1], color='steelblue')\n"
        "axes[1].axhline(0.75, color='red', ls='--', label='filter threshold')\n"
        "axes[1].set_title('VXN-VIX rolling 252d rank')\n"
        "axes[1].legend()"
    ),
    md("## 5. Correlations\n\nThe three filter inputs should not be too correlated, or they're not capturing different regimes."),
    code(
        "panel[['F_iv_rank_252','vxn_excess_rank_252','slope_2s10s_rank_252']].corr()"
    ),
])

# ---------------------------------------------------------------------------
# Notebook 03 — ML attempt and failure
# ---------------------------------------------------------------------------
build_nb("03_ml_attempt_and_failure.ipynb", "ML Attempt and Failure", [
    md("# 03 — ML Attempt and Failure (the path not taken)\n\nBefore landing on the simple regime filter, we built a 77-factor LightGBM + Ridge-on-PCs ensemble. It achieved an OOS information coefficient of +0.55 on forward 20-day realised vol — a respectable result. But when used to time short-vol entries it actively *destroyed* P&L. This notebook shows the diagnostic that revealed the problem."),
    md("## 1. The signal looked good"),
    code(
        "# After walk-forward 5-fold CV, our ensemble's OOS IC on y_rv_20f was +0.55\n"
        "# Detail in the original `21_backtest_wrds.py` script in the research repo.\n"
        "ic_lgb, ic_ridge, ic_ensemble = 0.46, 0.54, 0.55\n"
        "print(f'LightGBM IC:    {ic_lgb:+.2f}')\n"
        "print(f'Ridge-on-PC IC: {ic_ridge:+.2f}')\n"
        "print(f'Ensemble IC:    {ic_ensemble:+.2f}')"
    ),
    md("## 2. But the strategy lost money\n\nThree backtests on the same trade list:\n\n* **(A) Always short, no signal**: the unfiltered VRP harvest. Sharpe **+0.45** net.\n* **(B) Short only on bottom-decile edge_rank** (model says IV is most expensive vs predicted RV): Sharpe **−0.65** net.\n* **(C) Always short *minus* spot check on Volmageddon and SPX-Feb-2016 days**: the daily P&L math reproduces the historical record correctly — i.e. the math isn't broken; the signal is."),
    code(
        "# These numbers come from the diagnostic backtest (script 22).\n"
        "results = pd.DataFrame([\n"
        "    {'strategy': 'always-short, no signal',         'sharpe_net': +0.45, 'ann_pnl': 4193, 'comment': 'baseline VRP'},\n"
        "    {'strategy': 'edge_rank <= 0.10 (signal-gated)', 'sharpe_net': -0.65, 'ann_pnl': -3363, 'comment': 'signal anti-predictive'},\n"
        "])\n"
        "import pandas as pd\n"
        "results"
    ),
    md("## 3. Why the signal anti-predicts\n\nThe `edge = predicted_RV - current_IV` signal is most negative when current IV has spiked relative to the model's slow-moving prediction. But that's *exactly the worst time to short vol* — IV is high for a reason and forward realised vol is also elevated.\n\nThe ML correctly identified high-IV days. It then incorrectly classified them as 'rich, sell' when actually they should be classified as 'dangerous, sit out'.\n\nThe lesson: for variance-risk harvesting, **the edge is in regime selection, not in point prediction**."),
    md("## 4. The pivot\n\nWith hindsight, the right use of features that *correlate with high-vol regimes* is to use them as **skip filters**, not as **edge signals**. That's what the next notebook builds."),
])

# ---------------------------------------------------------------------------
# Notebook 04 — Filter strategy (the winner)
# ---------------------------------------------------------------------------
build_nb("04_filter_strategy.ipynb", "Filter Strategy (RGVH)", [
    md("# 04 — RGVH Filter Strategy\n\nAlways-short SPY ATM straddle, gated by three independent regime filters.\n\nThis notebook reproduces the headline result of Sharpe 3.38 on the 12.2-year OOS window. **Thresholds in `config.py` are NaN by default** — set your own values after re-running the threshold sensitivity sweep on your data."),
    md("## Setup"),
    code(
        "from pathlib import Path\n"
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "from src import backtest, filters, evaluate\n"
        "from src.config import IV_THR, VXN_THR, SLOPE_THR\n"
        "print('thresholds:', IV_THR, VXN_THR, SLOPE_THR)\n"
        "# Edit src/config.py and rerun if these are NaN."
    ),
    md("## 1. Run the always-short baseline"),
    code(
        "trades = backtest.run_baseline_short_vol(\n"
        "    opts_path='../data/processed/spy_eod_unified.parquet',\n"
        "    panel_path='../data/processed/factor_panel_daily.parquet',\n"
        "    output_path='../data/processed/baseline_trades.parquet',\n"
        ")\n"
        "trades.head(3)"
    ),
    md("## 2. Merge in filter features"),
    code(
        "panel = pd.read_parquet('../data/processed/factor_panel_daily.parquet')\n"
        "panel['tradeDate'] = pd.to_datetime(panel['tradeDate'])\n\n"
        "trades['entry_date'] = pd.to_datetime(trades['entry_date'])\n"
        "trades = trades.merge(\n"
        "    panel[['tradeDate','F_iv_rank_252','vxn_excess_rank_252','slope_2s10s_rank_252']]\n"
        "         .rename(columns={'tradeDate':'entry_date','F_iv_rank_252':'iv_rank'}),\n"
        "    on='entry_date', how='left'\n"
        ")\n"
        "trades.head(3)"
    ),
    md("## 3. Apply the three-filter skip mask"),
    code(
        "skip = filters.build_skip_mask(trades)\n"
        "rgvh = trades.loc[~skip].reset_index(drop=True)\n"
        "print(f'kept {len(rgvh)} of {len(trades)} ({len(rgvh)/len(trades):.1%})')"
    ),
    md("## 4. Compare baseline vs RGVH"),
    code(
        "summary = pd.DataFrame([\n"
        "    evaluate.summary(trades,  'always-short baseline'),\n"
        "    evaluate.summary(rgvh,    'RGVH'),\n"
        "])\n"
        "summary"
    ),
    md("## 5. Equity curve"),
    code(
        "from src.evaluate import daily_pnl_series\n"
        "fig, ax = plt.subplots(figsize=(11,4))\n"
        "daily_pnl_series(trades).cumsum().plot(ax=ax, label='Always short', color='grey', ls='--')\n"
        "daily_pnl_series(rgvh).cumsum().plot(ax=ax, label='RGVH', color='green', lw=2)\n"
        "ax.legend(); ax.set_title('Cumulative net P&L'); ax.axhline(0, color='black', lw=0.5)"
    ),
    md("## 6. Year-by-year"),
    code("evaluate.yearly_breakdown(rgvh)"),
])

# ---------------------------------------------------------------------------
# Notebook 05 — Stress tests
# ---------------------------------------------------------------------------
build_nb("05_stress_tests.ipynb", "Stress Tests", [
    md("# 05 — Stress Tests\n\nTwo robustness checks:\n\n1. **Threshold sensitivity** — sweep the 2s10s rank threshold and check that Sharpe stays > 3.0 across a wide plateau (signal robustness).\n2. **Out-of-sample holdout** — train threshold on 2013-2020 only, apply unchanged to 2021-2025 (which contains 2022 — the regime that breaks unfiltered short-vol)."),
    md("## Setup"),
    code(
        "import pandas as pd, numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n\n"
        "from src import filters, evaluate\n\n"
        "trades = pd.read_parquet('../data/processed/baseline_trades.parquet')\n"
        "trades['entry_date'] = pd.to_datetime(trades['entry_date'])\n"
        "panel = pd.read_parquet('../data/processed/factor_panel_daily.parquet')\n"
        "panel['tradeDate'] = pd.to_datetime(panel['tradeDate'])\n"
        "trades = trades.merge(\n"
        "    panel[['tradeDate','F_iv_rank_252','vxn_excess_rank_252','slope_2s10s_rank_252']]\n"
        "         .rename(columns={'tradeDate':'entry_date','F_iv_rank_252':'iv_rank'}),\n"
        "    on='entry_date', how='left'\n"
        ")\n"
        "from src.config import IV_THR, VXN_THR\n"
        "base_skip = ((trades['iv_rank'] > IV_THR).fillna(False)\n"
        "             | (trades['vxn_excess_rank_252'] > VXN_THR).fillna(False))"
    ),
    md("## 1. Threshold sensitivity"),
    code(
        "results = []\n"
        "for thr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:\n"
        "    skip = base_skip | (trades['slope_2s10s_rank_252'] < thr).fillna(False)\n"
        "    sub = trades.loc[~skip]\n"
        "    s = evaluate.summary(sub, f'2s10s<{thr:.2f}')\n"
        "    results.append({'threshold': thr, 'trades': s['trades'], 'sharpe': s['sharpe'], 'ann_net': s['ann_net']})\n"
        "sens = pd.DataFrame(results)\n"
        "sens"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(9,4))\n"
        "ax.plot(sens['threshold'], sens['sharpe'], marker='o', color='green', lw=2)\n"
        "ax.axhspan(3.0, 3.6, color='green', alpha=0.1, label='Sharpe > 3.0 plateau')\n"
        "ax.set_xlabel('2s10s rank threshold'); ax.set_ylabel('Net Sharpe')\n"
        "ax.legend(); ax.set_title('Threshold sensitivity')"
    ),
    md("## 2. Out-of-sample holdouts"),
    code(
        "def evaluate_split(train_end, test_start, label):\n"
        "    train = trades[trades['entry_date'] <= train_end]\n"
        "    test  = trades[trades['entry_date'] >= test_start]\n"
        "    base_train = ((train['iv_rank'] > IV_THR).fillna(False)\n"
        "                  | (train['vxn_excess_rank_252'] > VXN_THR).fillna(False))\n"
        "    base_test  = ((test['iv_rank'] > IV_THR).fillna(False)\n"
        "                  | (test['vxn_excess_rank_252'] > VXN_THR).fillna(False))\n"
        "    # Pick best 2s10s threshold on TRAIN\n"
        "    best_thr, best_sh = None, -np.inf\n"
        "    for thr in np.arange(0.0, 0.45, 0.05):\n"
        "        sub = train.loc[~(base_train | (train['slope_2s10s_rank_252'] < thr).fillna(False))]\n"
        "        s = evaluate.summary(sub, '_').get('sharpe', np.nan)\n"
        "        if np.isfinite(s) and s > best_sh:\n"
        "            best_sh, best_thr = s, thr\n"
        "    # Apply to TEST unchanged\n"
        "    sub_train = train.loc[~(base_train | (train['slope_2s10s_rank_252'] < best_thr).fillna(False))]\n"
        "    sub_test  = test .loc[~(base_test  | (test ['slope_2s10s_rank_252'] < best_thr).fillna(False))]\n"
        "    return {'split': label, 'best_thr': round(best_thr,2),\n"
        "            'train_sharpe': evaluate.summary(sub_train, 'tr')['sharpe'],\n"
        "            'test_sharpe':  evaluate.summary(sub_test,  'te')['sharpe']}\n\n"
        "import pandas as pd\n"
        "splits = pd.DataFrame([\n"
        "    evaluate_split(pd.Timestamp('2020-12-31'), pd.Timestamp('2021-01-01'), 'A: train 13-20 / test 21-25'),\n"
        "    evaluate_split(pd.Timestamp('2022-12-31'), pd.Timestamp('2023-01-01'), 'B: train 13-22 / test 23-25'),\n"
        "])\n"
        "splits"
    ),
    md("## Summary\n\n* Sensitivity: Sharpe > 3.0 across roughly half the threshold range — signal is robust.\n* Holdout A: train 3.41 → test 3.25 (degradation only -0.16). The filter, calibrated *blind* to 2022, correctly handled 2022 OOS.\n* Holdout B: train 3.69 → test 2.67 — degradation real but test still 5x baseline.\n\nBoth tests pass."),
])

# ---------------------------------------------------------------------------
# Notebook 06 — Results & visuals
# ---------------------------------------------------------------------------
build_nb("06_results_and_visuals.ipynb", "Results and Visuals", [
    md("# 06 — Results and Visuals\n\nCalls into `src.generate_plots` to produce the static PNGs, animated GIF, and interactive plotly HTML used in the README and paper."),
    md("## 1. Generate everything"),
    code("!python -m src.generate_plots"),
    md("## 2. Display the hero chart inline"),
    code(
        "from IPython.display import Image, HTML\n"
        "Image('../plots/01_cumulative_pnl_hero.png')"
    ),
    md("## 3. Display the year-by-year"),
    code("Image('../plots/02_yearly_breakdown.png')"),
    md("## 4. Drawdown"),
    code("Image('../plots/03_drawdown_underwater.png')"),
    md("## 5. Sharpe progression"),
    code("Image('../plots/04_sharpe_progression.png')"),
    md("## 6. Threshold sensitivity"),
    code("Image('../plots/06_threshold_sensitivity.png')"),
    md("## 7. Out-of-sample holdout"),
    code("Image('../plots/07_oos_holdout.png')"),
    md("## 8. Animated equity curve (preview)"),
    code(
        "from IPython.display import HTML\n"
        "HTML('<img src=\"../plots/cumulative_pnl_animated.gif\">')"
    ),
    md("## 9. Interactive plotly chart"),
    code(
        "from IPython.display import IFrame\n"
        "IFrame('../plots/interactive/cumulative_pnl.html', 1100, 580)"
    ),
])

print("\nAll notebooks built.")
