# Trading Strategies

A collection of quantitative trading strategies, organised by asset class. Each strategy folder contains:

- **README** with the executive summary
- **PAPER** (Markdown + PDF) with the full research writeup
- **`src/`** with cleaned, importable Python modules
- **`notebooks/`** with reproducible Jupyter walkthroughs
- **`plots/`** with static figures, GIFs, and interactive HTMLs
- **`results/`** with anonymised summary tables

---

## Index

### Options
| Strategy | One-liner | Sharpe (net) | Sample |
|---|---|---|---|
| [RGVH — Regime-Gated Vol Harvester](Options/RGVH-Regime-Gated-Vol-Harvester) | Always-short SPY ATM straddle, gated by IV regime + cross-asset stress + yield-curve inversion | 3.38 (12y OOS) | 2013-07 → 2025-08 |

*More to come.*

---

## Disclaimer

This repository is **research and educational material only**. Nothing here constitutes financial, investment, legal, or tax advice. Past simulated performance does not guarantee future results. Options trading involves substantial risk of loss. Do your own research; deploy with real capital at your own risk.

Proprietary data sources (e.g., WRDS / OptionMetrics) used to produce backtests are **not redistributed** in this repository. Each strategy folder includes a `data/README.md` with sourcing instructions.

## License

[MIT](LICENSE) — permissive use with attribution. See LICENSE for the full text.

## Author

[Weculp](https://github.com/Weculp)
