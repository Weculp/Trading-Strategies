# LinkedIn launch — caption + media options

## Recommended single-media format: the animated GIF ⭐

[`plots/pro/100k_three_strategies.gif`](plots/pro/100k_three_strategies.gif) (1.3 MB, 10.8 sec, 1430×770 px) — auto-plays in LinkedIn feed, tells the whole story in 10 seconds: $100k starting capital, 12 years compounded, RGVH on Portfolio Margin reaches $1M while S&P 500 reaches $491k. Maximum hook.

## Alternative: 4-image carousel

If you'd rather post a static carousel:

1. `plots/pro/01_executive_summary.png` (the exec summary — opens the story)
2. `plots/pro/03_rgvh_vs_spy_dashboard.png` (4-panel head-to-head)
3. `plots/pro/02_equity_curve_pro.png` (the equity curve)
4. `plots/pro/06_threshold_sensitivity.png` (plateau — defends against "curve-fit" comments)

---

## Variant A — "story + result" (recommended for max reach)

> I spent the last few weeks building an options trading strategy.
>
> First version used a 77-factor ML pipeline (LightGBM + Ridge) to predict forward realised volatility. The model achieved a respectable +0.55 OOS information coefficient. I was excited.
>
> Then I backtested it across 12 years of SPY options data.
>
> It lost money. Net Sharpe: −1.37.
>
> The signal was technically right (it identified high-vol days correctly) but strategically wrong (those are exactly the worst days to short volatility).
>
> So I threw it out and tried something simpler:
>
> Always sell SPY ATM straddles. Skip the trade if any of three regime filters fire:
> 1. SPY's IV percentile is in its upper range (vol already elevated)
> 2. Nasdaq vol is pulling away from broad-market vol (cross-asset stress)
> 3. The 2s10s yield curve is inverted (recession-warning regime)
>
> No ML. Three rules. Each based on well-known academic literature.
>
> Result: net Sharpe **3.38** over 12 years out-of-sample, max drawdown only 1.3× annual P&L, robust on a wide threshold plateau, survives strict train/test holdout.
>
> Sometimes the lesson is that you don't need ML — you just need to know when not to trade.
>
> Full writeup, code, paper, and visualisations: github.com/Weculp/Trading-Strategies
>
> #QuantitativeTrading #OptionsTrading #Volatility #Python #ResearchPaper

---

## Variant B — "thesis-driven, more academic"

> Variance Risk Premium harvesting on equity-index options is a well-documented strategy in academic finance (Bondarenko 2004, Carr & Wu 2009). The expected return is positive on average, but the distribution has a fat left tail concentrated in specific regimes.
>
> I asked: can a deliberately minimal regime-gating layer cap that left tail without sacrificing the harvest?
>
> Answer (12 years of strict OOS backtest on WRDS OptionMetrics SPY data):
>
> ✅ Net Sharpe **3.38** (vs 0.48 for unfiltered VRP)
> ✅ Max drawdown 1.3× annual P&L
> ✅ Hit rate 68%
> ✅ Robust on a wide threshold plateau (Sharpe > 3.0 across most cutoffs)
> ✅ Survives strict train/test holdout where 2022 (the regime-breaking year) is in the test set
>
> The three filters: IV-percentile rank, VXN-VIX cross-asset spread, and 2s10s yield-curve inversion. Each independently captures a documented driver of historical short-vol drawdowns.
>
> The paper also documents a methodological negative result — a 77-factor ML ensemble that achieved +0.55 IC on forward realised vol but failed to generate profitable trade timing, because the signals coincided with the regimes where short-vol entries are most dangerous.
>
> Code, notebooks, formal writeup, and reproducibility instructions:
> 🔗 github.com/Weculp/Trading-Strategies
>
> #Quant #SystematicTrading #Volatility #VRP #DerivativesResearch

---

## Variant C — "punchy, retail-friendly"

> Built a trading strategy that does one thing: sells SPY straddles every day, but doesn't trade when any of three signals say "stress is coming."
>
> 12 years of out-of-sample backtest. Net Sharpe 3.38 after costs. Max drawdown 1.3× annual profit.
>
> The three signals are dead simple:
> 🟧 SPY's IV is already too high
> 🟥 Nasdaq vol is moving faster than broad-market vol
> 🟦 The yield curve is inverted
>
> No ML. No black box. Three rules, each from textbook macro/finance literature.
>
> Open-source: code, paper, results, and the failed ML attempt I threw out before this:
> 👉 github.com/Weculp/Trading-Strategies
>
> #Options #VolatilityTrading #Python #OpenSource

---

## Engagement notes

- **Best post timing**: Tuesday or Wednesday, 9-11am ET, when the quant/finance crowd is online.
- **Reply hook**: in the comments, drop a line like "happy to walk through any specific year's P&L logic if anyone's curious". Encourages comments and re-engagement.
- **Pinned reply**: link to the repo + paper PDF directly from the top comment so the algorithm sees engagement.
- **Re-share** ~3 days later: post the threshold-sensitivity image alone with caption "the most important chart from last week's strategy post — wide plateau = signal is real, not curve-fit". Different angle, same content.
- **For your website later**: embed the interactive plotly HTML (`plots/interactive/cumulative_pnl.html`) directly in an iframe — gives readers a hover-able equity curve.

## Disclaimer to add (if you want to be cautious)

> Educational/research material. Not financial advice. Past simulated performance does not guarantee future results. Real options trading involves substantial risk of loss.

---

## Optional follow-up posts (drip campaign)

After the launch post lands, you can space out follow-ups:

**Post 2 (1 week later)**: "Here's the negative result from the project I posted last week — a 77-factor ML ensemble that scored well on the prediction task but lost money on the strategy. Why simpler rules beat ML for variance-risk harvesting." → links to notebook 03.

**Post 3 (2 weeks later)**: "What QQQ told me about diversification — 0.76 daily P&L correlation between two equity-vol books means almost zero diversification. Real diversification needs bond vol." → screenshots of correlation table.

**Post 4 (1 month later)**: "Three months into paper-trading the RGVH strategy. Here's what's matching the backtest and what isn't." → live numbers vs backtest numbers.
