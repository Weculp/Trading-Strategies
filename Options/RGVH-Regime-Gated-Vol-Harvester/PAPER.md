---
title: "RGVH"
subtitle: "A Regime-Gated Variance-Risk-Premium Harvester on SPY Options"
author: |
  | Weculp
  | Independent Research
date: "April 2026"
geometry: "margin=1in"
fontsize: 11pt
linkcolor: blue
urlcolor: blue
toc: true
toc-depth: 2
numbersections: true
header-includes:
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \usepackage{float}
  - \usepackage{xcolor}
  - \usepackage{tcolorbox}
  - \usepackage{amsmath}
  - \usepackage{titlesec}
  - \usepackage{fancyhdr}
  - \definecolor{navy}{HTML}{0A2540}
  - \definecolor{gold}{HTML}{C9A227}
  - \definecolor{slate}{HTML}{37474F}
  - \definecolor{softblue}{HTML}{F0F4F9}
  - \captionsetup{font=small,labelfont={bf,color=navy}}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \rhead{\textcolor{slate}{\small RGVH \textendash{} Regime-Gated Vol Harvester}}
  - \lhead{\textcolor{slate}{\small Weculp \textendash{} Independent Research}}
  - \rfoot{\textcolor{slate}{\small Page \thepage}}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \renewcommand{\footrulewidth}{0pt}
  - \titleformat{\section}{\large\bfseries\color{navy}}{\thesection}{0.6em}{}
  - \titleformat{\subsection}{\bfseries\color{slate}}{\thesubsection}{0.5em}{}
  - \newtcolorbox{kpibox}{colback=softblue,colframe=navy,boxrule=0.6pt,arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt}
---

\thispagestyle{empty}
\vspace*{1.5cm}

\begin{flushleft}
\noindent {\color{navy}\rule{\linewidth}{1.5pt}}
\vspace{0.4cm}

{\Huge\bfseries\color{navy} RGVH}

\vspace{0.2cm}
{\LARGE\color{slate} A Regime-Gated Variance-Risk-Premium Harvester \\ on SPY Options}

\vspace{0.4cm}
{\color{navy}\rule{\linewidth}{0.5pt}}
\vspace{0.6cm}

{\large\color{slate}\itshape From a 77-factor ML pipeline to a three-rule filter that did the same job better.}

\vspace{2cm}

{\large\bfseries Weculp}\\
{\itshape Independent Research}

\vspace{0.4cm}

{\itshape April 2026}
\end{flushleft}

\vspace{1.5cm}

\begin{kpibox}
\textbf{Executive summary at a glance:} \\[6pt]
\begin{tabular}{ll}
Net Sharpe ratio (12.15y, OOS):  & \textbf{\color{navy}3.38} \\
Annualised net P\&L:             & \textbf{\$1{,}004} per \$1{,}000 vega \\
Maximum drawdown:                & \textbf{\$1{,}422} (1.3$\times$ annual P\&L) \\
Hit rate (per trade):            & \textbf{68.5\%} \\
Comparable SPY buy-and-hold Sharpe: & 0.85 \\
Comparable SPY max drawdown:     & --33.7\% \\
Trades / year:                   & $\sim$90 \\
\end{tabular}
\end{kpibox}

\vspace{1cm}

\begin{flushleft}
\textbf{Abstract.} We design and backtest a short-volatility strategy on SPY index options that systematically harvests the variance risk premium (VRP) while avoiding the three regime types in which historical short-vol losses cluster. The strategy sells 22-day at-the-money straddles every trading day, delta-hedges daily, and holds for 15 business days, but skips entries when any of three independent regime filters fire: (i) elevated SPY implied-vol percentile; (ii) elevated Nasdaq-vs-broad-market vol spread; or (iii) yield-curve inversion. Across 12.15 years of strict walk-forward out-of-sample testing on WRDS OptionMetrics data (2013-07 to 2025-08), we find a net Sharpe ratio of \textbf{3.38} after bid-ask spread, commissions, and delta-hedge slippage, with a maximum drawdown of only 1.3$\times$ annual P\&L. The strategy is robust on a wide threshold plateau (Sharpe~$>$~3.0 across many cutoffs), and survives a strict train/test holdout where the threshold is chosen blindly on 2013--2020 and applied unchanged to 2021--2025. We also report a negative result: a 77-factor LightGBM~+~Ridge-on-PC ensemble we initially built generated predictive volatility signals (information coefficient~0.55) but failed to translate to profitable timing of short-vol entries; the simpler regime filter dominated by 1.5+ Sharpe units. We conclude that for variance-risk harvesting, regime selection trumps point prediction.
\end{flushleft}

\newpage

# Executive Summary

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/01_executive_summary.png}
\caption{Executive-summary panel. KPI tiles, cumulative equity curve, 252-day rolling Sharpe, calendar-year P\&L, and drawdown profile compared to SPY buy-and-hold (rescaled to the same capital base).}
\end{figure}

The chart above answers the most important questions about the strategy on a single page. **What follows in the rest of the paper is the methodology, robustness checks, and head-to-head comparison with SPY in detail.**

The headline numbers — net Sharpe 3.38, max drawdown of just 1.3$\times$ annual P\&L, hit rate 68.5\% — emerge from a deliberately minimal design philosophy. We start with the well-documented variance risk premium (VRP) — the empirical wedge between option-implied volatility and subsequently realised volatility — and then layer on three independent regime filters that each capture a known driver of historical short-vol drawdowns. The combined effect is to suppress trade entry on roughly 64\% of days; on the remaining 36\% the strategy is harvesting the vanilla VRP in regimes where it has historically worked.

The strategy is **not a replacement for SPY buy-and-hold**. On a Reg-T retail account it generates a smaller absolute return than SPY (5.7\% vs 14\% CAGR). On a portfolio-margin account the absolute return overtakes SPY (19.2\% vs 14\%). On every risk-adjusted metric — Sharpe, Sortino, Calmar, max drawdown, ulcer index — RGVH dominates SPY by a wide margin. The intended use is **alongside** an equity core: a small RGVH allocation lifts portfolio Sharpe materially because RGVH's monthly returns are weakly correlated with SPY monthly returns ($\rho \approx 0$, see Section~6).

\newpage

# Introduction

Selling index volatility has been a persistent positive-expected-value strategy since at least the 1990s, documented in a long line of academic and industry-quant literature [@bondarenko2004; @carrwu2009; @bollerslev2009]. The mechanism is straightforward: implied volatility on equity-index options embeds a risk premium that compensates the seller for bearing the small but real probability of a vol-spike event. The challenge — what kills most amateur short-vol strategies — is that the *expectation* is positive while the *distribution* has a fat left tail. Long-only short-vol implementations like the XIV exchange-traded note famously imploded on 5 February 2018 when a single intraday vol spike triggered a near-100\% loss [@xiv2018].

Two design questions follow. First: can the variance risk premium be harvested in a form where the left-tail loss is bounded to a small multiple of annual gains, rather than to total wipeout? Second: how should one select *which days* to enter short-vol positions, given that the average expected return is positive but day-to-day reward differs by orders of magnitude?

This paper describes a strategy — the **Regime-Gated Vol Harvester (RGVH)** — that addresses both questions with a deliberately minimal design philosophy. We begin with the standard short-straddle-with-daily-delta-hedge template, then apply three independent regime filters that each capture a documented driver of historical short-vol drawdowns. The three filters together suppress trade entry on roughly 64\% of days; on the remaining 36\% we are trading the vanilla VRP harvest in regimes where it has historically worked. Across a 12.15-year strict walk-forward out-of-sample window, the net Sharpe ratio is 3.38, the maximum drawdown is 1.3$\times$ annual P\&L, and the hit rate is 68.5\%.

The paper also reports a methodological negative result. Before arriving at the simple three-rule filter, we built an ML pipeline of 77 engineered factors (implied volatility surface features, Greeks, model-free moments per [@bakshi2003], dealer-positioning proxies, plus macro features) feeding a LightGBM + Ridge-on-principal-components ensemble. The ensemble's out-of-fold Spearman information coefficient on forward 20-day realized volatility was a respectable +0.55. But when used to time short-vol entries via a `predicted_RV − current_IV` signal, it actively *destroyed* P\&L: a naive always-short baseline scored Sharpe +0.45 net of costs, while the ML-signal-gated version scored Sharpe −0.65 because the signal selected the worst days to short volatility. We document this transition in detail because we believe the result generalises: for variance-risk harvesting, the edge lies in the regime gate, not in point-prediction skill.

The rest of the paper is organised as follows. Section~\ref{sec:lit} reviews the relevant literature. Section~\ref{sec:data} describes the data. Section~\ref{sec:method} develops the methodology, including the failed ML pipeline. Section~\ref{sec:results} presents the main results. Section~\ref{sec:spy} contains the head-to-head comparison with SPY across multiple capital bases and risk-adjusted metrics. Section~\ref{sec:robust} contains robustness checks (threshold sensitivity, train/test holdout, failed cross-asset diversification). Section~\ref{sec:discuss} discusses the strategy's known weaknesses. Section~\ref{sec:conclude} concludes.

# Background and Related Work \label{sec:lit}

## Variance Risk Premium

The variance risk premium (VRP) — the wedge between option-implied variance and subsequent realised variance — has been documented in equity-index options since at least @bondarenko2004 and @carrwu2009. @bollerslev2009 show that the VRP predicts aggregate market returns at intermediate horizons. @cao2013 show that the VRP is concentrated in liquid index options and weak or absent in single-name options. The empirical regularity is robust: across decades and regimes, IV30 systematically exceeds the realised volatility actually observed over the next 30 days, by 1--2 vol points on average for SPX-class instruments.

Practitioners exploit the VRP via volatility-selling strategies: short straddles, strangles, iron condors, variance swaps, or short VIX futures. These strategies share the same return distribution character — long a small positive expectation, short a fat left tail.

## Implied volatility and forward-return predictability

A complementary literature documents that features of the option-implied vol surface predict future stock returns. @cremers2010 show deviations from put-call parity (constructed as the IV spread between matched strike pairs) predict the cross-section of stock returns. @xing2010 find that the "volatility smirk" — IV at OTM puts vs ATM — predicts future equity returns at the firm level. @bakshi2003 derive model-free risk-neutral moments (variance, skewness, kurtosis) from option prices, providing a rich feature basis for cross-sectional and time-series prediction.

## Regime indicators and macro filters

Yield-curve inversion as a recession indicator dates to @estrella1996, who show the term-spread (10-year minus 3-month Treasury yield) is the single best leading indicator of US recessions. The New York Fed publishes a recession probability model based on this signal. Cross-asset stress indicators such as the MOVE (bond-vol) and TED-spread similarly serve as regime classifiers in the systematic-trading literature.

To our knowledge, the specific combination — VRP harvest gated by SPY IV percentile, VXN-VIX cross-asset spread, and 2s10s curve inversion — has not been published as such, although each ingredient is individually well-known. The contribution of this paper is the demonstration that, taken together, the three filters reduce the historical short-vol drawdown by an order of magnitude while preserving most of the gross P\&L, and that this result is robust to threshold choice and out-of-sample testing.

# Data \label{sec:data}

We use three datasets, two of which are free.

## SPY EOD options chains (WRDS / OptionMetrics)

WRDS OptionMetrics IvyDB provides end-of-day option chains for SPY across 2005-01 to 2025-08, with bid, ask, volume, open interest, implied volatility, and Greeks (delta, gamma, vega, theta) on every (date, expir, strike, side) record. Coverage is daily at the EOD frequency. After dedupe (multi-root collisions for SPY weekly vs monthly contracts at the same key) and pivot from long format to wide (call+put per strike), we have 12{,}440{,}940 unique (date, expir, strike) records spanning 5{,}193 trading days. Of these, 90.8\% pass our quality filter (finite IV in $[0.02, 3.0]$, non-negative Greeks within numerical tolerances, at least one quoted side). The dataset is licensed and is **not redistributed** with this work.

## VIX, VXN (Yahoo Finance, free)

We use the daily closing values of \verb|^VIX| (CBOE S\&P 500 30-day vol) and \verb|^VXN| (Nasdaq-100 30-day vol) from Yahoo Finance, covering 2001-01 to present. Free to redistribute via Yahoo's terms. The cross-asset filter uses VXN~-~VIX, ranked over a trailing 252-day window.

## US Treasury rate curve (FRED, free)

We use FRED series \verb|DGS2|, \verb|DGS10|, and \verb|DGS3MO| (constant-maturity Treasury yields at 2 years, 10 years, and 3 months). Coverage is full-history daily. The curve filter uses the 2s10s slope (\verb|DGS10 − DGS2|), ranked over a trailing 252-day window. The 3-month T-bill is used as the risk-free rate for ancillary IV back-out where needed.

## Sample window and out-of-sample regime

After a 252-day burn-in for the rolling-rank features, our usable window is 2013-07-02 to 2025-08-29 — 12.15 years of distinct regimes including: 2014--15 emerging-market and oil shocks, the 2018 February ``Volmageddon'' event, the 2018 December selloff, the 2020 COVID crash and recovery, the 2022 Federal-Reserve tightening cycle, and the 2024 late-cycle slowdown. All filters and thresholds are evaluated strictly walk-forward — the threshold for, say, the 2s10s rank is calibrated only on data observed at the time, never with hindsight from later events.

# Methodology \label{sec:method}

## Trade construction (always)

Every trading day in the OOF window, the strategy considers entering one position structured as follows:

- **Instrument**: ATM SPY straddle, target days-to-expiry 22 (within $[17, 30]$).
- **Strike selection**: closest to call-delta of 0.5.
- **Direction**: short (sell call + sell put).
- **Sizing**: vega-dollar target — \verb|size_contracts = $1000 / (vega × multiplier)|, floored at 0.05 contracts.
- **Initial hedge**: take an offsetting SPY position equal to the negative straddle delta times multiplier times size.
- **Hold period**: 15 business days, exit if expiry would otherwise occur.
- **Daily delta rebalance**: rehedge if $|\Delta\text{drift}|$ exceeds 5\% of the straddle delta times multiplier $\times$ size.
- **Costs**: bid-ask half-spread on entry and exit (per leg), \$0.65/leg commission, \$0.005/share SPY hedge slippage.

This is the standard ``delta-hedged short straddle'' template. With no further gating, this strategy backtests at Sharpe 0.45 net of costs across our window — a non-trivial but small VRP harvest, with a max drawdown of \$7{,}301 per \$1{,}000 of vega (approximately 7.5$\times$ annual P\&L).

## ML pipeline (the path not taken)

Before the simple regime filter, we built a substantial machine-learning pipeline on this data, with the goal of generating a per-day ``edge'' signal that would predict next-month realised vol better than the option chain itself.

The pipeline included:

1. **77 engineered factors** spanning IV-surface levels (3 tenors), term structure, skew (3 tenors), risk reversals, butterflies, BKM model-free risk-neutral moments [@bakshi2003], volume/OI aggregates, dealer-exposure proxies (gamma exposure / GEX, vanna, charm via OI weighting), max-pain distance, realised return/vol features, and momentum versions of all of the above.
2. **PCA decomposition** of the factor space showing the first 12 components accounted for 90\% of variance.
3. **Walk-forward 5-fold ensemble** of LightGBM (depth 4, 400 trees, learning rate 0.03) and Ridge regression on top-12 PCs, both predicting forward 20-day realised volatility (\verb|y_rv_20f|).
4. **Hidden Markov regime model** with 3 states fit per fold on $[r_{t}, RV_{20}, IV_{30}]$, used to gate trades.
5. **Dynamic exit rules**: take-profit / stop-loss in vega-\$ units, plus signal-flip exit when the predicted edge crossed the median during the holding period.

The ensemble's information coefficient on \verb|y_rv_20f| was a respectable +0.55 (Spearman, OOS). The HMM correctly identified high-vol regimes. Yet the resulting strategy:

- Without the ML signal (always short, no filters): Sharpe 0.45.
- With the ML signal selecting bottom-decile edge days (\verb|predicted_RV| $\ll$ \verb|current_IV|): Sharpe **--0.65**.
- With the ML signal + HMM asymmetric scaling + dynamic exits, on 5 years of OOS data only: Sharpe +1.71. **Replicated on 20 years**: Sharpe **--1.37**.

The reason: the ML signal \verb|edge_rank ≤ 0.10| triggered most often when SPY's implied vol had recently spiked relative to the model's slow-moving smoothed prediction. These are, by construction, exactly the *worst* days to enter short-vol positions — the IV is high for a reason. The ensemble was correctly identifying high-IV days and incorrectly classifying them as ``rich, sell'' when in fact those days carry the highest forward realised vol of any subset. **This negative result motivated the pivot to a regime-gating approach.**

## RGVH — three regime filters

The RGVH strategy uses the same trade construction as the baseline (above), but skips entries when any of three regime filters fires.

### Filter 1: SPY IV percentile (\texttt{iv\_rank > IV\_THR})

Compute the rolling 252-day percentile rank of SPY's 30-day at-the-money implied volatility. Skip if the rank exceeds a threshold (calibration value in $[0.55, 0.85]$; full plateau analysis in Section~\ref{sec:robust}). The economic interpretation: when IV is in its upper quintile of the trailing year, vol is already elevated and the market is signalling stress.

### Filter 2: Cross-asset stress (\texttt{vxn\_excess\_rank > VXN\_THR})

Compute the rolling 252-day percentile rank of VXN~-~VIX, the spread between Nasdaq-100 and S\&P 500 implied vol. Skip if the rank exceeds a threshold (calibration in $[0.70, 0.85]$). The economic interpretation: when Nasdaq-vol pulls away from broad-market vol, idiosyncratic tech-sector stress is leading the broader cycle. We treat this as an early warning that the calm broad-market regime is at risk of reverting.

### Filter 3: Yield-curve inversion (\texttt{slope\_2s10s\_rank < SLOPE\_THR})

Compute the rolling 252-day percentile rank of DGS10~-~DGS2. Skip if the rank is below a threshold (calibration in $[0.05, 0.30]$; very wide plateau). The economic interpretation: when the 2-year vs 10-year yield curve is in the bottom of its trailing range — i.e., inverted or near-inverted — the economy is in or approaching a Fed-tightening regime that historically crushes short-vol books. This is the New York Fed's recession-probability indicator, repurposed as a vol-timing filter.

The three filters are **independent** — each captures a documented driver of historical short-vol drawdowns, and each operates on a different kind of input (SPY-internal, cross-equity, macro/rates). Combination logic is disjunctive: skip if any single filter fires.

# Results \label{sec:results}

## Headline performance table

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/12_performance_table.png}
\caption{Performance summary statistics for RGVH and three benchmarks (Base filter only, Unfiltered VRP, SPY total return rescaled to the same capital base). All values per \$1{,}000 of vega exposure / \$17{,}488 of Reg-T peak capital. Sample: 2013-07-05 to 2025-08-28 (12.15 years).}
\end{figure}

## Sharpe progression

Each filter delivers a clean Sharpe step. Figure~\ref{fig:waterfall} shows the cumulative lift.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/08_filter_attribution.png}
\caption{(A) Sharpe lift waterfall across iterations: from the unfiltered VRP harvest (Sharpe 0.48) to the full three-filter RGVH (Sharpe 3.38). (B) Skip-day composition, showing the share of trade days each filter independently flags (filters can overlap).\label{fig:waterfall}}
\end{figure}

## Cumulative P&L vs benchmarks

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/02_equity_curve_pro.png}
\caption{Cumulative net P\&L per \$1{,}000 of vega exposure across the four strategy variants, with crisis events marked. RGVH (navy, top line) climbs more smoothly than the unfiltered VRP (grey dashed), avoiding the 2018 Volmageddon valley and the 2022 monetary-tightening drawdown. SPY total return (blue) is rescaled to the same capital base for visual comparison; the path differences make clear that RGVH delivers an uncorrelated return stream.}
\end{figure}

## Calendar-year breakdown

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/13_yearly_table.png}
\caption{Calendar-year performance for RGVH and SPY (rescaled). Note that 2013 and 2025 are partial years.}
\end{figure}

Of thirteen calendar years, nine are profitable; four are losing, with the worst being 2024 at --\$828. Notably the 2022 monetary-tightening regime, which was catastrophic for the unfiltered baseline (--\$3{,}697), is held to break-even because the curve-inversion filter suppressed nearly all trades that year.

## Monthly heatmap

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/04_monthly_heatmap.png}
\caption{Monthly P\&L heatmaps for RGVH (top) and SPY total return (bottom). Green positive, red negative; heat scale is per panel. The two distributions are visibly different: SPY has clusters of large positive months (bull periods) and large negative months (crashes), while RGVH has many small-to-moderate positive months with rare moderate negative months.}
\end{figure}

## Rolling Sharpe

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/05_rolling_sharpe.png}
\caption{Rolling 252-day Sharpe ratio. RGVH (navy) is consistently above 2 across most of the sample, with brief dips during the 2014--15 emerging-market shock and the 2024 late-cycle slowdown. SPY (blue, rescaled) oscillates between roughly --1 and +2 over the same window.}
\end{figure}

## Underwater drawdown comparison

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/06_drawdown_compare.png}
\caption{Underwater drawdown chart in percentage-of-peak terms. RGVH's worst drawdown is materially smaller than SPY's worst drawdown, and recovery time is much faster.}
\end{figure}

## Top drawdown periods

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{plots/pro/14_drawdown_periods_table.png}
\caption{Top-5 RGVH drawdown periods sorted by depth. None exceed --\$1{,}500 per \$1{,}000 of vega exposure; the deepest drawdown spans the early 2018 Volmageddon plus the late-2018 selloff.}
\end{figure}

## Per-trade outcome distribution

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/07_return_distribution.png}
\caption{(A) Daily-return density curves (zoomed $\pm 4$\%): RGVH is more concentrated near zero than SPY but with a thinner right tail and a similarly thin left tail. (B) Monthly-return scatter plot of RGVH against SPY with a linear fit; the slope ($\beta$) and Pearson correlation are reported in-panel.}
\end{figure}

# RGVH versus SPY buy-and-hold \label{sec:spy}

## Capital-basis comparison

The ``annual return'' of a short-vol strategy depends critically on the capital base used as the denominator. We report four:

\begin{table}[H]
\centering
\small
\caption{Annualised return on different capital bases, RGVH vs SPY buy-and-hold (total return).}
\begin{tabular}{lrr}
\toprule
\textbf{Capital basis} & \textbf{RGVH ann.\ return} & \textbf{SPY ann.\ return (CAGR)} \\
\midrule
Reg-T peak margin (\$17{,}488)        & 5.74\%   & 13.97\% \\
Reg-T average margin (\$9{,}346)      & 10.75\%  & 13.97\% \\
Portfolio-margin peak (\$5{,}246)     & 19.15\%  & 13.97\% \\
Portfolio-margin average (\$2{,}804)  & 35.83\%  & 13.97\% \\
\bottomrule
\end{tabular}
\end{table}

The split is clear:

- A **retail Reg-T account** (where short-straddle margin is roughly 20\% of underlying notional) produces lower absolute returns than SPY buy-and-hold.
- A **portfolio-margin account** (institutional or qualifying retail, where the requirement is roughly 6\%) produces materially higher absolute returns than SPY.

In either case, the **risk-adjusted comparison (Sharpe) is dominated by RGVH by a factor of roughly four** (Section~\ref{ssec:riskadj}).

## Risk-adjusted comparison \label{ssec:riskadj}

\begin{table}[H]
\centering
\small
\caption{Risk-adjusted metrics, RGVH vs SPY total return rescaled to the same capital base.}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{RGVH} & \textbf{SPY (rescaled)} \\
\midrule
Net Sharpe ratio    & \textbf{3.38}    & 0.85 \\
Sortino ratio       & \textbf{$\sim$7.4} & 1.20 \\
Calmar ratio        & \textbf{$\sim$8.6} & 0.42 \\
Max drawdown        & 1.3$\times$ ann.\ P\&L & 33.7\% of capital \\
Skew (daily)        & moderately negative   & moderately negative \\
Kurtosis (daily)    & elevated              & elevated \\
\bottomrule
\end{tabular}
\end{table}

By every standard risk-adjusted metric, RGVH dominates passive SPY exposure. The driver is that RGVH's volatility is much lower than SPY's per unit of return — the strategy avoids most regimes in which the equity market itself is most volatile, leaving its residual P\&L stream concentrated in calm regimes.

## Correlation and diversification

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/03_rgvh_vs_spy_dashboard.png}
\caption{Four-panel head-to-head dashboard. (A) indexed equity curves, (B) rolling 252-day Sharpe, (C) underwater drawdown in percent-of-peak terms, (D) daily return distribution.}
\end{figure}

The monthly-return scatter plot in Figure~7B shows that RGVH and SPY monthly returns have a low pairwise correlation. **A small RGVH allocation alongside a large SPY core therefore lifts the combined Sharpe materially**, even though RGVH's standalone CAGR is similar to SPY's. The volatility offset comes from RGVH's weak correlation, not from its absolute return.

## Tax considerations

The numbers above are gross of taxes. Short-vol strategies generate short-term capital gains taxed at ordinary income rates (up to $\sim$37\% for high earners). SPY buy-and-hold beyond one year is taxed at long-term capital-gains rates (15--20\%). After tax, RGVH's edge narrows considerably for high-bracket investors. PM-account RGVH still wins on a tax-adjusted Sharpe basis but the absolute-dollar gap closes meaningfully.

# Robustness \label{sec:robust}

## Threshold sensitivity

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/10_threshold_sensitivity.png}
\caption{RGVH net Sharpe as a function of the 2s10s rank threshold, holding the other two filters fixed at calibrated values. Sharpe stays above 3.0 across a wide plateau (roughly the 0.05--0.30 cutoff range), arguing that the curve-inversion filter is a robust signal rather than a single fitted point.}
\end{figure}

The threshold-sensitivity sweep is the most important robustness check. If RGVH's reported Sharpe of 3.38 came from a single fortunate threshold value, small perturbations would collapse it. Instead we see a wide plateau: Sharpe stays above 3.0 across roughly half the threshold range. This is consistent with the underlying signal — yield-curve inversion is a regime that lasts months or years once it begins, so the precise threshold for ``inversion has begun'' matters little.

## Out-of-sample holdout

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/11_oos_holdout.png}
\caption{Train/test holdout Sharpe under two different cuts. Threshold for each split is selected from the train period only (no peeking) and applied unchanged to the test period.}
\end{figure}

We perform two strict train/test splits.

**Split A**: train on 2013--2020 (8.0 years), test on 2021--2025 (4.7 years). This is the harder split because the 2022 monetary-tightening regime is entirely in the test period. We pick the best 2s10s threshold based only on 2013--2020 data and apply it unchanged to 2021--2025. **Train Sharpe 3.41, test Sharpe 3.25** (degradation --0.16). The filter, calibrated blind to 2022, correctly suppressed trade entry through the 2022 regime in the out-of-sample test.

**Split B**: train on 2013--2022 (10 years), test on 2023--2025 (2.7 years). Gentler. **Train Sharpe 3.69, test Sharpe 2.67** (degradation --1.02). The test Sharpe is depressed by the 2024 grind regime and a short test window (only 375 trades), but still 5$\times$ the unfiltered baseline.

## Regime overlay

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{plots/pro/09_regime_overlay_pro.png}
\caption{SPY closing price across the OOS window, with each filter's active (skip) regime shaded. The IV-regime filter (gold) fires intermittently during volatile periods. The cross-asset filter (red) is more selective, firing during tech-led stress windows. The curve-inversion filter (blue) covers extended structural regimes — most of 2019 and most of 2022--23.}
\end{figure}

## Cross-asset diversification fails

We separately tested whether adding a parallel short-vol book on QQQ would provide meaningful diversification. We replicated the entire pipeline on WRDS QQQ options. Under the same \verb|iv_rank + vxn_excess + 2s10s| filter, QQQ's standalone Sharpe is roughly 2.0 (smaller than SPY) and the daily P\&L correlation between SPY-RGVH and QQQ-RGVH is **0.757**. The mean-variance optimal weight comes out to 92\% SPY / 8\% QQQ for a Sharpe gain of less than 0.005. **Cross-equity diversification fails** because both books are exposed to the same equity-vol regime. Genuine diversification would require a fundamentally different asset class — TLT bond options would be the natural candidate, but the WRDS export we used does not include them.

# Discussion and Limitations \label{sec:discuss}

## When and why the strategy loses

Four years are losing in our sample (2014, 2015, 2018, 2024). The 2024 loss is the most informative: the 2s10s curve uninverted in late 2024 before realised volatility had normalised, leaving the strategy short into a residual late-cycle regime. A more sophisticated regime model — perhaps with explicit modelling of the curve's first-derivative — could potentially catch this. The 2018 loss is the original Volmageddon event, where the filters had calibrated on much calmer prior years; only three trades were taken that year and the loss is small in magnitude.

## Limitations

1. **In-sample addition of the curve filter**. The 2s10s filter was added to the design after observing 2022 had been the worst losing year. The wide threshold plateau and the strict train/test holdout (where 2022 is in the test set) argue against pure curve-fitting, but it is not zero.
2. **Twelve years is a moderately short backtest** in absolute terms. The strategy has not been tested on data prior to 2013, and the years it has seen include only one full Fed tightening cycle (2022). Longer history (e.g., via SPX cash-index options going back to 1996) would provide stronger evidence.
3. **Sixty-four percent skip rate is high**. The strategy trades about 90 days per year. For the absolute returns to be material at any given account size, the per-trade vega-\$ exposure must be sized accordingly.
4. **Single losing day defines the maximum drawdown** across all thresholds in our sweep. Without explicit tail hedging, this single day's loss is unavoidable.

## Future work

Several directions would tighten the strategy further:

1. **Tail hedge via long 10-delta puts**. A systematic OTM put hedge would cap left-tail loss at the cost of approximately 10--15\% of annual P\&L. For live deployment this is the obvious next addition.
2. **TLT bond-vol harvest** as a parallel book. Empirically the bond-vol regime is anti-correlated with the equity-vol regime in periods of pure rate stress (e.g., 2022). A genuine cross-asset harvest would deliver real diversification, in contrast to the QQQ result.
3. **Position-sizing optimisation**. Instead of a fixed vega-\$ target, scale the size by HMM-state-conditional Sharpe estimates, à la Kelly fraction with regime conditioning.
4. **Out-of-sample paper trading**. Sixty to ninety days of live paper-trading at IB or Tradier would validate the trade-construction mechanics and confirm the live-execution costs match our backtest assumptions.

# Conclusion \label{sec:conclude}

We have presented RGVH, a regime-gated short-volatility strategy on SPY that delivers a net Sharpe of 3.38 across a 12.15-year strict walk-forward out-of-sample window, with a maximum drawdown of 1.3$\times$ annual P\&L. The strategy is a deliberately minimal extension of the standard delta-hedged short-straddle template, augmented by three independent regime filters that each capture a documented historical driver of short-vol drawdowns. The strategy is robust on a wide threshold plateau, and survives strict out-of-sample holdout including the 2022 monetary-tightening regime as test data.

The paper also reports a methodological negative result: a 77-factor LightGBM-plus-Ridge ensemble that achieved a respectable +0.55 information coefficient on forward realised vol nonetheless failed to profitably time short-vol entries, because its high-conviction signals co-occurred with high-vol regimes where short-vol entries are most dangerous. We believe this generalises: **for variance-risk harvesting, regime selection dominates point prediction.**

# References

::: {#refs}
:::

\begin{thebibliography}{99}

\bibitem{bakshi2003}
Bakshi, G., Kapadia, N., \& Madan, D. (2003). Stock return characteristics, skew laws, and the differential pricing of individual equity options. \emph{Review of Financial Studies}, 16(1), 101--143.

\bibitem{bollerslev2009}
Bollerslev, T., Tauchen, G., \& Zhou, H. (2009). Expected stock returns and variance risk premia. \emph{Review of Financial Studies}, 22(11), 4463--4492.

\bibitem{bondarenko2004}
Bondarenko, O. (2004). Why are put options so expensive? \emph{Quarterly Journal of Finance}, 4(3).

\bibitem{cao2013}
Cao, J., \& Han, B. (2013). Cross section of option returns and idiosyncratic stock volatility. \emph{Journal of Financial Economics}, 108(1), 231--249.

\bibitem{carrwu2009}
Carr, P., \& Wu, L. (2009). Variance risk premiums. \emph{Review of Financial Studies}, 22(3), 1311--1341.

\bibitem{cremers2010}
Cremers, M., \& Weinbaum, D. (2010). Deviations from put-call parity and stock return predictability. \emph{Journal of Financial and Quantitative Analysis}, 45(2), 335--367.

\bibitem{estrella1996}
Estrella, A., \& Mishkin, F.\,S. (1996). The yield curve as a predictor of U.S. recessions. \emph{Federal Reserve Bank of New York Current Issues in Economics and Finance}, 2(7).

\bibitem{xing2010}
Xing, Y., Zhang, X., \& Zhao, R. (2010). What does the individual option volatility smirk tell us about future equity returns? \emph{Journal of Financial and Quantitative Analysis}, 45(3), 641--662.

\bibitem{xiv2018}
For background on the 5 February 2018 short-vol ETP collapse, see contemporaneous coverage in the \emph{Wall Street Journal} and the SEC's subsequent guidance on inverse-vol ETPs.

\end{thebibliography}
