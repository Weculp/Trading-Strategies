"""
Configuration for RGVH backtest.

----- THRESHOLDS DELIBERATELY REDACTED -----

The three filter thresholds (IV_THR, VXN_THR, SLOPE_THR) are intentionally not
disclosed in this open-source release. The methodology is fully documented;
when you re-run the backtest on your own option-chain data, you'll find your
own optimum. Our work shows a wide plateau of acceptable values (Sharpe > 3.0
across roughly half the threshold space) so the result is not knife-edge.

If you replicate this and find your numbers — congrats, you've earned the
edge. If you'd like to compare notes, file an issue on the GitHub repo.
"""

# === Strategy parameters ===
TARGET_DTE      = 22         # business days to expiry at entry, ATM
DTE_MIN         = 17         # acceptable window for "near-22-DTE"
DTE_MAX         = 30
HOLD_DAYS       = 15         # business-day hold period
TARGET_VEGA_USD = 1000.0     # vega-$ exposure target per trade

# === Hedging ===
HEDGE_DRIFT_THR = 0.05       # rebalance when |delta drift| > 5% of straddle delta

# === Costs ===
COMMISSION_PER_LEG_USD = 0.65    # retail-broker default
SPY_HEDGE_SLIP_PER_SHARE = 0.005 # half-cent slippage on SPY shares
CONTRACT_MULTIPLIER = 100        # standard equity option

# === Filter thresholds — REDACTED ===
# Replace with your own values after re-running on your data.
# Their order of magnitude is documented in the research paper:
#   IV_THR    in roughly  [0.55, 0.85]
#   VXN_THR   in roughly  [0.70, 0.85]
#   SLOPE_THR in roughly  [0.05, 0.30]   (Sharpe > 3.0 plateau)

IV_THR    = float("nan")  # SPY iv_rank_252 cutoff (skip if rank > IV_THR)
VXN_THR   = float("nan")  # vxn_excess_rank_252 cutoff (skip if rank > VXN_THR)
SLOPE_THR = float("nan")  # 2s10s_rank_252 cutoff (skip if rank < SLOPE_THR)

# === Risk-free rate / dividend yield assumptions ===
DEFAULT_R = 0.03   # used as fallback if Treasury rate file unavailable
DEFAULT_Q = 0.015  # SPY dividend yield (~1.5%)

# === Walk-forward CV ===
N_SPLITS = 5
MIN_TRAIN_FRAC = 0.40

# === Rolling-rank window (252 = ~1 trading year) ===
RANK_LOOKBACK = 252
