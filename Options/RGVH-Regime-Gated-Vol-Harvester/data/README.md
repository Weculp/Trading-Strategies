# Data Sourcing

This strategy is backtested on three datasets. The largest is WRDS / OptionMetrics, which is licensed and **not redistributed**. The other two are free.

## 1. SPY EOD options chains (proprietary — not redistributed)

**Source**: [WRDS OptionMetrics IvyDB](https://wrds-www.wharton.upenn.edu/) — Phase_4 / Options_SPY in our internal export.

**Coverage required**: 2005-01 → present, daily EOD chains for SPY.

**Required columns**:
```
secid, date, exdate, cp_flag, strike_price, best_bid, best_offer,
volume, open_interest, impl_volatility, delta, gamma, vega, theta,
optionid, close (underlying), volume_underlying
```

**To reproduce**:
1. Subscribe to WRDS (institutional access; some universities provide it free)
2. Pull the `OptionMetrics → Standardized Options` tables for `secid` matching SPY
3. Save yearly parquet files as `spy_options_YYYY.parquet`
4. Place in `data/raw/spy/`

If you don't have WRDS, [ThetaData](https://www.thetadata.net/) (~$80/mo) and [Polygon.io](https://polygon.io/) (~$200/mo) sell similar data. Coverage may differ.

## 2. Treasury rate curve (free)

**Source**: [FRED](https://fred.stlouisfed.org/) via Python's `pandas_datareader` or via the WRDS Treasury_Rates table.

**Required series**:
- `DGS2` (2-year Treasury yield)
- `DGS10` (10-year Treasury yield)
- `DGS3MO` (3-month T-bill yield, used as risk-free rate)
- `BAMLH0A0HYM2` (high-yield credit spread, optional)
- `TEDRATE` (TED spread, optional)

**To reproduce**:
```python
import pandas_datareader.data as web
import pandas as pd
start = "2005-01-01"
series = ["DGS2","DGS10","DGS3MO","BAMLH0A0HYM2"]
df = web.DataReader(series, "fred", start)
df.to_parquet("data/raw/treasury_rates.parquet")
```

## 3. MOVE & VIX indices (free)

**Source**: Yahoo Finance via `yfinance`.

```python
import yfinance as yf
move = yf.Ticker("^MOVE").history(period="max")["Close"]
vix  = yf.Ticker("^VIX").history(period="max")["Close"]
vxn  = yf.Ticker("^VXN").history(period="max")["Close"]
```

**Coverage**:
- VIX: 1990-present
- VXN: 2001-present
- MOVE: 2002-present

VXN is critical — the cross-asset filter `vxn_excess_rank` uses VXN−VIX spread.

## Data integrity notes

- WRDS OptionMetrics has occasional duplicate `(date, expir, strike)` rows from multi-root quirks (SPY vs SPYW weeklys at same key). Dedup keeping the row with highest combined OI — see `src/data_pipeline.py`.
- WRDS `strike_price` is in **dollars × 1000** (a $400 strike is stored as `400000.0`). Divide by 1000.
- Greeks from OptionMetrics are computed against their proprietary IV. We use them directly rather than back out our own.
- A small number of rows (<0.01%) have numerical-instability greek values (gamma > 1, vega > 500). Filter before aggregation.

## File layout expected

```
data/
├── README.md                            (this file)
└── raw/                                 (gitignored)
    ├── spy/
    │   ├── spy_options_2005.parquet
    │   ├── spy_options_2006.parquet
    │   └── ...
    ├── treasury_rates.parquet
    └── vix_move.parquet
```
