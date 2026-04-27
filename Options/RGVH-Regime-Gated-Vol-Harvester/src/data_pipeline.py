"""
Transform WRDS / OptionMetrics yearly SPY-options parquet files into a single
unified wide-format parquet suitable for the rest of the pipeline.

Inputs:
    data/raw/spy/spy_options_YYYY.parquet   for YYYY in 2005..present

Output:
    data/processed/spy_eod_unified.parquet

WRDS schema (input, long format — one row per option leg):
    secid, date, exdate, cp_flag, strike_price (=$ x 1000),
    best_bid, best_offer, volume, open_interest,
    impl_volatility, delta, gamma, vega, theta, optionid,
    close (underlying)

Unified schema (output, wide — one row per (date, expir, strike), call+put together):
    tradeDate, expirDate, dte, strike, spotPrice, stockPrice, T (= years),
    delta (call-side), gamma, vega, theta,
    callValue, callBidPrice, callAskPrice, callOpenInterest, callVolume,
    putValue,  putBidPrice,  putAskPrice,  putOpenInterest,  putVolume,
    iv_call, iv_put, iv_mid,
    optionid_call, optionid_put

Notes
-----
* `strike_price` from WRDS is in dollars × 1000 (a $400 strike is `400000.0`).
* The same `(date, expir, strike)` can appear with multiple `root` values when SPY
  has weekly/monthly contracts at the same key. We dedupe by keeping the row with
  highest combined OI (call + put).
* OptionMetrics greeks are computed against their proprietary IV; we preserve them
  unchanged rather than back out our own.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl


def transform_spy_wrds_to_unified(
    raw_dir: Path | str,
    output_path: Path | str,
) -> int:
    """
    Read all `spy_options_YYYY.parquet` files in `raw_dir`, pivot calls/puts
    into a wide format, dedupe, and write `output_path`.

    Returns the number of rows written.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("spy_options_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No SPY option files found in {raw_dir}")

    print(f"Concat {len(files)} yearly files ...")
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
    print(f"  raw rows: {df.height:,}")

    # Type coercion + strike normalisation
    df = df.with_columns([
        pl.col("date").str.to_date("%Y-%m-%d").alias("tradeDate"),
        pl.col("exdate").str.to_date("%Y-%m-%d").alias("expirDate"),
        (pl.col("strike_price") / 1000).alias("strike"),
        pl.col("close").alias("spotPrice"),
    ]).drop(["date", "exdate", "strike_price", "close"])

    calls = df.filter(pl.col("cp_flag") == "C").select([
        "tradeDate", "expirDate", "strike", "spotPrice",
        pl.col("best_bid").alias("callBidPrice"),
        pl.col("best_offer").alias("callAskPrice"),
        pl.col("volume").alias("callVolume"),
        pl.col("open_interest").alias("callOpenInterest"),
        pl.col("impl_volatility").alias("iv_call"),
        pl.col("delta").alias("delta"),
        pl.col("gamma").alias("gamma"),
        pl.col("vega").alias("vega"),
        pl.col("theta").alias("theta"),
        pl.col("optionid").alias("optionid_call"),
    ])
    puts = df.filter(pl.col("cp_flag") == "P").select([
        "tradeDate", "expirDate", "strike",
        pl.col("best_bid").alias("putBidPrice"),
        pl.col("best_offer").alias("putAskPrice"),
        pl.col("volume").alias("putVolume"),
        pl.col("open_interest").alias("putOpenInterest"),
        pl.col("impl_volatility").alias("iv_put"),
        pl.col("delta").alias("delta_put"),
        pl.col("gamma").alias("gamma_put"),
        pl.col("vega").alias("vega_put"),
        pl.col("theta").alias("theta_put"),
        pl.col("optionid").alias("optionid_put"),
    ])

    wide = calls.join(puts, on=["tradeDate", "expirDate", "strike"],
                      how="full", coalesce=True)

    # If only put-side present, recover call-delta via put-call parity approx
    wide = wide.with_columns([
        pl.when(pl.col("delta").is_null() & pl.col("delta_put").is_not_null())
          .then(pl.col("delta_put") + 1.0)
          .otherwise(pl.col("delta")).alias("delta"),
        pl.coalesce([pl.col("gamma"), pl.col("gamma_put")]).alias("gamma"),
        pl.coalesce([pl.col("vega"),  pl.col("vega_put")]).alias("vega"),
        pl.coalesce([pl.col("theta"), pl.col("theta_put")]).alias("theta"),
    ]).drop(["delta_put", "gamma_put", "vega_put", "theta_put"])

    # Derived columns
    wide = wide.with_columns([
        ((pl.col("expirDate") - pl.col("tradeDate")).dt.total_days() + 1).alias("dte"),
        ((pl.col("expirDate") - pl.col("tradeDate")).dt.total_days() / 365.25).alias("T"),
        pl.col("spotPrice").alias("stockPrice"),
        ((pl.col("callBidPrice") + pl.col("callAskPrice")) / 2).alias("callValue"),
        ((pl.col("putBidPrice")  + pl.col("putAskPrice"))  / 2).alias("putValue"),
        pl.when(pl.col("iv_call").is_not_null() & pl.col("iv_put").is_not_null())
          .then((pl.col("iv_call") + pl.col("iv_put")) / 2)
          .otherwise(pl.coalesce([pl.col("iv_call"), pl.col("iv_put")])).alias("iv_mid"),
    ])

    # Dedupe (date,expir,strike) keeping max-OI row
    wide = wide.with_columns(
        (pl.col("callOpenInterest").fill_null(0) + pl.col("putOpenInterest").fill_null(0))
            .alias("_totoi")
    ).sort(
        ["tradeDate", "expirDate", "strike", "_totoi"],
        descending=[False, False, False, True],
    )
    wide = wide.unique(subset=["tradeDate", "expirDate", "strike"], keep="first").drop("_totoi")

    final_cols = [
        "tradeDate", "spotPrice", "stockPrice", "expirDate", "dte", "strike", "T",
        "delta", "gamma", "vega", "theta",
        "callValue", "callBidPrice", "callAskPrice", "callOpenInterest", "callVolume",
        "putValue",  "putBidPrice",  "putAskPrice",  "putOpenInterest",  "putVolume",
        "iv_call", "iv_put", "iv_mid",
        "optionid_call", "optionid_put",
    ]
    final_cols = [c for c in final_cols if c in wide.columns]
    wide = wide.select(final_cols).sort(["tradeDate", "expirDate", "strike"])

    wide.write_parquet(output_path, compression="zstd", compression_level=3)
    print(f"  wrote {wide.height:,} rows -> {output_path}  ({output_path.stat().st_size / 1e9:.2f} GB)")
    return wide.height


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw",  default="data/raw/spy",
                   help="dir containing spy_options_YYYY.parquet")
    p.add_argument("--out",  default="data/processed/spy_eod_unified.parquet",
                   help="output unified parquet")
    args = p.parse_args()
    transform_spy_wrds_to_unified(args.raw, args.out)
