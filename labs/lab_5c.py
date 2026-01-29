"""Example code for generating a returns chart and table for MVO backtests."""

import datetime as dt

import polars as pl
import sf_quant.data as sfd
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "momentum"
gamma = 50

# Load MVO weights
weights = pl.read_parquet(f"weights/{signal_name}/{gamma}/*.parquet")

# Get returns
returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
    )
)

# Compute portfolio returns
portfolio_returns = (
    weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
    .sort("date")
)

# Compute cumulative log returns
cumulative_returns = portfolio_returns.select(
    "date", pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return")
)

# Plot cumulative log returns
plt.figure(figsize=(10, 6))
sns.lineplot(cumulative_returns, x="date", y="cumulative_return")
plt.title("Momentum Active MVO Backtest")
plt.xlabel("")
plt.ylabel("Cumulative Log Returns (%)")
plt.savefig("backtest_chart.png")

# Create summary table
summary = portfolio_returns.select(
    pl.col("return").mean().mul(252 * 100).alias("mean_return"),
    pl.col("return").std().mul(pl.lit(252).sqrt() * 100).alias("volatility"),
).with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))

# Print summary
print(summary)
