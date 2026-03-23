# %% [markdown]
# # Custom Historical Data — A Plain-English Guide
#
# Tutorial for [NautilusTrader](https://nautilustrader.io/docs/latest/) a high-performance
# algorithmic trading platform and event-driven backtester.
#
# [View source on GitHub](https://github.com/nautechsystems/nautilus_trader/blob/develop/docs/tutorials/custom_historical_data.py).

# %% [markdown]
# ## What is NautilusTrader in plain English?
#
# Think of NautilusTrader as a **very fast trading simulator and live-trading engine** built
# in two layers:
#
# | Layer | Language | Role |
# |-------|----------|------|
# | **Engine** (the "brain") | Rust | Handles numbers, orders, and timing at C-level speed |
# | **Control plane** (the "hands") | Python | Where you write your strategy logic |
#
# ### The core idea: an event loop
#
# NautilusTrader feeds market data — bars, ticks, order-book updates — into your strategy one
# event at a time, in strict time order. For each event the engine calls the matching handler
# on your strategy (`on_bar`, `on_quote_tick`, …). Your strategy reacts, maybe submits an
# order, and the engine moves on to the next event.
#
# ```
# [historical data]  ──►  [engine clock]  ──►  on_bar() / on_quote_tick()  ──►  orders
# ```
#
# Because every backtest uses the *same* event loop as live trading, your strategy behaves
# identically in research and production — no surprises when you go live.
#
# ### Key building blocks
#
# | Concept | What it is (plain English) |
# |---------|---------------------------|
# | **Instrument** | A description of *what* you are trading (e.g. EUR/USD, AAPL, BTC-USDT) |
# | **Bar** | A single OHLCV candle (open, high, low, close, volume) for a time period |
# | **QuoteTick** | A bid/ask pair at a point in time |
# | **TradeTick** | A single executed trade (price + quantity) |
# | **BarType** | A label that says *which* bars: instrument + period + price type (e.g. `EUR/USD.SIM-1-MINUTE-MID-EXTERNAL`) |
# | **Wrangler** | A helper that converts a pandas DataFrame into Nautilus objects |
# | **BacktestEngine** | Runs one backtest directly in memory — simple, great for experimenting |
# | **ParquetDataCatalog** | A folder of `.parquet` files that stores your data on disk for reuse |
# | **BacktestNode** | Runs one or more backtests using data from a catalog — the production path |
#
# ### Three operating modes
#
# ```
# Backtest  ──  historical data  +  simulated venue
# Sandbox   ──  live data        +  simulated venue
# Live      ──  live data        +  real venue
# ```
#
# The same strategy code runs unchanged in all three modes.

# %% [markdown]
# ## How to add your own historical data
#
# NautilusTrader reads data through **wranglers** — thin adapters that turn a pandas
# DataFrame (read from a CSV, a database, or computed in memory) into the exact Nautilus
# objects the engine expects.
#
# The typical workflow is:
#
# ```
# raw file / DataFrame
#       │
#       ▼
#   Wrangler.process(df)        ← normalises columns, adds nanosecond timestamps
#       │
#       ▼
#   list of Nautilus objects    (Bar / QuoteTick / TradeTick)
#       │
#       ├──► engine.add_data(objects)         (in-memory, BacktestEngine)
#       │
#       └──► catalog.write_data(objects)      (saved to disk, BacktestNode)
# ```
#
# The sections below show both paths for three common data shapes:
# - **OHLCV bars** (the most common)
# - **Quote ticks** (bid/ask pairs)
# - **Trade ticks** (individual trades)

# %% [markdown]
# ## Prerequisites
#
# - Python 3.12+ installed.
# - [NautilusTrader](https://pypi.org/project/nautilus_trader/) latest release installed
#   (`uv pip install nautilus_trader`).

# %% [markdown]
# ## Part 1 — OHLCV bar data (most common)
#
# ### Step 1 · Build or load a pandas DataFrame
#
# Your DataFrame must have:
# - A **DatetimeIndex** (UTC, timezone-aware or naive — both work).
# - Columns named **`open`**, **`high`**, **`low`**, **`close`** (required).
# - An optional **`volume`** column.
#
# Here we generate synthetic EUR/USD data so the tutorial runs without any downloads.

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5_000

price = 1.10 + np.cumsum(rng.normal(0, 0.0002, n))
spread = np.abs(rng.normal(0, 0.0003, n))

bars_df = pd.DataFrame(
    {
        "open": price,
        "high": price + spread,
        "low": price - spread,
        "close": price + rng.normal(0, 0.00005, n),
        "volume": rng.integers(100, 5_000, n).astype(float),
    },
    index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
)

# Make sure high >= open/close and low <= open/close
bars_df["high"] = bars_df[["open", "high", "close"]].max(axis=1)
bars_df["low"] = bars_df[["open", "low", "close"]].min(axis=1)

print(bars_df.head(3))

# %% [markdown]
# ### Step 2 · Define an Instrument
#
# An `Instrument` carries the contract specification: tick size, lot size, base currency, etc.
# For built-in FX pairs you can use `TestInstrumentProvider`; for real instruments see the
# [Instruments](../concepts/instruments.md) guide.

# %%
from nautilus_trader.test_kit.providers import TestInstrumentProvider

EURUSD = TestInstrumentProvider.default_fx_ccy("EUR/USD")
print(EURUSD)

# %% [markdown]
# ### Step 3 · Choose a BarType
#
# A `BarType` string encodes four pieces of information:
#
# ```
# {instrument_id}-{step}-{aggregation}-{price_type}-{aggregation_source}
#    EUR/USD.SIM  -  1  -   MINUTE    -    LAST    -      EXTERNAL
# ```
#
# - **step** — number of units per bar (1, 5, 15, …)
# - **aggregation** — SECOND, MINUTE, HOUR, DAY, TICK, VOLUME, …
# - **price_type** — BID, ASK, MID, LAST
# - **aggregation_source** — `EXTERNAL` for pre-built bars loaded from outside Nautilus;
#   `INTERNAL` for bars that Nautilus aggregates itself from ticks

# %%
from nautilus_trader.model.data import BarType

bar_type = BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL")
print(bar_type)

# %% [markdown]
# ### Step 4 · Wrangle the DataFrame into Nautilus Bar objects

# %%
from nautilus_trader.persistence.wranglers import BarDataWrangler

wrangler = BarDataWrangler(bar_type=bar_type, instrument=EURUSD)
bars = wrangler.process(bars_df)

print(f"Produced {len(bars):,} Bar objects")
print("First bar:", bars[0])

# %% [markdown]
# ### Step 5a · Run an in-memory backtest (BacktestEngine)
#
# `BacktestEngine` is the simplest path — no files are written anywhere.
# This is great for quick experiments and parameter sweeps.

# %%
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money

# 1. Create the engine
engine = BacktestEngine(
    config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR"),
    ),
)

# 2. Add a simulated trading venue
SIM = Venue("SIM")
engine.add_venue(
    venue=SIM,
    oms_type=OmsType.NETTING,
    account_type=AccountType.MARGIN,
    starting_balances=[Money(1_000_000, USD)],
    base_currency=USD,
    default_leverage=Decimal(1),
)

# 3. Register the instrument so the engine knows its specifications
engine.add_instrument(EURUSD)

# 4. Add your historical bars — this is where your custom data enters
engine.add_data(bars)

# %% [markdown]
# Now attach a strategy and run. We reuse the built-in `EMACross` example strategy.

# %%
from nautilus_trader.examples.strategies.ema_cross import EMACross
from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig

strategy = EMACross(
    EMACrossConfig(
        instrument_id=EURUSD.id,
        bar_type=bar_type,
        trade_size=Decimal(100_000),
        fast_ema_period=10,
        slow_ema_period=20,
    ),
)
engine.add_strategy(strategy)

engine.run()

print("\n=== Account report ===")
print(engine.trader.generate_account_report(SIM))

print("\n=== Positions report ===")
print(engine.trader.generate_positions_report())

engine.dispose()

# %% [markdown]
# ## Part 2 — Quote tick data (bid / ask)
#
# Quote ticks store the best bid and ask at every instant. They are the raw feed that many
# FX brokers provide.
#
# ### Expected DataFrame schema
#
# | Column | Meaning |
# |--------|---------|
# | `bid_price` | Best bid price |
# | `ask_price` | Best ask price |
# | `bid_size` | Bid quantity (optional, defaults to 0) |
# | `ask_size` | Ask quantity (optional, defaults to 0) |
# | index | DatetimeIndex (UTC) |

# %%
# Simulate a simple quote-tick DataFrame
n_ticks = 10_000
ts = pd.date_range("2024-01-01 08:00", periods=n_ticks, freq="100ms", tz="UTC")
mid = 1.10 + np.cumsum(rng.normal(0, 0.00005, n_ticks))
half_spread = np.abs(rng.normal(0.00010, 0.00002, n_ticks))

quotes_df = pd.DataFrame(
    {
        "bid_price": mid - half_spread,
        "ask_price": mid + half_spread,
        "bid_size": rng.integers(100_000, 2_000_000, n_ticks).astype(float),
        "ask_size": rng.integers(100_000, 2_000_000, n_ticks).astype(float),
    },
    index=ts,
)
print(quotes_df.head(3))

# %%
from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler

quote_wrangler = QuoteTickDataWrangler(instrument=EURUSD)
quote_ticks = quote_wrangler.process(quotes_df)

print(f"Produced {len(quote_ticks):,} QuoteTick objects")
print("First tick:", quote_ticks[0])

# %% [markdown]
# Add quote ticks to a `BacktestEngine` exactly the same way as bars:
#
# ```python
# engine.add_instrument(EURUSD)
# engine.add_data(quote_ticks)
# ```

# %% [markdown]
# ## Part 3 — Trade tick data (individual trades)
#
# Trade ticks record every individual transaction: price, quantity, and which side was the
# aggressor. This is the most granular data type.
#
# ### Expected DataFrame schema
#
# | Column | Meaning |
# |--------|---------|
# | `price` | Trade price |
# | `quantity` | Trade quantity |
# | `aggressor_side` | `"BUY"` or `"SELL"` (optional) |
# | `trade_id` | Unique trade identifier (optional) |
# | index | DatetimeIndex (UTC) |

# %%
# Simulate trade tick data
n_trades = 8_000
ts_trades = pd.date_range("2024-01-01 08:00", periods=n_trades, freq="200ms", tz="UTC")
trade_price = 1.10 + np.cumsum(rng.normal(0, 0.00003, n_trades))

trades_df = pd.DataFrame(
    {
        "price": trade_price,
        "quantity": rng.integers(1_000, 500_000, n_trades).astype(float),
        "aggressor_side": rng.choice(["BUY", "SELL"], n_trades),
        "trade_id": [f"T{i:08d}" for i in range(n_trades)],
    },
    index=ts_trades,
)
print(trades_df.head(3))

# %%
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler

trade_wrangler = TradeTickDataWrangler(instrument=EURUSD)
trade_ticks = trade_wrangler.process(trades_df)

print(f"Produced {len(trade_ticks):,} TradeTick objects")
print("First tick:", trade_ticks[0])

# %% [markdown]
# ## Part 4 — Saving to and reading from the Parquet catalog
#
# For larger datasets or multi-run workflows, save your data to a
# **`ParquetDataCatalog`** — a folder of Parquet files on your local disk.
# You then feed the catalog to a `BacktestNode` which reads only the date
# range you specify, without loading everything into RAM.

# %%
import shutil
from pathlib import Path

from nautilus_trader.persistence.catalog import ParquetDataCatalog

# Use a temporary folder so the tutorial stays self-contained
CATALOG_PATH = Path("/tmp/nt_tutorial_catalog")
if CATALOG_PATH.exists():
    shutil.rmtree(CATALOG_PATH)
CATALOG_PATH.mkdir(parents=True)

catalog = ParquetDataCatalog(CATALOG_PATH)

# Write the instrument first, then the data
catalog.write_data([EURUSD])
catalog.write_data(bars)
catalog.write_data(quote_ticks)

print("Instruments in catalog:", catalog.instruments())
print("Bar types in catalog :", catalog.bar_types())

# %% [markdown]
# ### Reading data back from the catalog

# %%
from nautilus_trader.core.datetime import dt_to_unix_nanos

start_ns = dt_to_unix_nanos(pd.Timestamp("2024-01-01", tz="UTC"))
end_ns   = dt_to_unix_nanos(pd.Timestamp("2024-01-02", tz="UTC"))

catalog_bars = catalog.bars(
    instrument_ids=[EURUSD.id.value],
    start=start_ns,
    end=end_ns,
)
print(f"Read back {len(catalog_bars):,} bars from catalog")

# %% [markdown]
# ### Running a BacktestNode against the catalog

# %%
from nautilus_trader.backtest.node import BacktestDataConfig
from nautilus_trader.backtest.node import BacktestEngineConfig as NodeBacktestEngineConfig
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.node import BacktestRunConfig
from nautilus_trader.backtest.node import BacktestVenueConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.model.data import Bar

venue_configs = [
    BacktestVenueConfig(
        name="SIM",
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USD",
        starting_balances=["1000000 USD"],
    ),
]

data_configs = [
    BacktestDataConfig(
        catalog_path=str(catalog.path),
        data_cls=Bar,
        bar_spec="1-MINUTE-LAST",
        instrument_id=EURUSD.id,
        start_time=start_ns,
        end_time=end_ns,
    ),
]

strategies = [
    ImportableStrategyConfig(
        strategy_path="nautilus_trader.examples.strategies.ema_cross:EMACross",
        config_path="nautilus_trader.examples.strategies.ema_cross:EMACrossConfig",
        config={
            "instrument_id": EURUSD.id,
            "bar_type": bar_type,
            "trade_size": Decimal("100000"),
            "fast_ema_period": 10,
            "slow_ema_period": 20,
        },
    ),
]

run_config = BacktestRunConfig(
    engine=NodeBacktestEngineConfig(strategies=strategies),
    data=data_configs,
    venues=venue_configs,
)

node = BacktestNode(configs=[run_config])
[result] = node.run()
print(result)

# %% [markdown]
# ## Part 5 — Loading data from a CSV file
#
# If your data is in a CSV file on disk, read it with `pandas.read_csv` and then
# hand the DataFrame to the appropriate wrangler. Below is a concise recipe for
# each data type.
#
# ### OHLCV bar CSV
#
# Expected CSV layout:
#
# ```
# timestamp,open,high,low,close,volume
# 2024-01-01 00:00:00+00:00,1.10010,1.10050,1.09980,1.10030,1500000
# 2024-01-01 00:01:00+00:00,1.10030,1.10070,1.10000,1.10060,1200000
# ...
# ```
#
# ```python
# import pandas as pd
# from nautilus_trader.persistence.wranglers import BarDataWrangler
# from nautilus_trader.model.data import BarType
# from nautilus_trader.test_kit.providers import TestInstrumentProvider
#
# EURUSD = TestInstrumentProvider.default_fx_ccy("EUR/USD")
# bar_type = BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL")
#
# df = pd.read_csv(
#     "eurusd_1m.csv",
#     index_col="timestamp",
#     parse_dates=True,
# )
# df.index = df.index.tz_localize("UTC")   # add UTC timezone if not already present
#
# bars = BarDataWrangler(bar_type=bar_type, instrument=EURUSD).process(df)
# ```
#
# ### Quote tick CSV (bid / ask)
#
# Expected CSV layout:
#
# ```
# timestamp,bid_price,ask_price,bid_size,ask_size
# 2024-01-01 08:00:00.000+00:00,1.09995,1.10005,1000000,800000
# ...
# ```
#
# ```python
# from nautilus_trader.persistence.wranglers import QuoteTickDataWrangler
#
# df = pd.read_csv("eurusd_quotes.csv", index_col="timestamp", parse_dates=True)
# df.index = df.index.tz_localize("UTC")
#
# quote_ticks = QuoteTickDataWrangler(instrument=EURUSD).process(df)
# ```
#
# ### Trade tick CSV
#
# Expected CSV layout:
#
# ```
# timestamp,price,quantity,aggressor_side,trade_id
# 2024-01-01 08:00:00.100+00:00,1.10001,500000,BUY,T00000001
# ...
# ```
#
# ```python
# from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
#
# df = pd.read_csv("eurusd_trades.csv", index_col="timestamp", parse_dates=True)
# df.index = df.index.tz_localize("UTC")
#
# trade_ticks = TradeTickDataWrangler(instrument=EURUSD).process(df)
# ```

# %% [markdown]
# ## Quick-reference cheat sheet
#
# | I have… | Wrangler to use | Key columns needed |
# |---------|-----------------|-------------------|
# | OHLCV candles | `BarDataWrangler` | `open`, `high`, `low`, `close` (+ optional `volume`) |
# | Bid/ask pairs | `QuoteTickDataWrangler` | `bid_price`, `ask_price` (+ optional `bid_size`, `ask_size`) |
# | Individual trades | `TradeTickDataWrangler` | `price`, `quantity` (+ optional `aggressor_side`, `trade_id`) |
#
# **Common pitfalls**
#
# - The DataFrame index **must** be a `DatetimeIndex`.  Timestamps without a timezone are
#   treated as UTC; timezone-aware timestamps are converted to UTC automatically.
# - Column names are **case-sensitive**.  Use lowercase (`open`, not `Open`).
# - The `BarType` string must reference the **same venue name** you pass to
#   `engine.add_venue()` (e.g. `EUR/USD.SIM` and `Venue("SIM")`).
# - Add the **instrument before the data**: `engine.add_instrument(...)` must come before
#   `engine.add_data(...)`.

# %% [markdown]
# ## Next steps
#
# - [Quickstart](../getting_started/quickstart) — run your first backtest in five minutes.
# - [Backtest (low-level API)](../getting_started/backtest_low_level) — more control over
#   engine configuration, execution algorithms, and repeated runs.
# - [Backtest (high-level API)](../getting_started/backtest_high_level) — catalog-driven
#   workflows suited to production.
# - [Loading External Data](loading_external_data) — loading real-world FX data from
#   histdata.com.
# - [Data](../concepts/data) — full reference for all supported data types and the
#   data-processing pipeline.
# - [Custom Data](../concepts/custom_data) — defining entirely new data types and routing
#   them through the engine.

# %%
# Clean up the temporary catalog
shutil.rmtree(CATALOG_PATH, ignore_errors=True)
