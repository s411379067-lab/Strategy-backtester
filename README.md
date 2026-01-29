# Crypto Strategy Backtest

A minimal backtesting framework for crypto strategies.

## Features
- Strategy-driven intents (entry/exit/tp/sl/be)
- Conservative intrabar exits using OHLC
- Portfolio bookkeeping and equity curve
- Pluggable indicators via `IndicatorRegistry`

## Project Layout
- backtester/: Core engine, models, strategy base, indicators, execution and portfolio
- tests/: Unit tests for engine and execution
- docs/: Space for guides and design notes

## Quick Start
1. Prepare a pandas DataFrame with DatetimeIndex and columns: `open, high, low, close`.
2. Implement a strategy in `backtester/strategies/` extending `Strategy`.
3. Run the engine with your DataFrame and strategy.

## Development
Run tests:

```powershell
# From repository root
pytest -q
```

## License
TBD