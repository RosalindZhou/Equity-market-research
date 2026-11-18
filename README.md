# Equity Market Research

A Python application for equity ranking, historical backtests, and paper trading. The demo uses Chinese A-share data. The current data adapters, stock universe, and trading assumptions are specific to A-shares.

## Components

| Area | Implementation |
|---|---|
| Ingestion | Fetches market bars from supported data providers and stores them in SQLite |
| Features | Calculates time-series and cross-sectional factors with Polars |
| Modeling | Trains and runs a LightGBM scoring model |
| Simulation | Runs a historical backtest and tracks a separate paper account in SQLite |
| Interface | Exposes CLI commands, an optional notification pipeline, and a local web console |

## Setup and usage

Activate a Python virtual environment from the repository root and install dependencies. Live data providers require network access.

```sh
python -m pip install -r requirements.txt
python main.py --fetch
python main.py --calc
python main.py --train
python main.py --predict
```

Use `python main.py --help` for available flags. Run `python main.py --web` for the local console at `http://127.0.0.1:5000`. The paper account status is available through `python main.py --paper --paper-action status`.

Notification settings are optional. Copy `.env.example` to a local `.env` and fill only the fields needed for the chosen notification service.

## Limitations

Paper trades are simulated. Backtest interpretation depends on the data provider, universe construction, timestamp alignment, and execution assumptions. Training and prediction currently calculate features separately. Check feature parity and ensure the forward label window does not cross the train/test boundary before reporting model or backtest results.

Regenerate existing feature caches with the calculation step. Training excludes rows whose forward labels depend on holdout dates. Run the offline split tests with `python -m unittest discover -s tests`.
