# Saudi & U.S. Market Educational Signal Generator

> **Important:** This repository is for educational exploration only. It does **not** execute trades and should not be treated as financial advice.

## Overview

This project demonstrates an end-to-end workflow for generating daily buy/sell/hold signals for a small basket of Saudi and U.S. equities.  The system downloads historical prices via `yfinance`, engineers several technical indicators, trains a logistic regression classifier with walk-forward validation, and posts the resulting signals to Discord for study purposes.  Everything runs inside Docker so the scheduler can remain online continuously.

The schedule aligns with official market hours: the Saudi Stock Exchange (Tadawul) trades Sunday through Thursday from 10:00 AM to 3:10 PM GMT+3,【oaicite:9†】 while the NYSE and Nasdaq operate Monday through Friday from 9:30 AM to 4:00 PM Eastern Time.【oaicite:8†】  The container assumes the `America/Denver` time zone and converts the exchange opens (1:00 AM local for Tadawul and 7:30 AM local for the U.S. markets when daylight saving time is active) using `pytz`.

## Features

* Historical data collection via `yfinance` with optional Brent oil returns (`BZ=F`) as a macro indicator.
* Technical feature engineering: 1-day and 5-day returns, 20-day volatility, 10/50-day SMA gap, RSI(14), MACD and signal line, and MACD histogram.
* Logistic regression wrapped in a `scikit-learn` pipeline with `StandardScaler` and walk-forward (`TimeSeriesSplit`) validation.
* Probability-to-signal mapping with configurable thresholds (default: buy ≥ 0.55, sell ≤ 0.45) and position sizing that scales a notional capital pool by the distance from a neutral 0.50 probability.
* Discord webhook notifications summarizing probabilities, suggested educational position sizes, and recent closing prices.
* Minute-level polling of the market open windows using the `schedule` library so daylight saving transitions are handled through timezone-aware calculations.

## Repository Layout

```
.
├── Dockerfile
├── predictor.py
├── requirements.txt
└── README.md
```

## Configuration

Environment variable | Purpose | Default
--- | --- | ---
`DISCORD_WEBHOOK_URL` | Discord webhook to receive the daily summary | *(required for notifications)*
`SA_TICKERS` | Comma-separated Tadawul tickers (e.g., `2222.SR,2010.SR,1180.SR`) | `2222.SR,2010.SR,1180.SR`
`US_TICKERS` | Comma-separated U.S. tickers (e.g., `AAPL,MSFT,SPY`) | `AAPL,MSFT,SPY`
`BASE_CAPITAL` | Educational capital base used to scale suggested position sizes | `1000`
`BUY_THRESHOLD` | Probability threshold at/above which the system issues a “buy” | `0.55`
`SELL_THRESHOLD` | Probability threshold at/below which the system issues a “sell” | `0.45`
`RUN_ON_START` | If set to `1`, executes an immediate run for both markets on startup | `0`

All other behavior (feature engineering, thresholds, tickers) can be customized inside `predictor.py`.  Docstrings in the script highlight where to add or remove indicators.

## Building and Running with Docker

1. Build the image:

   ```bash
   docker build -t saudi-us-predictor .
   ```

2. Run the container, providing your Discord webhook:

   ```bash
   docker run \
     -e DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-id" \
     -e SA_TICKERS="2222.SR,2010.SR" \
     -e US_TICKERS="AAPL,MSFT" \
     saudi-us-predictor
   ```

   The container starts the scheduler and checks every minute whether the Saudi or U.S. market has just opened.  When an open window is detected on a valid trading day, the system trains models for each configured ticker and posts a consolidated Discord message.

3. (Optional) Force an immediate prediction cycle on launch:

   ```bash
   docker run \
     -e DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-id" \
     -e RUN_ON_START=1 \
     saudi-us-predictor
   ```

## How the Signals Are Generated

1. **Data fetch:** Up to five years of daily bars are downloaded for every ticker and for Brent oil futures.
2. **Feature engineering:** Technical indicators and rolling statistics are computed, and tomorrow’s directional move becomes the classification target.
3. **Walk-forward validation:** `TimeSeriesSplit` simulates out-of-sample testing; scores are logged for reference.
4. **Model fit & prediction:** A logistic regression pipeline is trained on all available data, and the latest feature vector produces a probability that the next close will exceed the current close.
5. **Signal translation:** Probabilities above the buy threshold become “buy,” below the sell threshold become “sell,” and everything else is “hold.”  Suggested position size = `BASE_CAPITAL * abs(probability - 0.5) * 2`.
6. **Notification:** The Discord summary reiterates the educational nature of the system and shares probabilities, signals, and notional sizing.

## Extending the Project

* Add or remove indicators inside `engineer_features` in `predictor.py`.
* Swap in gradient boosting or another classifier within the pipeline if you want to experiment with different models.
* Attach persistent storage or caching if you plan to analyze historical predictions.
* Integrate additional macro series by merging them in the feature engineering step.

## Educational Disclaimer

This tool is provided strictly for learning about data pipelines, feature engineering, and basic machine learning workflows for financial time series.  It does **not** recommend or execute trades, and the authors are not registered investment advisors.  Always perform your own due diligence before committing capital.

## References

* Saudi Stock Exchange (Tadawul) official trading hours.【oaicite:9†】
* NYSE & Nasdaq regular trading session hours.【oaicite:8†】
