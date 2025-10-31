"""Educational stock signal generator for Saudi and U.S. markets.

This module downloads historical data, engineers features, trains a logistic
regression classifier with a walk-forward style validation, and produces daily
buy/sell/hold signals.  The script is designed to be run inside Docker with the
`schedule` library polling every minute to check if either market has opened.

Configuration is done through environment variables:

- ``DISCORD_WEBHOOK_URL``: Discord webhook for notifications (required to post).
- ``SA_TICKERS``: Comma-separated tickers for the Saudi market (default ``2222.SR,2010.SR``).
- ``US_TICKERS``: Comma-separated tickers for the U.S. market (default ``AAPL,MSFT``).
- ``BASE_CAPITAL``: Notional capital used when scaling the suggested position
  sizes (default ``1000``).
- ``BUY_THRESHOLD`` / ``SELL_THRESHOLD``: Probability thresholds to classify the
  signals (default ``0.55`` / ``0.45``).
- ``RUN_ON_START``: If set to ``1``, the predictor will issue signals
  immediately on startup for both markets, in addition to the scheduled runs.

The script is intentionally educational: it does not place trades and the
Discord notifications clearly state that it is for learning purposes only.

To extend functionality, edit ``DEFAULT_US_TICKERS`` and ``DEFAULT_SA_TICKERS``
or pass your own via environment variables.  Thresholds can be adjusted either
in ``ProbabilityConfig`` or by environment variables.  Feature engineering can be
expanded inside :func:`engineer_features`.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import schedule
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Default ticker selections for each market.  These can be overridden via
# environment variables.
DEFAULT_US_TICKERS = ("AAPL", "MSFT", "SPY")
DEFAULT_SA_TICKERS = ("2222.SR", "2010.SR", "1180.SR")

# Constants for the exchange opening times.
SAUDI_MARKET_TZ = "Asia/Riyadh"
US_MARKET_TZ = "America/New_York"
LOCAL_TZ = "America/Denver"
SAUDI_OPEN_TIME = dtime(hour=10, minute=0)  # 10:00 AM GMT+3
US_OPEN_TIME = dtime(hour=9, minute=30)  # 9:30 AM ET

# Weekday indices for trading sessions (Python's Monday=0 ... Sunday=6).
SAUDI_TRADING_DAYS = {6, 0, 1, 2, 3}  # Sunday-Thursday
US_TRADING_DAYS = {0, 1, 2, 3, 4}  # Monday-Friday

# Window after market open (in minutes) in which the job will execute.
RUN_WINDOW_MINUTES = 15


@dataclass
class ProbabilityConfig:
    """Configuration for converting probabilities into signals."""

    buy_threshold: float = 0.55
    sell_threshold: float = 0.45

    def determine_signal(self, probability: float) -> str:
        """Return "buy", "sell", or "hold" based on configured thresholds."""
        if probability >= self.buy_threshold:
            return "buy"
        if probability <= self.sell_threshold:
            return "sell"
        return "hold"


@dataclass
class MarketConfig:
    """Per-market configuration and runtime state."""

    name: str
    exchange_timezone: str
    open_time: dtime
    trading_days: set
    tickers: Tuple[str, ...]
    probability_config: ProbabilityConfig
    base_capital: float
    last_run_date: Optional[datetime.date] = None

    def __post_init__(self) -> None:
        # Ensure tickers are uppercase and stripped.
        clean_tickers = tuple(ticker.strip().upper() for ticker in self.tickers if ticker.strip())
        if not clean_tickers:
            raise ValueError(f"No valid tickers configured for {self.name} market.")
        object.__setattr__(self, "tickers", clean_tickers)


def setup_logging() -> None:
    """Configure logging format for clarity inside Docker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_env_list(name: str, default: Iterable[str]) -> Tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of tickers."""
    value = os.getenv(name)
    if not value:
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_probability_config() -> ProbabilityConfig:
    """Load probability thresholds from the environment."""
    buy_threshold = float(os.getenv("BUY_THRESHOLD", "0.55"))
    sell_threshold = float(os.getenv("SELL_THRESHOLD", "0.45"))
    if buy_threshold <= sell_threshold:
        logging.warning(
            "BUY_THRESHOLD (%.2f) should be greater than SELL_THRESHOLD (%.2f)."
            " Reverting to defaults.",
            buy_threshold,
            sell_threshold,
        )
        return ProbabilityConfig()
    return ProbabilityConfig(buy_threshold=buy_threshold, sell_threshold=sell_threshold)


def load_market_configs() -> List[MarketConfig]:
    """Create MarketConfig instances for both Saudi and U.S. markets."""
    probability_config = get_probability_config()
    base_capital = float(os.getenv("BASE_CAPITAL", "1000"))

    saudi_config = MarketConfig(
        name="Saudi",
        exchange_timezone=SAUDI_MARKET_TZ,
        open_time=SAUDI_OPEN_TIME,
        trading_days=SAUDI_TRADING_DAYS,
        tickers=parse_env_list("SA_TICKERS", DEFAULT_SA_TICKERS),
        probability_config=probability_config,
        base_capital=base_capital,
    )

    us_config = MarketConfig(
        name="U.S.",
        exchange_timezone=US_MARKET_TZ,
        open_time=US_OPEN_TIME,
        trading_days=US_TRADING_DAYS,
        tickers=parse_env_list("US_TICKERS", DEFAULT_US_TICKERS),
        probability_config=probability_config,
        base_capital=base_capital,
    )
    return [saudi_config, us_config]


def fetch_price_history(ticker: str, start: datetime) -> pd.DataFrame:
    """Fetch historical price data using yfinance for a single ticker."""
    logging.info("Downloading data for %s", ticker)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        raise ValueError(f"No historical data returned for {ticker}.")
    df = df.rename(columns=str.lower)
    if "adj close" not in df.columns:
        raise ValueError(f"Adjusted close not found in data for {ticker}.")
    df = df[["adj close", "close", "volume"]].rename(columns={"adj close": "adj_close"})
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(price_df: pd.DataFrame, brent_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    """Create technical indicators and target variable for modeling."""
    df = price_df.copy()
    close = df["adj_close"]

    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    df["sma_10"] = sma_10
    df["sma_50"] = sma_50
    df["sma_gap"] = (sma_10 - sma_50) / sma_50

    df["rsi_14"] = compute_rsi(close, period=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    if brent_returns is not None:
        df["brent_return"] = brent_returns.reindex(df.index)

    df["target"] = (close.shift(-1) > close).astype(int)

    feature_cols = [
        "return_1d",
        "return_5d",
        "volatility_20d",
        "sma_gap",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    if "brent_return" in df.columns:
        feature_cols.append("brent_return")

    df = df.dropna(subset=feature_cols + ["target"])  # remove rows with missing features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_cols + ["target"], inplace=True)

    return df[feature_cols + ["target", "adj_close"]]


def time_series_cross_validation(
    features: pd.DataFrame, target: pd.Series, pipeline: Pipeline
) -> Tuple[Pipeline, float]:
    """Perform walk-forward cross-validation and refit the model on full data."""
    tscv = TimeSeriesSplit(n_splits=min(5, len(features) // 50 or 1))
    scores: List[float] = []
    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        model = Pipeline(pipeline.steps)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
    avg_score = float(np.mean(scores)) if scores else float("nan")
    pipeline.fit(features, target)
    return pipeline, avg_score


def train_and_predict(
    ticker: str,
    market_config: MarketConfig,
    brent_returns: Optional[pd.Series],
    lookback_years: int = 5,
) -> Optional[Dict[str, object]]:
    """Train a logistic regression model for the ticker and return the latest signal."""
    start_date = datetime.now(tz=pytz.UTC) - timedelta(days=lookback_years * 365)
    try:
        price_history = fetch_price_history(ticker, start=start_date)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to download data for %s: %s", ticker, exc)
        return None

    dataset = engineer_features(price_history, brent_returns=brent_returns)
    if dataset.empty:
        logging.warning("Not enough data after feature engineering for %s", ticker)
        return None

    features = dataset.drop(columns=["target", "adj_close"])
    target = dataset["target"]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    try:
        trained_model, cv_score = time_series_cross_validation(features, target, pipeline)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Model training failed for %s: %s", ticker, exc)
        return None

    latest_features = features.iloc[[-1]]
    probability = float(trained_model.predict_proba(latest_features)[0, 1])
    signal = market_config.probability_config.determine_signal(probability)

    distance_from_neutral = abs(probability - 0.5) * 2  # range [0, 1]
    suggested_size = market_config.base_capital * distance_from_neutral

    return {
        "ticker": ticker,
        "probability": probability,
        "signal": signal,
        "cv_score": cv_score,
        "suggested_size": suggested_size,
        "latest_close": float(dataset["adj_close"].iloc[-1]),
    }


def fetch_brent_returns(start: datetime) -> Optional[pd.Series]:
    """Download Brent oil prices and compute daily returns for optional features."""
    try:
        df = yf.download("BZ=F", start=start.strftime("%Y-%m-%d"), progress=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to download Brent oil data: %s", exc)
        return None
    if df.empty:
        logging.warning("Brent oil data is empty; continuing without it.")
        return None
    df = df.rename(columns=str.lower)
    if "adj close" not in df.columns:
        return None
    series = df["adj close"].pct_change().rename("brent_return")
    return series


def format_signal_message(
    market_config: MarketConfig, results: List[Dict[str, object]], current_time: datetime
) -> str:
    """Build a Discord message summarizing signals for a market."""
    date_str = current_time.strftime("%Y-%m-%d")
    header = f"**{market_config.name} market signals for {date_str}**"
    disclaimer = "This educational summary is **not** investment advice."
    lines = [header, disclaimer, "Ticker | Prob(up) | Signal | Size | Last close"]
    lines.append("--- | --- | --- | --- | ---")
    for item in results:
        lines.append(
            f"{item['ticker']} | {item['probability']:.2f} | {item['signal']} | "
            f"${item['suggested_size']:.0f} | ${item['latest_close']:.2f}"
        )
    lines.append("Walk-forward CV accuracy is based on in-sample evaluation and may not generalize.")
    return "\n".join(lines)


def post_to_discord(message: str) -> None:
    """Send the formatted message to Discord if a webhook is configured."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("DISCORD_WEBHOOK_URL not set; skipping Discord notification.")
        return
    payload = {"content": message}
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Posted signals to Discord successfully.")
    except requests.RequestException as exc:  # pragma: no cover - defensive logging
        logging.exception("Failed to post to Discord: %s", exc)


class MarketJob:
    """Callable used by the schedule library to trigger market predictions."""

    def __init__(self, config: MarketConfig) -> None:
        self.config = config

    def __call__(self) -> None:
        exchange_tz = pytz.timezone(self.config.exchange_timezone)
        local_tz = pytz.timezone(LOCAL_TZ)
        now_exchange = datetime.now(exchange_tz)

        if now_exchange.weekday() not in self.config.trading_days:
            self.config.last_run_date = None
            return

        market_open = exchange_tz.localize(
            datetime.combine(now_exchange.date(), self.config.open_time)
        )
        window_end = market_open + timedelta(minutes=RUN_WINDOW_MINUTES)

        if not (market_open <= now_exchange <= window_end):
            return

        if self.config.last_run_date == now_exchange.date():
            return

        logging.info("Running predictions for %s market.", self.config.name)
        self.config.last_run_date = now_exchange.date()

        start_date = datetime.now(tz=pytz.UTC) - timedelta(days=5 * 365)
        brent_returns = fetch_brent_returns(start=start_date)

        results: List[Dict[str, object]] = []
        for ticker in self.config.tickers:
            prediction = train_and_predict(ticker, self.config, brent_returns)
            if prediction:
                results.append(prediction)

        if not results:
            logging.warning("No predictions generated for %s market today.", self.config.name)
            return

        now_local = datetime.now(local_tz)
        message = format_signal_message(self.config, results, now_local)
        post_to_discord(message)


def run_immediate_predictions(configs: List[MarketConfig]) -> None:
    """Run predictions immediately for all markets (used by RUN_ON_START)."""
    start_date = datetime.now(tz=pytz.UTC) - timedelta(days=5 * 365)
    brent_returns = fetch_brent_returns(start=start_date)
    local_tz = pytz.timezone(LOCAL_TZ)
    now_local = datetime.now(local_tz)

    for config in configs:
        results: List[Dict[str, object]] = []
        for ticker in config.tickers:
            prediction = train_and_predict(ticker, config, brent_returns)
            if prediction:
                results.append(prediction)
        if results:
            message = format_signal_message(config, results, now_local)
            post_to_discord(message)
        else:
            logging.warning("Immediate prediction run produced no results for %s market.", config.name)


def main() -> None:
    setup_logging()
    configs = load_market_configs()

    if os.getenv("RUN_ON_START") == "1":
        logging.info("RUN_ON_START=1: executing immediate predictions before scheduling.")
        run_immediate_predictions(configs)

    for config in configs:
        job = MarketJob(config)
        schedule.every(1).minutes.do(job).tag(config.name)
        logging.info(
            "Scheduled %s market watcher every minute around %s local time.",
            config.name,
            config.open_time,
        )

    logging.info("Scheduler initialized. Waiting for market open windows...")
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()
