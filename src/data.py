import yfinance as yf
import numpy as np
import pandas as pd

TICKERS = ["ABBV", "PG", "JPM", "XOM", "JNJ"]
START   = "2021-01-01"
END     = "2027-01-01" 


def download_prices(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Args:
        tickers: list of ticker symbols
        start:   start date string YYYY-MM-DD
        end:     end date string YYYY-MM-DD

    Returns:
        DataFrame of shape (n_days, n_tickers), no NaN rows
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw.dropna(inplace=True)
    return raw


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns: r_t = log(P_t / P_{t-1})

    Why log-returns:
    - Additive over time: r(0,T) = sum of daily r_t
    - Symmetric: gains and losses treated equally on log scale
    - Normally distributed under GBM assumption

    Args:
        prices: DataFrame of closing prices

    Returns:
        DataFrame of log-returns, first row dropped (NaN)
    """
    return np.log(prices / prices.shift(1)).dropna()


def summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized summary statistics for each asset.

    Kurtosis > 0 means fat tails relative to normal distribution.
    Skewness < 0 means more large negative returns than positive.

    Args:
        returns: DataFrame of daily log-returns

    Returns:
        DataFrame with one row per asset
    """
    stats = pd.DataFrame({
        "mean_annual": returns.mean() * 252,
        "vol_annual":  returns.std()  * np.sqrt(252),
        "skewness":    returns.skew(),
        "kurtosis":    returns.kurt(),
        "min":         returns.min(),
        "max":         returns.max(),
    })
    return stats.round(4)