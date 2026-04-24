import numpy as np
import pandas as pd
from numpy.linalg import cholesky


def make_weights(n: int, custom: list = None) -> np.ndarray:
    """
    Build a portfolio weight vector.

    Args:
        n:      number of assets
        custom: optional list of custom weights (will be normalized to sum=1)

    Returns:
        numpy array of weights summing to 1
    """
    if custom is not None:
        w = np.array(custom, dtype=float)
        assert len(w) == n, "custom weights length must match number of assets"
        return w / w.sum()
    return np.ones(n) / n


def portfolio_returns(returns: pd.DataFrame,
                      weights: np.ndarray) -> pd.Series:
    """
    Compute daily portfolio returns as weighted sum of asset returns.

    r_portfolio = w^T r

    Args:
        returns: DataFrame of daily log-returns (n_days x n_assets)
        weights: weight vector (n_assets,)

    Returns:
        Series of daily portfolio returns
    """
    return returns @ weights


def portfolio_stats(port_ret: pd.Series) -> dict:
    """
    Compute annualized portfolio statistics.

    Args:
        port_ret: Series of daily portfolio returns

    Returns:
        dict with annual_return, annual_vol, sharpe_ratio
    """
    mu    = port_ret.mean() * 252
    sigma = port_ret.std()  * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0.0
    return {
        "annual_return": round(mu, 4),
        "annual_vol":    round(sigma, 4),
        "sharpe_ratio":  round(sharpe, 4),
    }


def covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate the annualized covariance matrix from daily returns.

    Args:
        returns: DataFrame of daily log-returns

    Returns:
        numpy array of shape (n_assets, n_assets), annualized
    """
    return returns.cov().values * 252


def simulate_portfolio_returns(returns: pd.DataFrame,
                                weights: np.ndarray,
                                n_simulations: int = 10_000,
                                seed: int = 42) -> np.ndarray:
    """
    Simulate one-day portfolio returns using correlated normal draws.

    Method (Cholesky):
    1. Estimate daily mean vector mu and covariance matrix Sigma
    2. Cholesky-decompose Sigma: Sigma = L @ L.T
    3. Draw Z ~ N(0, I), shape (n_simulations, n_assets)
    4. Correlated returns: R = Z @ L.T + mu
    5. Portfolio return: r_p = R @ weights

    Why Cholesky: plain N(0,1) draws are uncorrelated — Procter&Gamble and Exxon Mobile 
    would move independently. Cholesky transforms them to match the
    historical correlation structure.

    Args:
        returns:       DataFrame of historical daily log-returns
        weights:       portfolio weight vector
        n_simulations: number of Monte Carlo scenarios
        seed:          random seed for reproducibility

    Returns:
        numpy array of shape (n_simulations,) — one portfolio return for each scenario
    """
    rng   = np.random.default_rng(seed)
    mu    = returns.mean().values       # shape (n_assets,)
    Sigma = returns.cov().values        # shape (n_assets, n_assets)

    L     = cholesky(Sigma)             # lower triangular, Sigma = L @ L.T

    # independent standard normals
    Z     = rng.standard_normal((n_simulations, len(mu)))

    # correlated asset returns: each row is one scenario
    sim_asset_returns = Z @ L.T + mu   # shape (n_simulations, n_assets)

    # portfolio return per scenario
    sim_port_returns  = sim_asset_returns @ weights   # shape (n_simulations,)

    return sim_port_returns