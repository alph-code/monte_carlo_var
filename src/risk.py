import numpy as np
from scipy import stats


def returns_to_losses(returns: np.ndarray) -> np.ndarray:
    """
    Convert returns to losses.

    Loss = -return so that positive values mean losing money.

    Args:
        returns: array of returns (positive = gain)
cd 
    Returns:
        array of losses (positive = bad)
    """
    return -np.asarray(returns)


def var_historical(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Historical VaR: alpha-quantile of the empirical loss distribution.
    No distribution assumption — uses actual past data.

    Args:
        losses: array of historical losses
        alpha:  confidence level e.g. 0.95

    Returns:
        VaR as a positive loss fraction
    """
    return float(np.quantile(losses, alpha))


def var_parametric(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR.

    Assumes L ~ N(mu, sigma^2).
    VaR_alpha = mu + sigma * z_alpha

    Underestimates VaR when losses have fat tails (kurtosis > 0).

    Args:
        losses: array of historical losses
        alpha:  confidence level

    Returns:
        VaR estimate
    """
    mu    = losses.mean()
    sigma = losses.std()
    z     = stats.norm.ppf(alpha)
    return float(mu + sigma * z)


def var_montecarlo(sim_losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Monte Carlo VaR: quantile of the simulated loss distribution.

    Args:
        sim_losses: array of simulated losses
        alpha:      confidence level

    Returns:
        VaR estimate
    """
    return float(np.quantile(sim_losses, alpha))


def cvar_historical(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Historical CVaR (Expected Shortfall).

    CVaR_alpha = E[L | L > VaR_alpha]
    Average of losses that exceed VaR. Always >= VaR at same alpha.

    Args:
        losses: array of historical losses
        alpha:  confidence level

    Returns:
        CVaR estimate
    """
    var         = var_historical(losses, alpha)
    tail_losses = losses[losses > var]
    return float(tail_losses.mean()) if len(tail_losses) > 0 else var


def cvar_parametric(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Parametric CVaR under Gaussian assumption.

    CVaR_alpha = mu + sigma * phi(z_alpha) / (1 - alpha)

    where phi is the standard normal PDF.

    Args:
        losses: array of historical losses
        alpha:  confidence level

    Returns:
        CVaR estimate
    """
    mu    = losses.mean()
    sigma = losses.std()
    z     = stats.norm.ppf(alpha)
    return float(mu + sigma * stats.norm.pdf(z) / (1 - alpha))


def cvar_montecarlo(sim_losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Monte Carlo CVaR: average of simulated losses beyond MC VaR.

    Args:
        sim_losses: array of simulated losses
        alpha:      confidence level

    Returns:
        CVaR estimate
    """
    var  = var_montecarlo(sim_losses, alpha)
    tail = sim_losses[sim_losses > var]
    return float(tail.mean()) if len(tail) > 0 else var


def risk_summary(hist_losses: np.ndarray,
                 sim_losses: np.ndarray,
                 alphas: list = [0.90, 0.95, 0.99]) -> dict:
    """
    Compute all VaR and CVaR estimates across multiple confidence levels.

    Args:
        hist_losses: historical loss array
        sim_losses:  simulated loss array
        alphas:      list of confidence levels

    Returns:
        nested dict: {alpha: {metric_name: value}}
    """
    results = {}
    for alpha in alphas:
        results[alpha] = {
            "VaR_historical":  var_historical(hist_losses,  alpha),
            "VaR_parametric":  var_parametric(hist_losses,  alpha),
            "VaR_montecarlo":  var_montecarlo(sim_losses,   alpha),
            "CVaR_historical": cvar_historical(hist_losses, alpha),
            "CVaR_parametric": cvar_parametric(hist_losses, alpha),
            "CVaR_montecarlo": cvar_montecarlo(sim_losses,  alpha),
        }
    return results


def backtest_var(hist_losses: np.ndarray,
                 window: int = 250,
                 alpha: float = 0.95) -> dict:
    """
    Rolling historical VaR backtest.

    For each day t after the warm-up window:
      - Estimate VaR using the past `window` losses
      - Check if actual loss on day t exceeded the forecast

    Expected violation rate = 1 - alpha.
    If actual rate >> expected: model underestimates risk.

    Args:
        hist_losses: full array of historical losses
        window:      rolling estimation window in trading days
        alpha:       confidence level

    Returns:
        dict with violation stats and arrays for plotting
    """
    var_series   = []
    violations   = []

    for t in range(window, len(hist_losses)):
        past   = hist_losses[t - window : t]
        var_t  = float(np.quantile(past, alpha))
        actual = hist_losses[t]

        var_series.append(var_t)
        violations.append(int(actual > var_t))

    var_series   = np.array(var_series)
    violations   = np.array(violations)
    actual_tail  = hist_losses[window:]

    n_days       = len(violations)
    n_violations = int(violations.sum())
    actual_rate  = n_violations / n_days

    return {
        "n_days":          n_days,
        "n_violations":    n_violations,
        "actual_rate":     round(actual_rate, 4),
        "expected_rate":   1.0 - alpha,
        "var_series":      var_series,
        "actual_losses":   actual_tail,
        "violation_flags": violations,
    }