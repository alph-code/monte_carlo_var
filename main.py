import numpy as np
import pandas as pd

from src.data      import download_prices, compute_log_returns, summary_stats
from src.portfolio import make_weights, portfolio_returns, portfolio_stats, \
                          simulate_portfolio_returns
from src.risk      import returns_to_losses, risk_summary, backtest_var
from src.plotting  import full_dashboard

# ── configuration ─────────────────────────────────────────────────────────────
TICKERS       = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
START         = "2022-01-01"
END           = "2024-01-01"
ALPHA         = 0.95
N_SIMULATIONS = 10_000
SEED          = 42
BACKTEST_WIN  = 250
ALPHAS        = [0.90, 0.95, 0.99]


def main():
    print("=" * 60)
    print("  Monte Carlo VaR & CVaR Calculator")
    print("=" * 60)

    # ── 1. data ────────────────────────────────────────────────────
    print("\n[1/6] Downloading price data ...")
    prices  = download_prices(TICKERS, START, END)
    returns = compute_log_returns(prices)

    print(f"      {len(prices)} trading days  |  {len(TICKERS)} assets")
    print("\n  Asset summary stats:")
    print(summary_stats(returns).to_string())

    # ── 2. portfolio ───────────────────────────────────────────────
    print("\n[2/6] Building equal-weight portfolio ...")
    weights  = make_weights(len(TICKERS))
    port_ret = portfolio_returns(returns, weights)
    pstats   = portfolio_stats(port_ret)

    print(f"      Annual return : {pstats['annual_return']:.2%}")
    print(f"      Annual vol    : {pstats['annual_vol']:.2%}")
    print(f"      Sharpe ratio  : {pstats['sharpe_ratio']:.2f}")
    print(f"      Daily kurtosis: {port_ret.kurt():.4f}  "
          f"(>0 means fat tails — parametric VaR will underestimate risk)")

    # ── 3. Monte Carlo simulation ──────────────────────────────────
    print(f"\n[3/6] Running {N_SIMULATIONS:,} Monte Carlo scenarios ...")
    sim_returns = simulate_portfolio_returns(
        returns, weights,
        n_simulations=N_SIMULATIONS,
        seed=SEED,
    )
    print(f"      Simulated mean : {sim_returns.mean():.6f}"
          f"  (historical: {port_ret.mean():.6f})")
    print(f"      Simulated std  : {sim_returns.std():.6f}"
          f"  (historical: {port_ret.std():.6f})")

    # ── 4. risk measures ───────────────────────────────────────────
    print("\n[4/6] Computing VaR and CVaR ...")
    hist_losses = returns_to_losses(port_ret.values)
    sim_losses  = returns_to_losses(sim_returns)
    summary     = risk_summary(hist_losses, sim_losses, ALPHAS)

    rows = []
    for alpha, metrics in summary.items():
        for metric, value in metrics.items():
            rows.append({"alpha": alpha, "metric": metric,
                         "value": round(value, 5)})
    df_risk = (pd.DataFrame(rows)
                 .pivot(index="metric", columns="alpha", values="value"))
    print("\n" + df_risk.to_string())

    # ── 5. backtest ────────────────────────────────────────────────
    print(f"\n[5/6] Running rolling VaR backtest "
          f"(window={BACKTEST_WIN} days, α={ALPHA}) ...")
    bt = backtest_var(hist_losses, window=BACKTEST_WIN, alpha=ALPHA)

    direction = "over" if bt["actual_rate"] > bt["expected_rate"] else "under"
    gap = abs(bt["actual_rate"] - bt["expected_rate"])
    print(f"      Days tested  : {bt['n_days']}")
    print(f"      Violations   : {bt['n_violations']}")
    print(f"      Actual rate  : {bt['actual_rate']:.1%}  "
          f"(expected {bt['expected_rate']:.1%})")
    print(f"      Model {direction}estimates risk by {gap:.1%} of days")

    # ── 6. dashboard ───────────────────────────────────────────────
    print("\n[6/6] Generating dashboard ...")
    full_dashboard(
        port_ret    = port_ret,
        hist_losses = hist_losses,
        sim_losses  = sim_losses,
        summary     = summary,
        bt          = bt,
        alpha       = ALPHA,
        save_path   = "figures/dashboard.png",
    )

    print("\nDone. Figure saved to figures/dashboard.png")
    print("=" * 60)


if __name__ == "__main__":
    main()