import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def full_dashboard(port_ret: pd.Series,
                   hist_losses: np.ndarray,
                   sim_losses: np.ndarray,
                   summary: dict,
                   bt: dict,
                   alpha: float = 0.95,
                   save_path: str = "figures/dashboard.png") -> None:
    """
    Generate a 4-panel dashboard and save to disk.

    Panels:
        1. Cumulative portfolio returns
        2. Simulated loss distribution with VaR / CVaR lines
        3. Risk summary table (VaR & CVaR across confidence levels)
        4. Rolling VaR backtest with violation markers

    Args:
        port_ret:    Series of daily portfolio returns
        hist_losses: array of historical losses
        sim_losses:  array of Monte Carlo simulated losses
        summary:     output of risk_summary()
        bt:          output of backtest_var()
        alpha:       primary confidence level used in backtest
        save_path:   file path for the saved figure
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Monte Carlo VaR Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Cumulative returns ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cum = (1 + port_ret).cumprod()
    ax1.plot(cum.index, cum.values, color="steelblue", linewidth=1.2)
    ax1.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_title("Cumulative Portfolio Return")
    ax1.set_ylabel("Growth of $1")
    ax1.set_xlabel("Date")
    ax1.tick_params(axis="x", rotation=30)

    # ── Panel 2: Loss distribution ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(sim_losses, bins=80, density=True, alpha=0.6,
             color="salmon", label="MC simulated")
    ax2.hist(hist_losses, bins=60, density=True, alpha=0.5,
             color="steelblue", label="Historical")

    var_mc   = summary[alpha]["VaR_montecarlo"]
    cvar_mc  = summary[alpha]["CVaR_montecarlo"]
    ax2.axvline(var_mc,  color="red",    linestyle="--", linewidth=1.5,
                label=f"MC VaR {int(alpha*100)}%  = {var_mc:.3f}")
    ax2.axvline(cvar_mc, color="darkred", linestyle=":",  linewidth=1.5,
                label=f"MC CVaR {int(alpha*100)}% = {cvar_mc:.3f}")

    ax2.set_title("Loss Distribution")
    ax2.set_xlabel("Loss (positive = bad)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)

    # ── Panel 3: Risk summary table ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")

    alphas  = sorted(summary.keys())
    metrics = list(next(iter(summary.values())).keys())
    col_labels = [f"α={int(a*100)}%" for a in alphas]
    rows = []
    for m in metrics:
        rows.append([f"{summary[a][m]:.4f}" for a in alphas])

    table = ax3.table(
        cellText=rows,
        rowLabels=metrics,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax3.set_title("Risk Summary (VaR & CVaR)", pad=12)

    # ── Panel 4: Backtest ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    days         = np.arange(len(bt["var_series"]))
    actual       = bt["actual_losses"]
    var_series   = bt["var_series"]
    flags        = bt["violation_flags"].astype(bool)

    ax4.plot(days, actual,     color="steelblue", linewidth=0.8,
             alpha=0.7, label="Actual loss")
    ax4.plot(days, var_series, color="red",       linewidth=1.2,
             linestyle="--", label=f"Rolling VaR {int(alpha*100)}%")
    ax4.scatter(days[flags], actual[flags],
                color="red", s=15, zorder=5, label="Violation")

    ax4.set_title(f"Rolling VaR Backtest  "
                  f"(violations: {bt['n_violations']}/{bt['n_days']} = "
                  f"{bt['actual_rate']:.1%}, expected {bt['expected_rate']:.1%})")
    ax4.set_xlabel("Trading day")
    ax4.set_ylabel("Loss")
    ax4.legend(fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {save_path}")
