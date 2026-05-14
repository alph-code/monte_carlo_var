# Monte Carlo VaR & CVaR Calculator

Equal-weight portfolio: ABBV PG JPM XOM JNJ  
Period: 2021-01-01 to latest available trading day

Computes Value-at-Risk and Conditional Value-at-Risk using three
methods: historical simulation, parametric (Gaussian), and Monte Carlo.
Includes a rolling backtest and a 6-panel visualization dashboard.

## Project structure

monte_carlo_var/
├── src/
│   ├── data.py        # price download and log-return computation
│   ├── portfolio.py   # weights, portfolio returns, Cholesky simulation
│   ├── risk.py        # VaR, CVaR, backtest
│   └── plotting.py    # 6-panel dashboard
├── figures/           # output charts saved here
├── main.py            # runs the full pipeline
└── requirements.txt

## Install

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run

python main.py

Output: figures/dashboard.png

## What the dashboard shows

1. Return distribution vs normal fit — fat tails visible
2. Loss distribution with VaR and CVaR marked
3. VaR comparison: parametric underestimates vs simulation methods
4. CVaR comparison across methods
5. Rolling backtest: violation days marked in red
6. QQ-plot: departure from normality quantified