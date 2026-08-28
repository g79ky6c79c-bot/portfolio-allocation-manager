# Portfolio Allocation Manager

Portfolio allocation tool: Flask backend (market data via `yfinance`, optimization via `scipy`) + static web interface (HTML/JS, Chart.js charts).

The portfolio is built from a list of tickers, a historical period, and an allocation engine chosen from 10 approaches (Markowitz, risk, clustering, tail risk, Bayesian, naive). The tool then calculates the optimal weights, risk metrics (VaR, Drawdown, Sortino, Calmar), risk contribution per asset, and projects the future portfolio value via Monte Carlo simulation.

## Features

- Download historical prices (1, 2, or 5 years) via `yfinance`
- Risk-free rate automatically retrieved based on geographical zone (or entered manually)
- 10 selectable allocation engines (see below)
- Efficient frontier (cloud of simulated portfolios) highlighting the selected portfolio
- Allocation in weights (%) and amounts (€/$) deducted from the portfolio value and uninvested cash
- Risk contribution per asset
- Risk metrics: 95% daily VaR, Max Drawdown, Sortino ratio, Calmar ratio
- Monte Carlo simulation of the final portfolio value (mean, median, 5%/95% scenarios)
- Base 100 history and logarithmic returns per asset

## Allocation Engines

| Engine | Category | Principle |
|---|---|---|
| `max_sharpe` | Markowitz | Maximizes the Sharpe ratio (tangency portfolio) |
| `min_variance` | Markowitz | Minimizes total volatility, regardless of return |
| `risk_parity` | Risk | Equal Risk Contribution: each asset contributes equally to total risk |
| `max_diversification` | Risk | Maximizes the diversification ratio (Choueifaty & Coignard) |
| `hrp` | Clustering | Hierarchical Risk Parity (Lopez de Prado): correlation clustering + recursive bisection |
| `ap_trees` | Clustering | AP Trees (Bryzgalova, Pelger & Zhu, 2023) — **price proxy**. The original paper splits on firm fundamental characteristics; lacking data other than prices, the tree is built on momentum then volatility, combined as in HRP |
| `min_cvar` | Tail Risk | Minimizes the 95% historical CVaR (Rockafellar-Uryasev linear formulation) |
| `black_litterman` | Bayesian | Market prior (inverse-volatility proxy, lacking market caps) combined with an automatic momentum view |
| `equal_weight` | Naive | Equal weighting 1/N |
| `inverse_volatility` | Naive | Weights each asset inversely to its individual volatility |

All engines are **long-only** (0 ≤ weight ≤ 1, sum = 1, no short selling) and solved by numerical optimization (`scipy.optimize.minimize`/`linprog`), except for naive engines (direct formula) and HRP/AP Trees (allocation by recursive bisection, without matrix inversion).

## Architecture

```
.
├── main.py       # Flask backend: data downloading, allocation engines, REST API
└── index.html    # Static frontend: form, API calls, Chart.js charts
```

- `main.py` exposes:
  - `POST /api/portfolio` — calculates the allocation and all metrics for a given set of tickers/parameters/engine
  - `GET /api/engines` — lists the available engines (id, category, label, description) to populate the frontend selector
  - `GET /health` — backend health check
- `index.html` queries the backend at `http://localhost:5000` (CORS enabled on the Flask side via `flask-cors`)

## Installation

Prerequisites: Python 3.9+

```bash
pip install flask flask-cors numpy pandas yfinance scipy
```

## Launch

1. Start the backend:

   ```bash
   python main.py
   ```

   The server listens on `http://localhost:5000`.

2. Open `index.html` in a browser (double-click or via a local static server).

## Usage

1. Enter tickers (e.g., `META,JPM,TSLA,MSFT,AAPL,AMGN`), separated by commas
2. Choose an allocation engine from the dropdown menu
3. Adjust the historical period, geographical zone (risk-free rate), portfolio value, uninvested cash, projection horizon, and number of Monte Carlo simulations
4. Click on **Launch optimization**
   
The results are displayed: key statistics, efficient frontier, Monte Carlo distribution, performance history, allocation table (weights and amounts), risk contribution, and simulation statistics.

## Known Limitations

- `ap_trees` is a simplified adaptation of the Bryzgalova, Pelger & Zhu method: it only uses characteristics derived from prices (momentum, volatility), not firm fundamentals (value, size, quality...).
- `black_litterman` uses inverse-volatility weighting as a proxy for the market portfolio, due to the lack of reliably available market capitalizations via `yfinance`.
- The risk-free rate for France and the Eurozone must be entered manually (no reliable `yfinance` ticker available for these zones in the current implementation).
- The backend and frontend are not designed for production deployment as-is (no authentication, `debug=True` enabled on Flask).