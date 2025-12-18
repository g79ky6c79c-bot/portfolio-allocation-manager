# Portfolio Allocation Manager

A quantitative portfolio optimization and risk analysis tool, designed to bridge **financial theory** and **real-world portfolio management**.

This project implements modern portfolio theory concepts (Markowitz, Sharpe ratio, Monte Carlo simulations) through a full **Python backend + interactive frontend**, and is actively used to manage a *shadow equity portfolio*.

---

##  Project Overview

The **Portfolio Allocation Manager** allows users to:
- Build and analyze multi-asset portfolios
- Optimize allocations based on risk-return trade-offs
- Visualize portfolio behavior under uncertainty
- Apply quantitative finance concepts in near-real market conditions

The tool is currently used to manage a **shadow portfolio on eToro**, with a **+5% performance (as of Thursday, December 11)**.

---

##  Key Features

### Portfolio Optimization
- Efficient Frontier construction (Markowitz framework)
- Optimal allocation using **Maximum Sharpe Ratio**
- Annualized expected return and volatility
- Risk contribution analysis per asset

### Risk & Simulation
- Monte Carlo simulation (up to **100,000 scenarios**)
- Future portfolio value distribution
- Percentile-based risk metrics (5%, median, 95%)

### Market Analysis
- Historical performance visualization (**Base 100**)
- Logarithmic returns analysis (volatility dynamics)
- Multi-asset comparison over configurable horizons

---

##  Tech Stack

### Backend
- **Python**
- Flask (REST API)
- NumPy, Pandas
- yFinance (market data)

### Frontend
- HTML / CSS / JavaScript
- Chart.js (interactive financial visualizations)

### Architecture
- REST API-based design
- Separation of computation (backend) and visualization (frontend)
- Built as a mini **portfolio management engine**

---

##  Methodology

- Log returns computed from adjusted close prices
- Annualization based on 252 trading days
- Random portfolio generation for efficient frontier
- Risk-free rate dynamically fetched (or manually overridden)
- Monte Carlo simulation based on Geometric Brownian Motion

---

##  Use Case: Shadow Portfolio (eToro)

The tool is applied to a **realistic investment process**:
- Selection of **18 non-tech equities**
- Fundamental analysis (financial statements & historical prices)
- Sector outlook and forward-looking considerations
- Quantitative allocation and risk monitoring via the tool

This allows direct confrontation between **theoretical models** and **market behavior**.

---

##  API Endpoint

### POST `/api/portfolio`

**Payload example:**
```json
{
  "tickers": "META,JPM,TSLA,MSFT,AAPL,AMGN",
  "period": "1y",
  "geo": "USA",
  "rf_manual": 2.5,
  "portfolio_value_now": 10000,
  "horizon_years": 1,
  "n_paths": 100000
}
