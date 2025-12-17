from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

def compute_portfolio_metrics(symbol_list, period, geo, rf_manual, portfolio_value_now, horizon_years, n_paths, cash):
    """
    Calcule tous les métriques du portefeuille :
    - Frontière efficiente & Sharpe
    - Allocation optimale (Montants)
    - Key Metrics (VaR, Drawdown, Sortino, Calmar)
    - Simulation Monte Carlo
    - Historique
    """
    years_hist = int(period[0])
    start_date = (datetime.now() - timedelta(days=365 * years_hist)).strftime("%Y-%m-%d")

    rf_tickers = {
        'USA': '^IRX',
        'France': None,
        'Allemagne': '^DE10Y',
        'UK': '^GB10Y',
        'Zone Euro': None
    }

    # --- 1. Récupération Taux Sans Risque ---
    if geo in ['USA', 'Allemagne', 'UK']:
        selected_ticker = rf_tickers[geo]
        try:
            rf_series = yf.Ticker(selected_ticker).history(period='5d')['Close']
            rf = float(rf_series.dropna().iloc[-1] / 100.0)
        except Exception:
            rf = rf_manual / 100.0
    else:
        rf = rf_manual / 100.0

    # --- 2. Téléchargement des données ---
    data = yf.download(symbol_list, start=start_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close']
    else:
        df = data['Close'] if 'Close' in data.columns else data

    if isinstance(df, pd.Series):
        df = df.to_frame(name=symbol_list[0])

    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Aucune donnee telechargee. Verifiez les tickers.")

    # --- 3. Calculs Statistiques de Base ---
    df100 = df / df.iloc[0] * 100
    log_returns = np.log(df / df.shift(1)).dropna()
    
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    # --- 4. Frontière Efficiente (Simulation) ---
    num_portfolios = 5000
    np.random.seed(42)
    n_assets = len(df.columns)
    results = np.zeros((num_portfolios, 3))
    weights_list = []

    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - rf) / port_volatility if port_volatility > 0 else 0

        results[i, 0] = port_return
        results[i, 1] = port_volatility
        results[i, 2] = sharpe_ratio
        weights_list.append(weights)

    results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])
    
    # --- 5. Portefeuille Optimal (Max Sharpe) ---
    max_sharpe_idx = int(results_df['Sharpe'].idxmax())
    best_weights = weights_list[max_sharpe_idx]

    optimal_return = float(results_df.loc[max_sharpe_idx, 'Return'])
    optimal_volatility = float(results_df.loc[max_sharpe_idx, 'Volatility'])
    optimal_sharpe = float(results_df.loc[max_sharpe_idx, 'Sharpe'])

    # --- 6. Calcul des Montants (Allocation) ---
    investable_amount = portfolio_value_now - cash
    if investable_amount < 0:
        investable_amount = 0

    best_df = pd.DataFrame({'Asset': df.columns, 'Weight': best_weights}).sort_values('Weight', ascending=False)
    best_df['Amount'] = best_df['Weight'] * investable_amount

    # --- 7. Calculs de Risque Avancés (VaR, Drawdown, Sortino) ---
    # On reconstitue les rendements journaliers historiques du portefeuille optimal
    # Returns pondérés
    weighted_log_returns = log_returns.dot(best_weights)
    
    # Value at Risk (VaR 95%) - Journalier
    var_95 = np.percentile(weighted_log_returns, 5) # 5ème percentile

    # Max Drawdown
    cum_returns = (1 + weighted_log_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Ratio de Sortino (Rendement / Volatilité à la baisse uniquement)
    downside_returns = weighted_log_returns[weighted_log_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (optimal_return - rf) / downside_std if downside_std > 0 else 0

    # Ratio de Calmar (Rendement Annuel / Max Drawdown Absolu)
    calmar_ratio = optimal_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # --- 8. Contribution au Risque ---
    w_best = pd.Series(best_weights, index=df.columns)
    marginal_contribution = cov_matrix @ w_best
    risk_contribution = w_best * marginal_contribution
    total_variance = optimal_volatility ** 2
    percent_contribution = risk_contribution / total_variance

    risk_contrib_table = pd.DataFrame({
        'Asset': df.columns,
        'Weight': w_best.values,
        'RiskContribution': percent_contribution.values
    })

    # --- 9. Simulation Monte Carlo ---
    trading_days_per_year = 252
    n_steps = int(trading_days_per_year * horizon_years)
    dt = horizon_years / n_steps if n_steps > 0 else 0

    mu = optimal_return
    sigma = optimal_volatility

    final_total_values = None
    if n_steps > 0 and investable_amount > 0 and n_paths > 0:
        np.random.seed(123)
        z = np.random.normal(size=(n_steps, n_paths))
        daily_returns_port = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
        S_paths_risky = investable_amount * daily_returns_port.cumprod(axis=0)
        final_risky_values = S_paths_risky[-1, :]
        final_total_values = final_risky_values + cash

        expected_final = float(np.mean(final_total_values))
        median_final = float(np.median(final_total_values))
        pct5 = float(np.percentile(final_total_values, 5))
        pct95 = float(np.percentile(final_total_values, 95))
    elif investable_amount <= 0:
        expected_final = median_final = pct5 = pct95 = float(cash)
        final_total_values = np.full(n_paths, cash)
    else:
        expected_final = median_final = pct5 = pct95 = None

    # --- 10. Préparation Historique ---
    df100_filled = df100.ffill().bfill()
    returns_filled = log_returns.fillna(0)
    
    dates_str = df100.index.strftime('%Y-%m-%d').tolist()
    dates_ret_str = log_returns.index.strftime('%Y-%m-%d').tolist()

    history_data = {
        "dates": dates_str,
        "dates_returns": dates_ret_str,
        "base100": {ticker: df100_filled[ticker].tolist() for ticker in df100.columns},
        "log_returns": {ticker: returns_filled[ticker].tolist() for ticker in log_returns.columns}
    }

    # --- 11. Réponse JSON ---
    response = {
        "tickers": list(df.columns),
        "risk_free_rate": rf,
        "optimal_return": optimal_return,
        "optimal_volatility": optimal_volatility,
        "optimal_sharpe": optimal_sharpe,
        # NOUVELLES METRIQUES
        "var_95": var_95,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        # -------------------
        "portfolio_value_now": portfolio_value_now,
        "cash": cash,
        "investable_amount": investable_amount,
        "horizon_years": horizon_years,
        "best_weights": [
            {"asset": row["Asset"], "weight": float(row["Weight"]), "amount": float(row["Amount"])}
            for _, row in best_df.iterrows()
        ],
        "risk_contrib": [
            {"asset": row["Asset"], "weight": float(row["Weight"]), "risk_contribution": float(row["RiskContribution"])}
            for _, row in risk_contrib_table.iterrows()
        ],
        "mc_stats": {
            "expected_final": expected_final,
            "median_final": median_final,
            "pct5": pct5,
            "pct95": pct95
        },
        "efficient_frontier": results_df.to_dict('records'),
        "history": history_data
    }

    if final_total_values is not None:
        hist, bin_edges = np.histogram(final_total_values, bins=60)
        response["mc_histogram"] = {
            "values": hist.tolist(),
            "bins": bin_edges.tolist()
        }

    return response

@app.route("/api/portfolio", methods=["POST"])
def portfolio_api():
    try:
        data = request.get_json()
        tickers_str = data.get("tickers", "")
        symbol_list = [x.strip().upper() for x in tickers_str.split(",") if x.strip() != ""]
        
        if not symbol_list:
            return jsonify({"error": "Aucun ticker fourni"}), 400

        period = data.get("period", "1y")
        geo = data.get("geo", "USA")
        rf_manual = float(data.get("rf_manual", 2.5))
        portfolio_value_now = float(data.get("portfolio_value_now", 10000.0))
        cash = float(data.get("cash", 0.0))
        horizon_years = float(data.get("horizon_years", 1.0))
        n_paths = int(data.get("n_paths", 100000))

        result = compute_portfolio_metrics(
            symbol_list=symbol_list,
            period=period,
            geo=geo,
            rf_manual=rf_manual,
            portfolio_value_now=portfolio_value_now,
            horizon_years=horizon_years,
            n_paths=n_paths,
            cash=cash
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)