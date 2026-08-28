from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize, linprog
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

app = Flask(__name__)
CORS(app)

# =========================================================================
# MOTEURS D'ALLOCATION
# =========================================================================
# Chaque moteur reçoit mean_returns (np.array, rendements annualisés),
# cov_matrix (np.array n x n, covariance annualisée), rf (float),
# log_returns (DataFrame des rendements log journaliers) et tickers,
# et renvoie un vecteur de poids (np.array) sommant à 1, sans vente à
# découvert (0 <= w_i <= 1).

def _optimize(objective, n, x0=None):
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    x0 = x0 if x0 is not None else np.repeat(1.0 / n, n)
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
    w = np.clip(res.x, 0, None)
    total = w.sum()
    return w / total if total > 0 else np.repeat(1.0 / n, n)


def engine_max_sharpe(mean_returns, cov_matrix, rf, **kw):
    """Markowitz - maximise le ratio de Sharpe (portefeuille tangent)."""
    n = len(mean_returns)
    def neg_sharpe(w):
        vol = np.sqrt(max(w @ cov_matrix @ w, 1e-12))
        return -(w @ mean_returns - rf) / vol
    return _optimize(neg_sharpe, n)


def engine_min_variance(mean_returns, cov_matrix, rf, **kw):
    """Markowitz - minimise la variance totale du portefeuille."""
    n = len(mean_returns)
    return _optimize(lambda w: w @ cov_matrix @ w, n)


def engine_equal_weight(mean_returns, cov_matrix, rf, **kw):
    """Pondération naïve 1/N."""
    n = len(mean_returns)
    return np.repeat(1.0 / n, n)


def engine_inverse_volatility(mean_returns, cov_matrix, rf, **kw):
    """Pondère chaque actif inversement à sa volatilité individuelle."""
    vol = np.sqrt(np.diag(cov_matrix))
    inv = 1.0 / np.where(vol > 0, vol, 1e-9)
    return inv / inv.sum()


def engine_risk_parity(mean_returns, cov_matrix, rf, **kw):
    """Equal Risk Contribution : chaque actif contribue à parts égales au risque total."""
    n = len(mean_returns)
    def erc_objective(w):
        w = np.clip(w, 1e-6, None)
        port_var = w @ cov_matrix @ w
        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib
        target = port_var / n
        return np.sum((risk_contrib - target) ** 2)
    return _optimize(erc_objective, n)


def engine_max_diversification(mean_returns, cov_matrix, rf, **kw):
    """Maximise le ratio de diversification de Choueifaty & Coignard."""
    n = len(mean_returns)
    asset_vols = np.sqrt(np.diag(cov_matrix))
    def neg_diversification_ratio(w):
        port_vol = np.sqrt(max(w @ cov_matrix @ w, 1e-12))
        return -(w @ asset_vols) / port_vol
    return _optimize(neg_diversification_ratio, n)


def engine_min_cvar(mean_returns, cov_matrix, rf, log_returns=None, alpha=0.95, **kw):
    """Minimise la CVaR historique à 95% (formulation linéaire de Rockafellar-Uryasev)."""
    n = len(mean_returns)
    if log_returns is None or len(log_returns) < 30:
        return engine_min_variance(mean_returns, cov_matrix, rf)

    R = log_returns.values
    T = R.shape[0]
    # Variables : w (n), VaR (1), u_t (T, >= 0)
    c = np.concatenate([np.zeros(n), [1.0], np.ones(T) / ((1 - alpha) * T)])
    A_ub = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
    b_ub = np.zeros(T)
    A_eq = np.hstack([np.ones((1, n)), [[0.0]], np.zeros((1, T))])
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return engine_min_variance(mean_returns, cov_matrix, rf)
    w = np.clip(res.x[:n], 0, None)
    total = w.sum()
    return w / total if total > 0 else np.repeat(1.0 / n, n)


def engine_black_litterman(mean_returns, cov_matrix, rf, log_returns=None, tickers=None, **kw):
    """
    Black-Litterman simplifié : pas de capitalisations boursières disponibles, donc le
    prior de marché est approché par une pondération inverse-volatilité. Une vue
    automatique (momentum le plus fort vs le plus faible sur ~6 mois) est injectée
    avec une confiance modérée (tau=0.05), puis le portefeuille tangent est recalculé
    sur les rendements espérés a posteriori.
    """
    n = len(mean_returns)
    vol = np.sqrt(np.diag(cov_matrix))
    inv = 1.0 / np.where(vol > 0, vol, 1e-9)
    w_mkt = inv / inv.sum()

    mkt_return = w_mkt @ mean_returns
    mkt_var = w_mkt @ cov_matrix @ w_mkt
    delta = (mkt_return - rf) / mkt_var if mkt_var > 0 else 2.5
    pi = delta * (cov_matrix @ w_mkt)

    post_mean = pi
    if log_returns is not None and n >= 2:
        window = min(126, len(log_returns))
        momentum = log_returns.tail(window).sum().values
        high_idx, low_idx = int(np.argmax(momentum)), int(np.argmin(momentum))
        if high_idx != low_idx:
            tau = 0.05
            P = np.zeros((1, n))
            P[0, high_idx], P[0, low_idx] = 1.0, -1.0
            view_magnitude = (momentum[high_idx] - momentum[low_idx]) * (252.0 / window)
            Q = np.array([view_magnitude])
            try:
                omega = P @ (tau * cov_matrix) @ P.T
                tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
                omega_inv = np.linalg.inv(omega)
                post_cov_factor = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
                post_mean = post_cov_factor @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
            except np.linalg.LinAlgError:
                post_mean = pi

    def neg_sharpe(w):
        vol_p = np.sqrt(max(w @ cov_matrix @ w, 1e-12))
        return -(w @ post_mean - rf) / vol_p
    return _optimize(neg_sharpe, n)


def _get_quasi_diag(link):
    """Réordonne les feuilles d'un dendrogramme pour que les actifs proches soient adjacents (Lopez de Prado)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov_matrix, items):
    sub_cov = cov_matrix[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub_cov)
    ivp /= ivp.sum()
    return ivp @ sub_cov @ ivp


def _rec_bipart(cov_matrix, sort_ix):
    """Allocation par bissection récursive inverse-variance le long d'un ordre de feuilles donné."""
    w = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]
    while len(cluster_items) > 0:
        cluster_items = [c[j:k] for c in cluster_items
                          for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(cluster_items), 2):
            c0, c1 = cluster_items[i], cluster_items[i + 1]
            var0, var1 = _cluster_var(cov_matrix, c0), _cluster_var(cov_matrix, c1)
            alpha = 1 - var0 / (var0 + var1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w


def engine_hrp(mean_returns, cov_matrix, rf, log_returns=None, **kw):
    """Hierarchical Risk Parity (Lopez de Prado) : clustering des corrélations puis bissection récursive."""
    n = len(mean_returns)
    corr = log_returns.corr().values
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, None))
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method='single')
    sort_ix = _get_quasi_diag(link)
    w = _rec_bipart(cov_matrix, sort_ix)
    result = np.zeros(n)
    for idx, weight in w.items():
        result[idx] = weight
    return result / result.sum()


def engine_ap_trees(mean_returns, cov_matrix, rf, log_returns=None, tickers=None, **kw):
    """
    AP Trees (Bryzgalova, Pelger & Zhu, 2023) - proxy prix uniquement.
    L'article original partitionne les actifs sur des caractéristiques fondamentales
    (value, taille, momentum...) issues de données de firmes. Cet outil ne dispose
    que des prix, donc l'arbre est construit récursivement sur deux caractéristiques
    dérivées des prix - momentum (rendement log cumulé ~6 mois) puis volatilité
    annualisée - en scindant chaque nœud à la médiane. Les feuilles/nœuds sont ensuite
    combinés par bissection récursive inverse-variance, comme pour HRP.
    """
    n = len(mean_returns)
    window = min(126, len(log_returns))
    momentum = log_returns.tail(window).sum().values
    volatility = log_returns.std().values * np.sqrt(252)
    chars = pd.DataFrame({'momentum': momentum, 'volatility': volatility}, index=range(n))

    def split(node, depth):
        idx = list(node.index)
        if len(idx) <= 1 or depth >= 2:
            return idx
        col = 'momentum' if depth == 0 else 'volatility'
        median_val = node[col].median()
        high, low = node[node[col] >= median_val], node[node[col] < median_val]
        if len(high) == 0 or len(low) == 0:
            return idx
        return split(high, depth + 1) + split(low, depth + 1)

    sort_ix = split(chars, 0)
    w = _rec_bipart(cov_matrix, sort_ix)
    result = np.zeros(n)
    for idx, weight in w.items():
        result[idx] = weight
    return result / result.sum()


ENGINE_FUNCS = {
    'max_sharpe': engine_max_sharpe,
    'min_variance': engine_min_variance,
    'risk_parity': engine_risk_parity,
    'max_diversification': engine_max_diversification,
    'hrp': engine_hrp,
    'ap_trees': engine_ap_trees,
    'min_cvar': engine_min_cvar,
    'black_litterman': engine_black_litterman,
    'equal_weight': engine_equal_weight,
    'inverse_volatility': engine_inverse_volatility,
}

ENGINES_META = {
    'max_sharpe': {
        'label': 'Markowitz — Max Sharpe',
        'category': 'Markowitz',
        'description': "Maximise le ratio rendement excédentaire / volatilité (portefeuille tangent)."
    },
    'min_variance': {
        'label': 'Markowitz — Variance Minimale',
        'category': 'Markowitz',
        'description': "Minimise la volatilité totale du portefeuille, sans égard au rendement espéré."
    },
    'risk_parity': {
        'label': 'Risk Parity (ERC)',
        'category': 'Risque',
        'description': "Chaque actif contribue à parts égales au risque total du portefeuille."
    },
    'max_diversification': {
        'label': 'Maximum Diversification',
        'category': 'Risque',
        'description': "Maximise le ratio de diversification (Choueifaty & Coignard)."
    },
    'hrp': {
        'label': 'Hierarchical Risk Parity (HRP)',
        'category': 'Clustering',
        'description': "Alloue par clustering hiérarchique des corrélations, sans inversion de matrice (Lopez de Prado)."
    },
    'ap_trees': {
        'label': 'AP Trees (Bryzgalova) — proxy prix',
        'category': 'Clustering',
        'description': "Partitionnement en arbre sur caractéristiques dérivées des prix (momentum, volatilité), inspiré de Bryzgalova, Pelger & Zhu (2023). Proxy sans données fondamentales."
    },
    'min_cvar': {
        'label': 'Minimum CVaR (95%)',
        'category': 'Risque de queue',
        'description': "Minimise la perte conditionnelle attendue au-delà de la VaR 95% (Rockafellar-Uryasev)."
    },
    'black_litterman': {
        'label': 'Black-Litterman (simplifié)',
        'category': 'Bayésien',
        'description': "Combine un prior de marché (proxy inverse-vol) avec une vue automatique de momentum."
    },
    'equal_weight': {
        'label': 'Équipondéré (1/N)',
        'category': 'Naïf',
        'description': "Pondération identique pour chaque actif."
    },
    'inverse_volatility': {
        'label': 'Inverse Volatilité',
        'category': 'Naïf',
        'description': "Pondère chaque actif inversement à sa volatilité individuelle."
    },
}


def compute_engine_weights(engine, mean_returns, cov_matrix, log_returns, rf, tickers):
    if engine not in ENGINE_FUNCS:
        raise ValueError(f"Moteur d'allocation inconnu: {engine}")

    n = len(mean_returns)
    if n == 1:
        return np.array([1.0])

    fn = ENGINE_FUNCS[engine]
    w = fn(mean_returns=mean_returns.values, cov_matrix=cov_matrix.values,
           rf=rf, log_returns=log_returns, tickers=tickers)
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    total = w.sum()
    return w / total if total > 0 else np.repeat(1.0 / n, n)


# =========================================================================

def compute_portfolio_metrics(symbol_list, period, geo, rf_manual, portfolio_value_now, horizon_years, n_paths, cash, engine='max_sharpe'):
    """
    Calcule tous les métriques du portefeuille :
    - Frontière efficiente & Sharpe
    - Allocation optimale (Montants) selon le moteur choisi
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

    # --- 4. Frontière Efficiente (Simulation, à titre de visualisation) ---
    num_portfolios = 5000
    np.random.seed(42)
    n_assets = len(df.columns)
    results = np.zeros((num_portfolios, 3))

    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - rf) / port_volatility if port_volatility > 0 else 0

        results[i, 0] = port_return
        results[i, 1] = port_volatility
        results[i, 2] = sharpe_ratio

    results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe'])

    # --- 5. Portefeuille Optimal selon le moteur d'allocation choisi ---
    if engine not in ENGINE_FUNCS:
        raise ValueError(f"Moteur d'allocation inconnu: {engine}")

    best_weights = compute_engine_weights(engine, mean_returns, cov_matrix, log_returns, rf, list(df.columns))

    optimal_return = float(best_weights @ mean_returns.values)
    optimal_volatility = float(np.sqrt(best_weights @ cov_matrix.values @ best_weights))
    optimal_sharpe = float((optimal_return - rf) / optimal_volatility) if optimal_volatility > 0 else 0.0

    # --- 6. Calcul des Montants (Allocation) ---
    investable_amount = portfolio_value_now - cash
    if investable_amount < 0:
        investable_amount = 0

    best_df = pd.DataFrame({'Asset': df.columns, 'Weight': best_weights}).sort_values('Weight', ascending=False)
    best_df['Amount'] = best_df['Weight'] * investable_amount

    # --- 7. Calculs de Risque Avancés (VaR, Drawdown, Sortino) ---
    weighted_log_returns = log_returns.dot(best_weights)

    var_95 = np.percentile(weighted_log_returns, 5)

    cum_returns = (1 + weighted_log_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    downside_returns = weighted_log_returns[weighted_log_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (optimal_return - rf) / downside_std if downside_std > 0 else 0

    calmar_ratio = optimal_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # --- 8. Contribution au Risque ---
    w_best = pd.Series(best_weights, index=df.columns)
    marginal_contribution = cov_matrix @ w_best
    risk_contribution = w_best * marginal_contribution
    total_variance = optimal_volatility ** 2
    percent_contribution = risk_contribution / total_variance if total_variance > 0 else risk_contribution * 0

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
        "engine": engine,
        "engine_label": ENGINES_META[engine]['label'],
        "engine_description": ENGINES_META[engine]['description'],
        "risk_free_rate": rf,
        "optimal_return": optimal_return,
        "optimal_volatility": optimal_volatility,
        "optimal_sharpe": optimal_sharpe,
        "var_95": var_95,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
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
        engine = data.get("engine", "max_sharpe")

        if engine not in ENGINE_FUNCS:
            return jsonify({"error": f"Moteur d'allocation inconnu: {engine}"}), 400

        result = compute_portfolio_metrics(
            symbol_list=symbol_list,
            period=period,
            geo=geo,
            rf_manual=rf_manual,
            portfolio_value_now=portfolio_value_now,
            horizon_years=horizon_years,
            n_paths=n_paths,
            cash=cash,
            engine=engine
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/engines", methods=["GET"])
def engines_api():
    return jsonify([{"id": engine_id, **meta} for engine_id, meta in ENGINES_META.items()])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
