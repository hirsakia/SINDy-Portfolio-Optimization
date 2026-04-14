"""
SINDy Portfolio Optimization — v8
Backtest engine: daily, multi-asset, with regime-adaptive bounds.

Contains:
  • backtest              — core backtesting loop
  • bootstrap_sharpe_diff — paired bootstrap significance test
  • run_oos               — run OOS analysis for a single split
  • print_oos             — pretty-print OOS strategy metrics
  • print_bootstrap       — pretty-print bootstrap p-values
"""

import numpy as np

from .config import AF, COMMON_OOS
from .models import shrink_cov
from .forecasters import MomentumForecaster, SINDyRiskMomentumForecaster
from .solvers import solve_mvo, solve_cvar


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────

def max_drawdown(cum_curve):
    """Maximum drawdown from a cumulative return curve."""
    peak = np.maximum.accumulate(cum_curve)
    dd = (cum_curve - peak) / np.where(peak > 0, peak, 1)
    return np.min(dd)


# ─────────────────────────────────────────────────
# Core Backtest
# ─────────────────────────────────────────────────

def backtest(
    rets_df,
    forecaster_cls,
    forecaster_kwargs=None,
    solver="mvo",
    train_min=504,              # 2 years daily
    rebalance_every=21,         # monthly
    gamma=5.0,
    lam_tc=0.002,
    leverage=1.0,
    lb=0.0,
    ub=0.20,
    cvar_alpha=0.95,
    cvar_scen_len=126,
    realized_cost_bps=3.0,      # slightly higher for daily
    cov_window=126,
    adaptive_bounds=False,
    stress_ub_mult=0.5,
    stress_gamma_mult=3.0,      # triple risk aversion in stress
):
    forecaster_kwargs = forecaster_kwargs or {}
    X = rets_df.values
    T, N = X.shape
    cost_mult = realized_cost_bps / 10_000

    w = np.ones(N) / N
    pnl, turnover_list, weights_hist = [], [], []
    coef_history, regime_history = [], []

    for t0 in range(train_min, T - 1, rebalance_every):
        X_train = X[:t0]
        fc = forecaster_cls(**forecaster_kwargs).fit(X_train)
        mu_pred = fc.predict(X[t0 - 1])

        # Covariance
        Sigma = None
        if hasattr(fc, 'predict_cov'):
            try:
                Sigma = fc.predict_cov(cov_window)
            except Exception:
                pass
        if Sigma is None:
            X_cov = X[max(0, t0 - cov_window):t0]
            Sigma = shrink_cov(X_cov) + 1e-6 * np.eye(N)

        # Regime-adaptive
        ub_eff, gamma_eff = ub, gamma
        if adaptive_bounds and hasattr(fc, 'get_regime_info'):
            info = fc.get_regime_info()
            regime_history.append((rets_df.index[t0], info))
            if info['is_stressed']:
                ub_eff = ub * stress_ub_mult
                gamma_eff = gamma * stress_gamma_mult

        # Solve
        if solver == "cvar":
            scen = X[max(0, t0 - cvar_scen_len):t0]
            w_new = solve_cvar(scen, alpha=cvar_alpha, w_prev=w,
                               lam_tc=lam_tc, leverage=leverage, lb=lb, ub=ub_eff)
        else:
            w_new = solve_mvo(mu_pred, Sigma, w_prev=w, lam_tc=lam_tc,
                              leverage=leverage, lb=lb, ub=ub_eff, gamma=gamma_eff)

        to = np.sum(np.abs(w_new - w))
        turnover_list.append(to)
        cost = cost_mult * to

        if hasattr(fc, 'get_coefficients'):
            try:
                c = fc.get_coefficients()
                if len(c) > 0:
                    coef_history.append((rets_df.index[t0], c))
            except Exception:
                pass

        for tau in range(t0, min(t0 + rebalance_every, T)):
            gross = float(w_new @ X[tau])
            net = gross - cost if tau == t0 else gross
            pnl.append(net)

        w = w_new
        weights_hist.append((rets_df.index[t0], w_new.copy()))

    pnl = np.array(pnl)
    curve = np.cumprod(1 + pnl)

    mean_ret = pnl.mean() * AF
    vol = pnl.std(ddof=1) * np.sqrt(AF)
    sharpe = mean_ret / (vol + 1e-12)
    down = pnl[pnl < 0]
    sortino = mean_ret / (down.std(ddof=1) * np.sqrt(AF) + 1e-12) if len(down) > 0 else np.inf
    mdd = max_drawdown(curve)
    calmar = mean_ret / (abs(mdd) + 1e-12)

    return {
        "pnl": pnl, "curve": curve,
        "sharpe": sharpe, "sortino": sortino,
        "mean_ret": mean_ret, "vol": vol,
        "max_dd": mdd, "calmar": calmar,
        "avg_turnover": np.mean(turnover_list),
        "weights_hist": weights_hist,
        "coef_history": coef_history,
        "regime_history": regime_history,
    }


# ─────────────────────────────────────────────────
# Bootstrap Significance Test
# ─────────────────────────────────────────────────

def bootstrap_sharpe_diff(pnl_a, pnl_b, n_boot=10000, seed=42):
    """Paired bootstrap test for Sharpe ratio difference."""
    rng = np.random.RandomState(seed)
    T = min(len(pnl_a), len(pnl_b))
    pnl_a, pnl_b = pnl_a[:T], pnl_b[:T]

    def sharpe(p):
        return p.mean() / (p.std(ddof=1) + 1e-12) * np.sqrt(AF)

    obs = sharpe(pnl_a) - sharpe(pnl_b)
    combined = np.column_stack([pnl_a, pnl_b])
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, T, T)
        diffs[i] = sharpe(combined[idx, 0]) - sharpe(combined[idx, 1])
    centered = diffs - diffs.mean()
    p_val = np.mean(np.abs(centered) >= abs(obs))
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    return obs, ci, p_val


# ─────────────────────────────────────────────────
# OOS Analysis
# ─────────────────────────────────────────────────

def run_oos(rets_df, asset_classes, split_date, label=""):
    """Run OOS analysis for a single split date. Returns dict of results."""
    rets_test = rets_df[rets_df.index >= split_date]
    X_test = rets_test.values
    T_test, N = X_test.shape
    tm = COMMON_OOS["train_min"]
    rb = COMMON_OOS["rebalance_every"]

    if T_test < tm + 100:
        print(f"    {label}: insufficient OOS data ({T_test} days)")
        return {}

    oos = {}

    # Equal Weight
    pnl = X_test[tm:] @ (np.ones(N) / N)
    c = np.cumprod(1 + pnl)
    mr, vo = pnl.mean() * AF, pnl.std(ddof=1) * np.sqrt(AF)
    down = pnl[pnl < 0]
    oos["Equal Weight"] = {
        "pnl": pnl, "curve": c, "sharpe": mr / (vo + 1e-12),
        "sortino": mr / (down.std(ddof=1) * np.sqrt(AF) + 1e-12) if len(down) > 0 else np.inf,
        "mean_ret": mr, "vol": vo, "max_dd": max_drawdown(c),
        "calmar": mr / (abs(max_drawdown(c)) + 1e-12),
    }

    # Risk Parity
    pnl_rp = []
    for t0 in range(tm, T_test - 1, rb):
        vol_est = np.std(X_test[max(0, t0 - 126):t0], axis=0) + 1e-12
        w = (1.0 / vol_est); w = w / w.sum()
        for tau in range(t0, min(t0 + rb, T_test)):
            pnl_rp.append(float(w @ X_test[tau]))
    pnl_rp = np.array(pnl_rp)
    c = np.cumprod(1 + pnl_rp)
    mr, vo = pnl_rp.mean() * AF, pnl_rp.std(ddof=1) * np.sqrt(AF)
    down = pnl_rp[pnl_rp < 0]
    oos["Risk Parity"] = {
        "pnl": pnl_rp, "curve": c, "sharpe": mr / (vo + 1e-12),
        "sortino": mr / (down.std(ddof=1) * np.sqrt(AF) + 1e-12) if len(down) > 0 else np.inf,
        "mean_ret": mr, "vol": vo, "max_dd": max_drawdown(c),
        "calmar": mr / (abs(max_drawdown(c)) + 1e-12),
    }

    # Momentum MVO
    try:
        oos["Momentum MVO"] = backtest(rets_test,
            forecaster_cls=MomentumForecaster, forecaster_kwargs={"lookback": 60},
            solver="mvo", **COMMON_OOS)
    except Exception as e:
        print(f"    Momentum FAILED: {e}")

    # SINDy CVaR
    try:
        oos["SINDy CVaR"] = backtest(rets_test,
            forecaster_cls=SINDyRiskMomentumForecaster,
            forecaster_kwargs={
                "mom_lookback": 60, "regime_threshold_pctile": 75,
                "n_components": 4, "degree": 2,
                "equity_dampen": 0.3, "defensive_boost": 2.0,
                "asset_classes": asset_classes,
            },
            solver="cvar", adaptive_bounds=True, stress_ub_mult=0.6,
            stress_gamma_mult=2.0, **COMMON_OOS)
    except Exception as e:
        print(f"    SINDy CVaR FAILED: {e}")

    return oos


# ─────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────

def print_oos(oos, label=""):
    """Pretty-print OOS strategy metrics."""
    if not oos:
        return
    print(f"\n    {'Strategy':<20s} {'Sharpe':>7s} {'MaxDD':>8s} {'Vol':>7s} {'Return':>8s}")
    print(f"    {'-'*50}")
    for nm, r in oos.items():
        print(f"    {nm:<20s} {r['sharpe']:>7.3f} {r['max_dd']:>7.1%} {r['vol']:>6.1%} {r['mean_ret']:>7.2%}")


def print_bootstrap(oos, label=""):
    """Pretty-print bootstrap p-values for SINDy CVaR vs benchmarks."""
    if "SINDy CVaR" not in oos:
        return
    print(f"\n    Bootstrap (SINDy CVaR vs benchmarks):")
    for bm in ["Equal Weight", "Risk Parity", "Momentum MVO"]:
        if bm in oos:
            diff, ci, pval = bootstrap_sharpe_diff(oos["SINDy CVaR"]["pnl"], oos[bm]["pnl"])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "n.s."
            print(f"      vs {bm:<20s}: dSharpe={diff:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] p={pval:.3f} {sig}")
