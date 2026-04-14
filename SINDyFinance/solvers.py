"""
SINDy Portfolio Optimization — v8
Portfolio solvers: MVO (OSQP/SCS) and CVaR (SCS/CLARABEL).
"""

import numpy as np
import cvxpy as cp


def solve_mvo(mu, Sigma, w_prev=None, lam_tc=0.0,
              leverage=1.0, lb=0.0, ub=0.20, gamma=5.0):
    """Mean-variance optimisation with transaction cost penalty."""
    N = len(mu)
    w = cp.Variable(N)
    obj = cp.quad_form(w, cp.psd_wrap(Sigma)) - gamma * (mu @ w)
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub, cp.norm1(w) <= leverage]
    if w_prev is not None and lam_tc > 0:
        obj += lam_tc * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if w.value is None:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=8000)
    if w.value is None:
        return np.ones(N) / N
    return np.array(w.value).flatten()


def solve_cvar(R_scenarios, alpha=0.95, w_prev=None, lam_tc=0.0,
               leverage=1.0, lb=0.0, ub=0.20):
    """CVaR minimisation (Rockafellar-Uryasev) on historical scenarios."""
    S, N = R_scenarios.shape
    w = cp.Variable(N)
    z = cp.Variable(S)
    t = cp.Variable()
    losses = -R_scenarios @ w
    cvar = t + (1.0 / ((1 - alpha) * S)) * cp.sum(z)
    constr = [
        z >= 0, z >= losses - t,
        cp.sum(w) == 1, w >= lb, w <= ub, cp.norm1(w) <= leverage,
    ]
    obj = cvar
    if w_prev is not None and lam_tc > 0:
        obj += lam_tc * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=8000)
    if w.value is None:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    if w.value is None:
        return np.ones(N) / N
    return np.array(w.value).flatten()
