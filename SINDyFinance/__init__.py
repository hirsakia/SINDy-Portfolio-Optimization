"""
SINDy Portfolio Optimization — v8 (Extended History)

A SINDy-based portfolio optimization system that uses regime detection
via sparse identification of nonlinear dynamics to adaptively manage
risk across multiple asset universes.

Modules:
  config       — UNIVERSES, backtest parameters, OOS splits
  data         — yfinance return loader
  models       — RiskFeatureExtractor, SINDyRegimeDetector, SINDyCovarianceEngine
  forecasters  — SampleMean, Momentum, Zero, SINDyRiskMomentum forecasters
  solvers      — MVO and CVaR portfolio solvers
  backtester   — backtest engine, bootstrap tests, OOS analysis
  visualization — result plots
"""

from .config import UNIVERSES, COMMON, COMMON_OOS, OOS_SPLITS, AF
from .data import load_returns
from .models import (
    RiskFeatureExtractor,
    SINDyRegimeDetector,
    SINDyCovarianceEngine,
    shrink_cov,
)
from .forecasters import (
    SampleMeanForecaster,
    MomentumForecaster,
    ZeroForecaster,
    SINDyRiskMomentumForecaster,
)
from .solvers import solve_mvo, solve_cvar
from .backtester import (
    backtest,
    bootstrap_sharpe_diff,
    run_oos,
    print_oos,
    print_bootstrap,
    max_drawdown,
)
from .visualization import plot_results

__version__ = "8.0"

__all__ = [
    # Config
    "UNIVERSES", "COMMON", "COMMON_OOS", "OOS_SPLITS", "AF",
    # Data
    "load_returns",
    # Models
    "RiskFeatureExtractor", "SINDyRegimeDetector", "SINDyCovarianceEngine", "shrink_cov",
    # Forecasters
    "SampleMeanForecaster", "MomentumForecaster", "ZeroForecaster", "SINDyRiskMomentumForecaster",
    # Solvers
    "solve_mvo", "solve_cvar",
    # Backtester
    "backtest", "bootstrap_sharpe_diff", "run_oos", "print_oos", "print_bootstrap", "max_drawdown",
    # Visualization
    "plot_results",
]
