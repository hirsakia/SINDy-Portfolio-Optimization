"""
SINDy Portfolio Optimization — v8 (Extended History)
Configuration: universes, common parameters, OOS split dates.

Goal: Resolve the Momentum MVO comparison (p = 0.168 in v7) by adding more data.

Changes from v7:
- History extended to 2005 (was 2010). Adds ~1,250 trading days including the
  2008 financial crisis — the most important stress event in modern markets.
- Two OOS splits: 2015 (~2,500 OOS days) and 2019 (~1,800 OOS days).
  The 2015 split captures COVID + 2022 bear + most of the post-GFC cycle.
- Universes adjusted for ETF availability pre-2007.
  (XLC and XLRE dropped from Equity Sectors; some Global Macro/FI tickers substituted.)
"""

# ═══════════════════════════════════════════════
# UNIVERSES — adjusted for 2005 availability
# ═══════════════════════════════════════════════

UNIVERSES = {
    "US Multi-Asset": {
        "tickers": ["SPY", "QQQ", "XLF", "XLE", "XLV", "TLT", "IEF", "GLD", "VNQ", "EFA", "EEM"],
        "start": "2005-01-01",
        "class_map": {
            'SPY': 'equity', 'QQQ': 'equity', 'XLF': 'equity', 'XLE': 'equity', 'XLV': 'equity',
            'TLT': 'bond', 'IEF': 'bond', 'GLD': 'gold', 'VNQ': 'reit',
            'EFA': 'intl', 'EEM': 'intl',
        },
    },
    "Global Macro": {
        # BWX→AGG (2003), DBC→GLD already in, UUP→SHY (dollar proxy)
        "tickers": ["EFA", "EEM", "FXI", "EWZ", "TLT", "IEF", "GLD", "SHY", "VNQ", "SPY"],
        "start": "2005-01-01",
        "class_map": {
            'EFA': 'equity', 'EEM': 'equity', 'FXI': 'equity', 'EWZ': 'equity',
            'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond',
            'GLD': 'gold', 'VNQ': 'reit', 'SPY': 'equity',
        },
    },
    "Fixed Income+": {
        # HYG→drop, EMB→drop. Add XLU (bond proxy), XLP (defensive)
        "tickers": ["TLT", "IEF", "SHY", "LQD", "TIP", "GLD", "VNQ", "SPY", "XLU", "XLP"],
        "start": "2005-01-01",
        "class_map": {
            'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond', 'LQD': 'bond', 'TIP': 'bond',
            'GLD': 'gold', 'VNQ': 'reit', 'SPY': 'equity',
            'XLU': 'bond', 'XLP': 'bond',
        },
    },
    "Equity Sectors": {
        # Drop XLC (2018) and XLRE (2015). 9 sectors that existed since 1998.
        "tickers": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU"],
        "start": "2005-01-01",
        "class_map": {
            'XLK': 'equity', 'XLF': 'equity', 'XLE': 'equity', 'XLV': 'equity',
            'XLI': 'equity', 'XLY': 'equity',
            'XLP': 'bond', 'XLB': 'equity', 'XLU': 'bond',
        },
    },
}


# ═══════════════════════════════════════════════
# BACKTEST PARAMETERS
# ═══════════════════════════════════════════════

# Annualisation factor (daily frequency)
AF = 252

# Full-sample common parameters
COMMON = dict(
    train_min=504,            # 2 years daily
    rebalance_every=21,       # monthly
    gamma=5.0,
    lam_tc=0.002,
    leverage=1.0,
    lb=0.0,
    ub=0.15,
    realized_cost_bps=3.0,
    cov_window=126,
)

# OOS common parameters (shorter warm-up)
COMMON_OOS = {**COMMON, "train_min": 252}


# ═══════════════════════════════════════════════
# OOS SPLIT DATES
# ═══════════════════════════════════════════════

OOS_SPLITS = ["2015-01-01", "2019-01-01"]
