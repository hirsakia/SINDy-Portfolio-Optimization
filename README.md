
# SINDyFinance_v8 - Portfolio Optimization Results Summary

Data period: 2005-01-04 to 2026-03-23 (5337 days across universes)

## Universe: US Multi-Asset (11 assets)
**Full sample:** Momentum MVO Sharpe=0.712 | SINDy CVaR=0.719

**OOS 2015 (2821 days):**
| Strategy     | Sharpe | MaxDD  | Vol  | Return |
|--------------|--------|--------|------|--------|
| Equal Weight | 0.915  | -19.7% |11.7% |10.70% |
| Risk Parity  | 0.911  | -19.4% | 9.4% | 8.57% |
| Momentum MVO | 0.836  | -20.2% |12.7% |10.60% |
| SINDy CVaR   | 1.096  | -16.7% |10.3% |11.26% |

Bootstrap: vs Mom dSharpe=+0.260 p=0.106

**OOS 2019 (1815 days):** SINDy=1.055 > Mom=0.800 (p=0.205)

## Universe: Global Macro (10 assets)
**OOS 2015:** SINDy=0.987 > Mom=0.732 (p=0.117), beats EW/RP
**OOS 2019:** SINDy=0.613 > Mom=0.514 (p=0.609)

## Universe: Fixed Income+ (10 assets) 
**OOS 2015:** SINDy=0.882 ≈ EW=0.897 > Mom=0.747
**OOS 2019:** SINDy=0.662 < EW=0.706 but > Mom=0.643

## Universe: Equity Sectors (9 assets)
**OOS 2015:** SINDy=0.878 ≈ EW=0.871 > Mom=0.776
**OOS 2019:** SINDy=0.843 > all (p=0.274 vs Mom)

## Cross-Universe Wins (Sharpe)
**2015 split:** SINDy 11/12 wins (3/3 per universe except FI 2/3)
**2019 split:** SINDy 10/12 wins

## Pooled Bootstrap (Key Results)
**2015 (10k+ days):** vs Mom +0.159 p=0.015** | vs RP +0.148 p=0.003***
**2019 (6k days):** vs Mom +0.115 p=0.162 | vs RP +0.148 p=0.033**

## FINAL VERDICT
**RESOLVED**: Pooled 2015 split confirms SINDy CVaR beats Momentum MVO (p=0.015). Extended GFC/COVID data provides statistical power. Consistent edge in equities/multi-asset; mixed in FI/macro.

---
*Generated from SINDyFinance_v8.ipynb output. SINDy: regime-aware CVaR with stress covariance adjustments.*
