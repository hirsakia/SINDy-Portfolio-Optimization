"""
SINDy Portfolio Optimization — v8
Data loading: yfinance-based return fetcher with winsorisation.
"""

import pandas as pd

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


def load_returns(tickers, start="2005-01-01", end=None, freq="D"):
    """Download and clean daily returns via yfinance."""
    if HAS_YF:
        try:
            raw = yf.download(tickers, start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                top_cols = set(raw.columns.get_level_values(0))
                price_col = "Adj Close" if "Adj Close" in top_cols else "Close"
                px = raw[price_col]
            elif "Adj Close" in raw.columns:
                px = raw["Adj Close"]
            else:
                px = raw["Close"]
            if isinstance(px, pd.Series):
                px = px.to_frame()
            px = px.dropna(how="all").ffill().dropna()
            rets = px.pct_change().dropna()
            lo, hi = rets.quantile(0.005), rets.quantile(0.995)
            rets = rets.clip(lower=lo, upper=hi, axis=1)
            good_cols = rets.columns[rets.notna().mean() > 0.95]
            rets = rets[good_cols].dropna()
            return rets
        except Exception as e:
            print(f"[yfinance fallback] {e}")
    return None
