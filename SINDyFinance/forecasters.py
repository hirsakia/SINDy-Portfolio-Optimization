"""
SINDy Portfolio Optimization — v8
Forecasters: return prediction modules for the backtest engine.

Contains:
  • SampleMeanForecaster         — simple historical mean
  • MomentumForecaster           — time-series + cross-sectional momentum blend
  • ZeroForecaster               — zero-return forecast (risk-parity baseline)
  • SINDyRiskMomentumForecaster  — SINDy regime-aware momentum with covariance
"""

import numpy as np

from .models import SINDyRegimeDetector, SINDyCovarianceEngine


class SampleMeanForecaster:
    def __init__(self):
        self._mu = None

    def fit(self, X):
        self._mu = X.mean(axis=0)
        return self

    def predict(self, x_last):
        return self._mu


class MomentumForecaster:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self._mu = None

    def fit(self, X):
        ts_mom = X[-self.lookback:].mean(axis=0)
        xs_mom = ts_mom - ts_mom.mean()
        self._mu = 0.5 * ts_mom + 0.5 * xs_mom
        return self

    def predict(self, x_last):
        return self._mu


class ZeroForecaster:
    def __init__(self):
        self._N = None

    def fit(self, X):
        self._N = X.shape[1]
        return self

    def predict(self, x_last):
        return np.zeros(self._N)


class SINDyRiskMomentumForecaster:
    def __init__(self, mom_lookback=60, regime_threshold_pctile=75,
                 n_components=4, degree=2,
                 equity_dampen=0.3, defensive_boost=2.0,
                 asset_classes=None):
        self.mom_lookback = mom_lookback
        self.regime_threshold_pctile = regime_threshold_pctile
        self.n_components = n_components
        self.degree = degree
        self.equity_dampen = equity_dampen
        self.defensive_boost = defensive_boost
        self.asset_classes = asset_classes or []
        self._regime_detector = None
        self._cov_engine = SINDyCovarianceEngine(asset_classes=self.asset_classes)
        self._last_train = None
        self._stress_score = 0.0
        self._is_stressed = False
        self._vol_direction = 0.0

    def fit(self, X):
        self._last_train = X
        self._regime_detector = SINDyRegimeDetector(
            n_components=self.n_components, degree=self.degree,
            threshold_pctile=self.regime_threshold_pctile,
        ).fit(X)
        self._stress_score, self._is_stressed, self._vol_direction = \
            self._regime_detector.predict_regime(X)
        return self

    def predict(self, x_last):
        X = self._last_train
        N = X.shape[1]
        ts_mom = X[-self.mom_lookback:].mean(axis=0)
        xs_mom = ts_mom - ts_mom.mean()
        mu = 0.5 * ts_mom + 0.5 * xs_mom
        if self._is_stressed and len(self.asset_classes) == N:
            weights = np.ones(N)
            for i, ac in enumerate(self.asset_classes):
                if ac == 'equity':
                    weights[i] = self.equity_dampen
                elif ac in ('bond', 'gold'):
                    weights[i] = self.defensive_boost
                elif ac == 'intl':
                    weights[i] = self.equity_dampen * 1.5
            mu = mu * weights
        return mu

    def predict_cov(self, cov_window=126):
        return self._cov_engine.predict_cov(
            self._last_train,
            stress_score=self._stress_score,
            is_stressed=self._is_stressed,
        )

    def get_regime_info(self):
        return {
            "stress_score": self._stress_score,
            "is_stressed": self._is_stressed,
            "vol_direction": self._vol_direction,
        }

    def get_coefficients(self):
        if self._regime_detector:
            return self._regime_detector.get_coefficients()
        return np.array([])

    def get_feature_names(self):
        if self._regime_detector:
            return self._regime_detector.get_feature_names()
        return []
