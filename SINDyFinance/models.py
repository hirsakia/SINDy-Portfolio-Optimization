"""
SINDy Portfolio Optimization — v8
Risk models: feature extraction, SINDy regime detection, covariance engine.

Contains:
  • RiskFeatureExtractor  — rolling risk features (vol, correlation, dispersion, …)
  • SINDyRegimeDetector   — SINDy-based regime classification via PCA + polynomial library
  • SINDyCovarianceEngine — asset-class-aware stress-adjusted covariance prediction (v5)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.covariance import LedoitWolf


# ─────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────

def shrink_cov(X_window):
    """Ledoit-Wolf shrinkage covariance."""
    if X_window.shape[0] > 5:
        try:
            return LedoitWolf().fit(X_window).covariance_
        except Exception:
            pass
    return np.cov(X_window.T)


# ─────────────────────────────────────────────────
# Risk Feature Extractor
# ─────────────────────────────────────────────────

class RiskFeatureExtractor:
    def __init__(self, short_window=20, long_window=60):
        self.short_window = short_window
        self.long_window = long_window

    def extract(self, X):
        T, N = X.shape
        df = pd.DataFrame(X)
        features = {}

        vol_short = df.rolling(self.short_window).std()
        vol_long = df.rolling(self.long_window).std()
        features['vol_short_mean'] = vol_short.mean(axis=1).values
        features['vol_long_mean'] = vol_long.mean(axis=1).values
        features['vol_ratio'] = features['vol_short_mean'] / (features['vol_long_mean'] + 1e-12)

        # Equity vs other vol — use asset_classes if available
        # For now, just use first half vs second half as a proxy
        half = N // 2
        if N > 2:
            first_vol = vol_short.iloc[:, :half].mean(axis=1).values
            second_vol = vol_short.iloc[:, half:].mean(axis=1).values
            features['group_vol_ratio'] = first_vol / (second_vol + 1e-12)

        corr_series = np.full(T, np.nan)
        for t in range(self.long_window, T, 5):
            window = X[t - self.long_window:t]
            c = np.corrcoef(window.T)
            mask = ~np.eye(N, dtype=bool)
            corr_series[t] = c[mask].mean()
        corr_series = pd.Series(corr_series).ffill().bfill().values
        features['avg_corr'] = corr_series

        # Stock-bond correlation — computed dynamically if we have enough assets
        if N > 4:
            sb_corr = np.full(T, np.nan)
            for t in range(self.long_window, T, 5):
                window = X[t - self.long_window:t]
                # Use first half vs second half as proxy groups
                g1 = window[:, :half].mean(axis=1)
                g2 = window[:, half:].mean(axis=1)
                if len(g1) > 5:
                    sb_corr[t] = np.corrcoef(g1, g2)[0, 1]
            sb_corr = pd.Series(sb_corr).ffill().bfill().values
            features['cross_group_corr'] = sb_corr

        features['dispersion'] = df.std(axis=1).values
        vol_short_s = pd.Series(features['vol_short_mean'])
        features['vol_of_vol'] = vol_short_s.rolling(self.short_window).std().values

        down_corr = np.full(T, np.nan)
        for t in range(self.long_window, T, 5):
            window = X[t - self.long_window:t]
            mkt = window.mean(axis=1)
            down_mask = mkt < 0
            if down_mask.sum() > 5:
                c = np.corrcoef(window[down_mask].T)
                mask_d = ~np.eye(N, dtype=bool)
                down_corr[t] = c[mask_d].mean()
            else:
                down_corr[t] = corr_series[t]
        down_corr = pd.Series(down_corr).ffill().bfill().values
        features['down_corr'] = down_corr
        features['corr_asymmetry'] = down_corr - corr_series

        feat_matrix = np.column_stack(list(features.values()))
        feat_names = list(features.keys())
        offset = self.long_window + self.short_window
        valid = np.nan_to_num(feat_matrix[offset:], nan=0.0)
        return valid, offset, feat_names


# ─────────────────────────────────────────────────
# SINDy Regime Detector
# ─────────────────────────────────────────────────

class SINDyRegimeDetector:
    def __init__(self, n_components=4, degree=2, threshold_pctile=75):
        self.n_components = n_components
        self.degree = degree
        self.threshold_pctile = threshold_pctile
        self.pca = None
        self.poly = None
        self.models = []
        self._feat_mean = None
        self._feat_std = None
        self._stress_threshold = None
        self.feat_extractor = RiskFeatureExtractor()

    def fit(self, X_raw):
        feats, offset, self.feat_names = self.feat_extractor.extract(X_raw)
        if len(feats) < 100:
            self.models = None
            return self
        self._feat_mean = feats.mean(axis=0)
        self._feat_std = feats.std(axis=0) + 1e-12
        feats_sc = (feats - self._feat_mean) / self._feat_std
        n_comp = min(self.n_components, feats_sc.shape[1], len(feats_sc) - 2)
        self.pca = PCA(n_components=n_comp).fit(feats_sc)
        Z = self.pca.transform(feats_sc)
        Z_in, Z_out = Z[:-1], Z[1:]
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        Phi = self.poly.fit_transform(Z_in)
        self.models = []
        for j in range(n_comp):
            lasso = LassoCV(alphas=np.logspace(-5, -1, 20), cv=5,
                            max_iter=5000, n_jobs=None)
            lasso.fit(Phi, Z_out[:, j])
            self.models.append(lasso)
        stress_scores = self._compute_stress_scores(Z)
        self._stress_threshold = np.percentile(stress_scores, self.threshold_pctile)
        return self

    def _compute_stress_scores(self, Z):
        Phi = self.poly.transform(Z[:-1])
        Z_pred = np.column_stack([m.predict(Phi) for m in self.models])
        delta = Z_pred - Z[:-1]
        stress = np.sqrt(np.sum(delta ** 2, axis=1))
        return np.concatenate([[0], stress])

    def predict_regime(self, X_raw_recent):
        if self.models is None:
            return 0.0, False, 0.0
        feats, offset, _ = self.feat_extractor.extract(X_raw_recent)
        if len(feats) < 5:
            return 0.0, False, 0.0
        feats_sc = (feats - self._feat_mean) / self._feat_std
        Z = self.pca.transform(feats_sc)
        stress_scores = self._compute_stress_scores(Z)
        current_stress = stress_scores[-1]
        is_stressed = current_stress > self._stress_threshold
        Phi_last = self.poly.transform(Z[-1:])
        z_pred = np.array([m.predict(Phi_last)[0] for m in self.models])
        vol_direction = z_pred[0] - Z[-1, 0]
        return current_stress, is_stressed, vol_direction

    def get_coefficients(self):
        if self.models is None:
            return np.array([])
        return np.column_stack([m.coef_ for m in self.models])

    def get_feature_names(self):
        if self.poly is None:
            return []
        return self.poly.get_feature_names_out()


# ─────────────────────────────────────────────────
# SINDy Covariance Engine (v5)
# ─────────────────────────────────────────────────

class SINDyCovarianceEngine:
    """
    v5: Smarter stress adjustment.
    - Inflate WITHIN-equity correlations (they converge to 1 in crises)
    - PRESERVE equity-bond negative correlation (flight to quality)
    - Boost predicted vol for equities, reduce for bonds
    """
    def __init__(self, vol_window=20, cov_window=126,
                 stress_eq_corr_boost=0.4, asset_classes=None):
        self.vol_window = vol_window
        self.cov_window = cov_window
        self.stress_eq_corr_boost = stress_eq_corr_boost
        self.asset_classes = asset_classes or []

    def predict_cov(self, X_train, stress_score=0.0, is_stressed=False):
        T, N = X_train.shape
        vol_series = pd.DataFrame(X_train).rolling(self.vol_window).std().dropna().values
        predicted_vols = np.zeros(N)

        for j in range(N):
            y = vol_series[:, j]
            if len(y) < 30:
                predicted_vols[j] = X_train[:, j].std()
                continue
            n_lags = min(4, len(y) // 4)
            X_ar = np.column_stack([y[i:len(y) - n_lags + i] for i in range(n_lags)])
            y_target = y[n_lags:]
            if len(y_target) < 10:
                predicted_vols[j] = y[-1]
                continue
            ridge = RidgeCV(alphas=np.logspace(-3, 1, 10)).fit(X_ar, y_target)
            predicted_vols[j] = max(ridge.predict(y[-n_lags:].reshape(1, -1))[0], 1e-6)

        recent = X_train[-self.cov_window:]
        lw_cov = shrink_cov(recent) + 1e-6 * np.eye(N)
        lw_std = np.sqrt(np.diag(lw_cov))
        corr = lw_cov / np.outer(lw_std, lw_std)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1, 1)

        if is_stressed and len(self.asset_classes) == N:
            boost = self.stress_eq_corr_boost * min(stress_score / 2.0, 1.0)
            for i in range(N):
                for j in range(i + 1, N):
                    ac_i, ac_j = self.asset_classes[i], self.asset_classes[j]
                    if ac_i == 'equity' and ac_j == 'equity':
                        # Equities converge: push correlation toward 1
                        corr[i, j] += boost * (1.0 - corr[i, j])
                        corr[j, i] = corr[i, j]
                    elif (ac_i == 'equity' and ac_j in ('bond', 'gold')) or \
                         (ac_j == 'equity' and ac_i in ('bond', 'gold')):
                        # Equity-bond/gold: push MORE negative (flight to quality)
                        corr[i, j] -= boost * 0.5 * (1.0 + corr[i, j])
                        corr[j, i] = corr[i, j]
                    # else: leave other pairs alone

            corr = np.clip(corr, -1, 1)

            # Stress vol adjustment: equities up, bonds down
            for i, ac in enumerate(self.asset_classes):
                if ac == 'equity':
                    predicted_vols[i] *= (1.0 + boost * 0.5)
                elif ac in ('bond', 'gold'):
                    predicted_vols[i] *= max(1.0 - boost * 0.2, 0.5)

        D = np.diag(predicted_vols)
        pred_cov = D @ corr @ D
        pred_cov = (pred_cov + pred_cov.T) / 2
        eigvals = np.linalg.eigvalsh(pred_cov)
        if eigvals.min() < 0:
            pred_cov += (abs(eigvals.min()) + 1e-6) * np.eye(N)
        return pred_cov
