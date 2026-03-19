"""
forecasting.py — Phân tích chuỗi thời gian: trend, drift detection, time-series split,
                 hệ số biến động, và train RF trên time-split.
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Phân tích chuỗi thời gian cho Crop Yield.

    Attributes
    ----------
    trend_coef_   : hệ số xu hướng (hg/ha/năm)
    ks_stat_, ks_pval_ : kết quả KS-test drift detection
    ts_results_   : dict metrics của RF time-series split
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def analyze_trend(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Tính trend năng suất theo năm."""
        year_mean = (df_clean.groupby("Year")["Yield"]
                     .mean().reset_index()
                     .rename(columns={"Yield": "Yield_mean"}))
        coefs = np.polyfit(year_mean["Year"], year_mean["Yield_mean"], 1)
        self.trend_coef_ = coefs[0]
        self.year_mean_  = year_mean
        self.trend_poly_ = np.poly1d(coefs)
        logger.info(f"Trend tổng thể: {coefs[0]:+.1f} hg/ha/năm")
        return year_mean

    def drift_detection(self, df_clean: pd.DataFrame,
                         alpha: float = None) -> dict:
        """KS-test để phát hiện data drift theo thời gian."""
        if alpha is None:
            alpha = self.cfg["time_series"]["drift_alpha"]

        years = sorted(df_clean["Year"].unique())
        mid   = len(years) // 2
        early = years[:mid]
        late  = years[mid:]

        y_early = df_clean[df_clean["Year"].isin(early)]["Yield"]
        y_late  = df_clean[df_clean["Year"].isin(late)]["Yield"]

        self.ks_stat_, self.ks_pval_ = stats.ks_2samp(y_early, y_late)
        has_drift = self.ks_pval_ < alpha

        result = {
            "years_early":   f"{early[0]}-{early[-1]}",
            "years_late":    f"{late[0]}-{late[-1]}",
            "mean_early":    round(float(y_early.mean()), 1),
            "mean_late":     round(float(y_late.mean()), 1),
            "ks_stat":       round(float(self.ks_stat_), 4),
            "ks_pval":       round(float(self.ks_pval_), 4),
            "has_drift":     has_drift,
            "interpretation": "CÓ drift phân phối đáng kể!" if has_drift
                              else "Không có drift đáng kể.",
        }
        self.drift_result_ = result
        logger.info(f"Drift KS: stat={self.ks_stat_:.4f}, p={self.ks_pval_:.4f} → {result['interpretation']}")
        self.y_early_ = y_early
        self.y_late_  = y_late
        return result

    def cv_analysis(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Hệ số biến động (CV%) theo năm."""
        year_std  = df_clean.groupby("Year")["Yield"].std()
        year_mean = df_clean.groupby("Year")["Yield"].mean()
        cv = (year_std / year_mean * 100).dropna()
        self.cv_series_ = cv
        logger.info(f"CV% trung bình: {cv.mean():.1f}%  min={cv.min():.1f}%  max={cv.max():.1f}%")
        return cv.reset_index().rename(columns={"Yield": "CV_pct"})

    def time_split_train(self, fb,
                          rf_params: dict = None) -> dict:
        """
        Train RF trên time-series split; so sánh với random split.

        Parameters
        ----------
        fb : FeatureBuilder đã fit (để lấy df_fe_, features_)
        """
        X_tr, X_te, y_tr, y_te, _, _, split_year = fb.get_time_split()

        params = rf_params or {}
        params.setdefault("n_estimators", 200)
        params.setdefault("max_depth", 15)
        params.setdefault("min_samples_leaf", 3)
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", self.cfg["project"]["seed"])

        rf = RandomForestRegressor(**params)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)

        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = r2_score(y_te, y_pred)

        self.ts_results_ = {
            "split_year": split_year,
            "n_train":    len(X_tr),
            "n_test":     len(X_te),
            "MAE":        round(mae, 2),
            "RMSE":       round(rmse, 2),
            "R2":         round(r2, 4),
        }
        logger.info(f"Time-split RF (≤{split_year}): MAE={mae:,.0f}  R2={r2:.4f}")
        return self.ts_results_
