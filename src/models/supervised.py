"""
supervised.py — Huấn luyện và dự đoán các mô hình regression + bin-classification.
"""
import logging
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("XGBoost chưa được cài đặt.")


class Trainer:
    """
    Huấn luyện các mô hình Regression (Baseline, Linear, Ridge, RF, XGBoost).

    Attributes
    ----------
    models_   : dict {name: fitted_model}
    results_  : dict {name: {MAE, RMSE, R2, time_s}}
    preds_    : dict {name: y_pred array}
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg["models"]
        self.seed   = cfg["project"]["seed"]
        self.models_  = {}
        self.results_ = {}
        self.preds_   = {}

    def fit_all(self, X_train, X_test, y_train, y_test,
                X_train_s=None, X_test_s=None) -> "Trainer":
        """
        Huấn luyện tất cả mô hình.
        X_train_s / X_test_s : phiên bản đã StandardScale (dùng cho Linear/Ridge).
        """
        # Baseline
        self._fit_baseline(y_train, y_test)

        # Linear Regression (cần scaled)
        if X_train_s is not None:
            self._fit_sklearn("Linear", LinearRegression(), X_train_s, X_test_s, y_train, y_test)
            self._fit_sklearn("Ridge",  Ridge(alpha=self.cfg["ridge"]["alpha"]),
                              X_train_s, X_test_s, y_train, y_test)
        else:
            self._fit_sklearn("Linear", LinearRegression(), X_train, X_test, y_train, y_test)
            self._fit_sklearn("Ridge",  Ridge(alpha=self.cfg["ridge"]["alpha"]),
                              X_train, X_test, y_train, y_test)

        # Random Forest (không cần scale)
        rf_params = {k: v for k, v in self.cfg["random_forest"].items()}
        rf_params["random_state"] = self.seed
        self._fit_sklearn("RandomForest", RandomForestRegressor(**rf_params),
                          X_train, X_test, y_train, y_test)

        # XGBoost
        if XGB_OK:
            xgb_params = {k: v for k, v in self.cfg["xgboost"].items()}
            xgb_params["random_state"] = self.seed
            self._fit_sklearn("XGBoost", xgb.XGBRegressor(**xgb_params),
                              X_train, X_test, y_train, y_test)

        self._make_summary()
        return self

    def _fit_baseline(self, y_train, y_test) -> None:
        y_pred = np.full(len(y_test), y_train.mean())
        self.preds_["Baseline"] = y_pred
        self.results_["Baseline"] = self._metrics(y_test, y_pred)
        self.results_["Baseline"]["time_s"] = 0.0
        logger.info(f"Baseline MAE={self.results_['Baseline']['MAE']:,.0f}")

    def _fit_sklearn(self, name, model, X_tr, X_te, y_tr, y_te) -> None:
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred = model.predict(X_te)
        m = self._metrics(y_te, y_pred)
        m["time_s"] = round(elapsed, 2)
        self.models_[name]  = model
        self.preds_[name]   = y_pred
        self.results_[name] = m
        logger.info(f"{name:<15}: MAE={m['MAE']:>10,.0f}  RMSE={m['RMSE']:>10,.0f}  "
                    f"R2={m['R2']:.4f}  ({elapsed:.1f}s)")

    @staticmethod
    def _metrics(y_true, y_pred) -> dict:
        return {
            "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "R2":   round(r2_score(y_true, y_pred), 4),
        }

    def _make_summary(self) -> None:
        self.df_results_ = (
            pd.DataFrame(self.results_).T
            .reset_index()
            .rename(columns={"index": "Model"})
            .sort_values("MAE")
            .reset_index(drop=True)
        )
        logger.info(f"\n{self.df_results_.to_string(index=False)}")

    def best_model_name(self) -> str:
        return self.df_results_.iloc[0]["Model"]

    def feature_importance(self, feature_names: list) -> pd.Series:
        """Lấy feature importance từ RandomForest."""
        rf = self.models_.get("RandomForest")
        if rf is None:
            return pd.Series(dtype=float)
        return pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
