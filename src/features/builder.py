"""
builder.py — Feature engineering: log transform, label encoding, discretization,
             train/test split, StandardScaler.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Xây dựng đặc trưng cho mô hình và rời rạc hóa cho Association Rules.

    Attributes
    ----------
    le_area_, le_item_ : LabelEncoder đã fit
    scaler_            : StandardScaler đã fit trên tập train
    features_          : list tên cột feature
    X_train, X_test, y_train, y_test : tập dữ liệu đã chia
    X_train_s, X_test_s              : tập đã chuẩn hóa
    df_disc_           : DataFrame đã rời rạc hóa
    """

    def __init__(self, cfg: dict):
        self.cfg      = cfg
        self.feat_cfg = cfg["features"]
        self.split_cfg = cfg["split"]
        self.disc_cfg  = cfg["discretization"]
        self.le_area_  = LabelEncoder()
        self.le_item_  = LabelEncoder()
        self.scaler_   = StandardScaler()
        self.features_ = cfg["models"]["features"]

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """Chạy toàn bộ feature pipeline."""
        df = df.copy()
        logger.info("=== FeatureBuilder: bắt đầu ===")

        df = self._log_transform(df)
        df = self._label_encode(df)
        df = self._ensure_no_nan(df)   # <-- guard: fill NaN còn sót
        self._split(df)
        self._scale()
        self.df_disc_ = self._discretize(df)

        logger.info(f"Features: {self.features_}")
        logger.info(f"Train: {len(self.X_train):,}  Test: {len(self.X_test):,}")
        return self

    # ------------------------------------------------------------------
    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.feat_cfg.get("log_transform_cols", []):
            new_col = col.replace("_tonnes", "_log").replace("pesticides", "pesticides")
            new_col = "pesticides_log"
            df[new_col] = np.log1p(df[col])
            logger.info(f"Log transform: {col} → {new_col}  "
                        f"skew {df[col].skew():+.3f} → {df[new_col].skew():+.3f}")

        if self.feat_cfg.get("log_transform_target", False):
            df["Yield_log"] = np.log1p(df["Yield"])
        return df

    def _ensure_no_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN còn sót trong feature columns (safety guard)."""
        nan_before = df[self.features_].isnull().sum().sum()
        if nan_before > 0:
            logger.warning(f"Phát hiện {nan_before} NaN trong features — fill bằng median.")
            for col in self.features_:
                if df[col].isnull().any():
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    logger.warning(f"  Filled NaN in {col} with median={fill_val:.4f}")
        nan_after = df[self.features_].isnull().sum().sum()
        logger.info(f"NaN in features: {nan_before} -> {nan_after}")
        return df

    def _label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Area_enc"] = self.le_area_.fit_transform(df["Area"])
        df["Item_enc"] = self.le_item_.fit_transform(df["Item"])
        logger.info(f"Label Encoding: Area {df['Area'].nunique()} nhãn, "
                    f"Item {df['Item'].nunique()} nhãn")
        return df

    def _split(self, df: pd.DataFrame) -> None:
        X = df[self.features_]
        y = df["Yield"]
        seed = self.cfg["project"]["seed"]
        test_size = self.split_cfg["test_size"]
        shuffle   = self.split_cfg["shuffle"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, shuffle=shuffle
        )
        # Lưu df_fe để dùng time-series split sau
        self.df_fe_ = df

    def _scale(self) -> None:
        self.X_train_s = self.scaler_.fit_transform(self.X_train)
        self.X_test_s  = self.scaler_.transform(self.X_test)
        logger.info("StandardScaler fit trên train — transform cả train/test (no leakage)")

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        d = self.disc_cfg
        df_d = df.copy()
        df_d["Rain_cat"]  = pd.cut(df_d["average_rain_fall_mm_per_year"],
                                    bins=d["rain_bins"], labels=d["rain_labels"])
        df_d["Temp_cat"]  = pd.cut(df_d["avg_temp"],
                                    bins=d["temp_bins"], labels=d["temp_labels"])
        df_d["Pest_cat"]  = pd.cut(df_d["pesticides_tonnes"],
                                    bins=d["pest_bins"], labels=d["pest_labels"])
        df_d["Yield_cat"] = pd.cut(df_d["Yield"],
                                    bins=d["yield_bins"], labels=d["yield_labels"])
        logger.info("Rời rạc hóa 4 biến: Rain_cat, Temp_cat, Pest_cat, Yield_cat")
        return df_d

    # ------------------------------------------------------------------
    def get_time_split(self):
        """Chia train/test theo năm (time-series split)."""
        df = self.df_fe_
        years = sorted(df["Year"].unique())
        split_year = years[int(len(years) * self.split_cfg["time_split_ratio"])]

        train_ts = df[df["Year"] <= split_year]
        test_ts  = df[df["Year"] >  split_year]

        X_tr = train_ts[self.features_].copy()
        y_tr = train_ts["Yield"]
        X_te = test_ts[self.features_].copy()
        y_te = test_ts["Yield"]

        # Fill NaN nếu còn sót
        for col in self.features_:
            if X_tr[col].isnull().any():
                X_tr[col] = X_tr[col].fillna(X_tr[col].median())
            if X_te[col].isnull().any():
                X_te[col] = X_te[col].fillna(X_tr[col].median())

        scaler_ts = StandardScaler()
        X_tr_s = scaler_ts.fit_transform(X_tr)
        X_te_s = scaler_ts.transform(X_te)

        logger.info(f"Time-series split: train ≤{split_year} ({len(X_tr):,}), "
                    f"test >{split_year} ({len(X_te):,})")
        return X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s, split_year
