"""
cleaner.py — Làm sạch dữ liệu: xóa trùng lặp, giá trị âm, xử lý NULL, cắt outlier.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Pipeline làm sạch dữ liệu Crop Yield.

    Attributes
    ----------
    report_ : dict — thống kê số bản ghi bị loại ở mỗi bước
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["preprocessing"]
        self.report_: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chạy toàn bộ pipeline làm sạch, trả về DataFrame sạch."""
        df = df.copy()
        n0 = len(df)
        logger.info(f"=== DataCleaner: bắt đầu với {n0:,} bản ghi ===")

        df = self._drop_duplicates(df)
        df = self._drop_invalid_values(df)
        df = self._fill_nulls(df)
        df = self._drop_null_target(df)
        df = self._clip_outliers(df)

        df = df.reset_index(drop=True)
        self.report_["total_removed"] = n0 - len(df)
        self.report_["pct_removed"] = round((n0 - len(df)) / n0 * 100, 2)
        logger.info(f"=== Làm sạch xong: {n0:,} → {len(df):,} bản ghi "
                    f"(loại {self.report_['total_removed']:,} = {self.report_['pct_removed']}%) ===")
        return df

    # ------------------------------------------------------------------
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        removed = before - len(df)
        self.report_["duplicates_removed"] = removed
        logger.info(f"[1] Xóa trùng lặp: -{removed} bản ghi")
        return df

    def _drop_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["Yield"] > self.cfg["yield_min"]]
        df = df[df["pesticides_tonnes"] >= 0]
        df = df[df["average_rain_fall_mm_per_year"] >= 0]
        removed = before - len(df)
        self.report_["invalid_removed"] = removed
        logger.info(f"[2] Xóa giá trị âm/bất thường: -{removed} bản ghi")
        return df

    def _fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        null_before = df.isnull().sum().sum()
        strategy = self.cfg.get("fill_strategy", "median")

        num_cols = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
        for col in num_cols:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                fill_val = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_val)
                logger.info(f"[3] Fill NULL {col}: {n_null} giá trị → {strategy}={fill_val:.2f}")

        cat_cols = ["Area", "Item"]
        for col in cat_cols:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"[3] Fill NULL {col}: {n_null} giá trị → mode={mode_val}")

        null_after = df.isnull().sum().sum()
        self.report_["nulls_filled"] = int(null_before - null_after)
        return df

    def _drop_null_target(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["Yield"])
        removed = before - len(df)
        self.report_["null_target_removed"] = removed
        if removed:
            logger.info(f"[3b] Xóa NULL Yield: -{removed} bản ghi")
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        lo_y  = self.cfg["yield_outlier_low"]
        hi_y  = self.cfg["yield_outlier_high"]
        lo_p  = self.cfg["pest_outlier_low"]
        hi_p  = self.cfg["pest_outlier_high"]

        q_lo_y, q_hi_y = df["Yield"].quantile(lo_y), df["Yield"].quantile(hi_y)
        q_lo_p, q_hi_p = df["pesticides_tonnes"].quantile(lo_p), df["pesticides_tonnes"].quantile(hi_p)

        df = df[(df["Yield"] >= q_lo_y) & (df["Yield"] <= q_hi_y)]
        df = df[(df["pesticides_tonnes"] >= q_lo_p) & (df["pesticides_tonnes"] <= q_hi_p)]

        removed = before - len(df)
        self.report_["outliers_removed"] = removed
        logger.info(f"[4] Cắt outlier Yield [{lo_y*100:.0f}%-{hi_y*100:.0f}%] "
                    f"& Pesticides [{lo_p*100:.0f}%-{hi_p*100:.0f}%]: -{removed} bản ghi")
        return df

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame([self.report_])
