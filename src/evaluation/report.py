"""
report.py — Tổng hợp bảng/biểu đồ kết quả, lưu ra outputs/.
"""
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class Reporter:
    """Tổng hợp và lưu kết quả thực nghiệm."""

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.out_dir   = cfg["paths"]["outputs"]
        self.tbl_dir   = cfg["paths"]["tables"]
        self.fig_dir   = cfg["paths"]["figures"]
        os.makedirs(self.tbl_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def save_table(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.tbl_dir, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved table: {path}")
        return path

    def print_model_comparison(self, df_results: pd.DataFrame) -> None:
        print("\n" + "="*65)
        print("  SO SÁNH CÁC MÔ HÌNH REGRESSION")
        print("="*65)
        print(df_results.to_string(index=False))
        print("="*65)

    def print_top_rules(self, rules_high: pd.DataFrame, n: int = 5) -> None:
        print(f"\nTOP {n} LUẬT DỰ ĐOÁN YIELD CAO:")
        for i, (_, r) in enumerate(rules_high.head(n).iterrows()):
            print(f"  {i+1}. NẾU {set(r['antecedents'])}")
            print(f"     THÌ {set(r['consequents'])}")
            print(f"     Conf={r['confidence']:.3f} | Lift={r['lift']:.3f}\n")

    def print_cluster_profile(self, profile: pd.DataFrame,
                               names: dict = None) -> None:
        print("\nPROFILE CÁC CỤM (K-Means):")
        df = profile.copy()
        if names:
            df.index = [names.get(i, f"Cụm {i}") for i in df.index]
        print(df.to_string())

    def print_rare_zone(self, df_zone: pd.DataFrame) -> None:
        print("\nPHÂN TÍCH LỖI VÙNG HIẾM (theo %ile Yield):")
        print(df_zone.to_string(index=False))

    def print_time_series(self, ts_result: dict,
                           random_split_mae: float) -> None:
        print("\nSO SÁNH RANDOM SPLIT vs TIME-SERIES SPLIT:")
        print(f"  Random split  MAE = {random_split_mae:,.0f} hg/ha")
        print(f"  Time-series split MAE = {ts_result['MAE']:,.0f} hg/ha  "
              f"R2={ts_result['R2']:.4f}")
        diff = ts_result["MAE"] - random_split_mae
        print(f"  Chênh lệch: {diff:+,.0f} hg/ha "
              f"({'Time-split kém hơn → có thể overfit theo năm' if diff > 0 else 'Ổn định'})")

    def print_summary(self, cfg, n_raw, n_clean,
                       best_model, best_mae, best_r2,
                       n_rules, k_clusters) -> None:
        print("\n" + "="*70)
        print("  KẾT LUẬN TỔNG HỢP — DỰ BÁO NĂNG SUẤT CÂY TRỒNG")
        print("="*70)
        print(f"  Dữ liệu gốc     : {n_raw:,} bản ghi")
        print(f"  Sau làm sạch    : {n_clean:,} bản ghi")
        print(f"  Mô hình tốt nhất: {best_model}")
        print(f"    MAE  = {best_mae:,.0f} hg/ha")
        print(f"    R²   = {best_r2:.4f}")
        print(f"  Association Rules: {n_rules} luật (FP-Growth)")
        print(f"  Clustering      : K={k_clusters} cụm")
        print("="*70)
