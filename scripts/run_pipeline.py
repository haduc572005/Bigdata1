"""
run_pipeline.py — Chạy toàn bộ pipeline Crop Yield Prediction.
"""

import pandas as pd
import joblib
import sys
import os
import argparse
import logging

# Thêm root vào sys.path để import src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.loader import load_config, load_raw
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.mining.association import AssociationMiner
from src.mining.clustering import ClusterMiner
from src.models.supervised import Trainer
from src.models.forecasting import TimeSeriesAnalyzer
from src.evaluation.metrics import (
    regression_metrics,
    bin_classification_metrics,
    rare_zone_analysis
)
from src.evaluation.report import Reporter
from src.visualization.plots import (
    plot_quality_check,
    plot_before_after,
    plot_yield_distribution,
    plot_correlation,
    plot_association_rules,
    plot_elbow,
    plot_clusters,
    plot_model_comparison,
    plot_actual_vs_pred,
    plot_rare_zone,
    plot_trend,
    plot_drift
)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config_path="configs/params.yaml"):

    setup_logging()
    logger = logging.getLogger("run_pipeline")

    logger.info("=" * 60)
    logger.info("CROP YIELD PREDICTION PIPELINE — START")
    logger.info("=" * 60)

    # ── 0. Load config
    cfg = load_config(config_path)
    fig_dir = cfg["paths"]["figures"]
    NUM_COLS = cfg["data"]["num_cols"] + ["Yield"]

    # ── 1. Load data
    logger.info("[STEP 1] Load dữ liệu")
    df_raw = load_raw(cfg)

    # ── 2. Quality check
    logger.info("[STEP 2] Quality check")
    plot_quality_check(df_raw, NUM_COLS, fig_dir)

    # ── 3. Clean data
    logger.info("[STEP 3] Cleaning")
    cleaner = DataCleaner(cfg)
    df_clean = cleaner.fit_transform(df_raw)
    plot_before_after(df_raw, df_clean, NUM_COLS, fig_dir)

    # ── 4. EDA
    logger.info("[STEP 4] EDA")
    plot_yield_distribution(df_clean, fig_dir)
    plot_correlation(df_clean, NUM_COLS, fig_dir)

    # ── 5. Feature engineering
    logger.info("[STEP 5] Feature engineering")
    fb = FeatureBuilder(cfg)
    fb.fit_transform(df_clean)

    # save processed data
    os.makedirs(os.path.dirname(cfg["paths"]["processed_data"]), exist_ok=True)
    df_clean.to_parquet(cfg["paths"]["processed_data"], index=False)

    # ── 6. Association mining
    logger.info("[STEP 6] Association rules")
    miner = AssociationMiner(cfg)
    miner.fit(fb.df_disc_)

    reporter = Reporter(cfg)

    reporter.save_table(miner.compare_algorithms(), "association_compare.csv")
    reporter.save_table(miner.top_rules(20), "top_rules.csv")

    plot_association_rules(miner.rules_, miner.rules_high_, fig_dir)

    # ── 7. Clustering
    logger.info("[STEP 7] Clustering")
    clusterer = ClusterMiner(cfg)
    clusterer.fit(fb.df_fe_)

    plot_elbow(clusterer.inertias_, fig_dir)
    plot_clusters(
        clusterer.X_pca_,
        clusterer.labels_,
        clusterer.pca_var_,
        clusterer.profile_,
        fig_dir
    )

    reporter.save_table(clusterer.profile_.reset_index(), "cluster_profile.csv")

    # ── 8. Regression models
    logger.info("[STEP 8] Regression models")

    trainer = Trainer(cfg)

    trainer.fit_all(
        fb.X_train,
        fb.X_test,
        fb.y_train,
        fb.y_test,
        fb.X_train_s,
        fb.X_test_s
    )

    reporter.save_table(trainer.df_results_, "model_comparison.csv")
    plot_model_comparison(trainer.df_results_, fig_dir)

    best_name = trainer.best_model_name()

    # save best model
    os.makedirs(cfg["paths"]["models"], exist_ok=True)

    best_model = trainer.models_[best_name]

    model_path = os.path.join(cfg["paths"]["models"], f"{best_name}.pkl")

    joblib.dump(best_model, model_path)

    logger.info(f"Saved model → {model_path}")

    y_pred_best = trainer.preds_.get(best_name)

    plot_actual_vs_pred(fb.y_test, y_pred_best, best_name, fig_dir)

    # feature importance
    fi = trainer.feature_importance(cfg["models"]["features"])

    if not fi.empty:
        reporter.save_table(
            fi.reset_index().rename(columns={"index": "Feature"}),
            "feature_importance.csv"
        )

    # ── 9. Rare zone
    logger.info("[STEP 9] Rare zone")

    df_zone = rare_zone_analysis(fb.y_test.values, y_pred_best)

    reporter.save_table(df_zone, "rare_zone_analysis.csv")

    plot_rare_zone(df_zone, fig_dir)

    # ── 10. Time-series
    logger.info("[STEP 10] Time-series")

    ts = TimeSeriesAnalyzer(cfg)

    year_mean = ts.analyze_trend(df_clean)

    plot_trend(year_mean, ts.trend_poly_, ts.trend_coef_, fig_dir)

    drift_result = ts.drift_detection(df_clean)

    plot_drift(
        ts.y_early_,
        ts.y_late_,
        drift_result["years_early"],
        drift_result["years_late"],
        drift_result["ks_pval"],
        fig_dir
    )

    ts_result = ts.time_split_train(fb)

    reporter.save_table(pd.DataFrame([ts_result]), "timeseries_split.csv")

    reporter.save_table(pd.DataFrame([drift_result]), "drift_detection.csv")

    # ── 11. Summary
    logger.info("[STEP 11] Summary")

    best_row = trainer.df_results_.iloc[0]

    reporter.print_summary(
        cfg,
        n_raw=len(df_raw),
        n_clean=len(df_clean),
        best_model=best_row["Model"],
        best_mae=best_row["MAE"],
        best_r2=best_row["R2"],
        n_rules=len(miner.rules_),
        k_clusters=cfg["clustering"]["k_best"]
    )

    # ── Save report
    os.makedirs(cfg["paths"]["reports"], exist_ok=True)

    report_path = os.path.join(
        cfg["paths"]["reports"],
        "pipeline_summary.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:

        f.write("CROP YIELD PREDICTION REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Raw samples: {len(df_raw)}\n")
        f.write(f"Clean samples: {len(df_clean)}\n")
        f.write(f"Best model: {best_row['Model']}\n")
        f.write(f"Best MAE: {best_row['MAE']}\n")
        f.write(f"Best R2: {best_row['R2']}\n")
        f.write(f"Association rules: {len(miner.rules_)}\n")
        f.write(f"Clusters: {cfg['clustering']['k_best']}\n")

    logger.info(f"Saved report → {report_path}")

    logger.info("=== PIPELINE HOÀN THÀNH ===")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/params.yaml"
    )

    args = parser.parse_args()

    main(args.config)