"""
metrics.py — Tính toán các metrics: MAE, RMSE, R², F1 (bin), phân tích lỗi vùng hiếm.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, f1_score, classification_report,
                              confusion_matrix)

logger = logging.getLogger(__name__)


def regression_metrics(y_true, y_pred, name: str = "") -> dict:
    """MAE, RMSE, R² cho regression."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    if name:
        logger.info(f"{name}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}")
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def bin_classification_metrics(y_true, y_pred_cont, cfg: dict) -> dict:
    """
    Chuyển giá trị liên tục thành nhóm và tính F1-macro.
    Dùng cho yêu cầu 'F1 + phân tích lỗi vùng hiếm' trong rubric.
    """
    bins   = cfg["yield_bins_classification"]["bins"]
    labels = cfg["yield_bins_classification"]["labels"]

    y_true_bin = pd.cut(y_true, bins=bins, labels=labels).astype(str)
    y_pred_bin = pd.cut(y_pred_cont, bins=bins, labels=labels).astype(str)

    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    f1_per   = f1_score(y_true_bin, y_pred_bin, average=None,
                        labels=labels, zero_division=0)
    cm       = confusion_matrix(y_true_bin, y_pred_bin, labels=labels)

    result = {
        "F1_macro":   round(f1_macro, 4),
        "F1_per_class": dict(zip(labels, f1_per.round(4))),
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
        "classification_report": classification_report(
            y_true_bin, y_pred_bin, labels=labels, zero_division=0)
    }
    logger.info(f"F1-macro (bins): {f1_macro:.4f}")
    logger.info(f"F1 per class: {result['F1_per_class']}")
    return result


def rare_zone_analysis(y_true, y_pred, n_zones: int = 6) -> pd.DataFrame:
    """
    Phân tích MAE và lỗi tương đối theo vùng percentile Yield.

    Returns
    -------
    DataFrame với: Zone, N, Yield_mean, MAE_abs, MAE_rel_pct
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pcts   = np.linspace(0, 100, n_zones + 1)
    bins   = [np.percentile(y_true, p) for p in pcts]
    bins[0]  -= 1
    bins[-1] += 1

    rows = []
    for i in range(n_zones):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() == 0:
            continue
        yt = y_true[mask]; yp = y_pred[mask]
        mae_abs = mean_absolute_error(yt, yp)
        mae_rel = mae_abs / yt.mean() * 100
        rows.append({
            "Zone":        f"{pcts[i]:.0f}-{pcts[i+1]:.0f}%ile",
            "N":           int(mask.sum()),
            "Yield_mean":  round(float(yt.mean()), 1),
            "MAE_abs":     round(mae_abs, 1),
            "MAE_rel_pct": round(mae_rel, 2),
        })

    df = pd.DataFrame(rows)
    logger.info(f"\n{df.to_string(index=False)}")
    return df


def residual_analysis(y_true, y_pred) -> pd.DataFrame:
    """Tính residuals và thống kê mô tả."""
    residuals = np.array(y_true) - np.array(y_pred)
    return pd.DataFrame({
        "residual": residuals,
        "abs_error": np.abs(residuals),
        "y_true": y_true,
        "y_pred": y_pred,
    })
