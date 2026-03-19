"""
plots.py — Hàm vẽ dùng chung cho toàn bộ project.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

logger = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
          "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def save_fig(fig, fig_dir: str, filename: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


# ─── EDA ────────────────────────────────────────────────────────────────────

def plot_quality_check(df_raw: pd.DataFrame, num_cols: list,
                        fig_dir: str) -> None:
    fig, axes = plt.subplots(2, len(num_cols), figsize=(18, 8))
    for i, col in enumerate(num_cols):
        axes[0, i].boxplot(df_raw[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor="steelblue", alpha=0.7),
                           medianprops=dict(color="red", linewidth=2))
        axes[0, i].set_title(f"Boxplot\n{col}", fontweight="bold", fontsize=8)
        axes[1, i].hist(df_raw[col].dropna(), bins=50,
                        color="coral", edgecolor="white", alpha=0.85)
        axes[1, i].set_title(f"Histogram\n{col}", fontweight="bold", fontsize=8)
    plt.suptitle("Kiểm tra chất lượng dữ liệu — Boxplot & Histogram",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "01_quality_check.png")


def plot_before_after(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                       num_cols: list, fig_dir: str) -> None:
    fig, axes = plt.subplots(2, len(num_cols), figsize=(18, 8))
    for i, col in enumerate(num_cols):
        axes[0, i].hist(df_raw[col].dropna(), bins=50,
                        color="#e74c3c", edgecolor="white", alpha=0.85)
        axes[0, i].set_title(f"TRƯỚC: {col}", fontweight="bold", fontsize=8)
        axes[1, i].hist(df_clean[col], bins=50,
                        color="#2ecc71", edgecolor="white", alpha=0.85)
        axes[1, i].set_title(f"SAU: {col}", fontweight="bold", fontsize=8)
    plt.suptitle("Phân phối Trước (đỏ) vs Sau (xanh) khi làm sạch",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "02_before_after_clean.png")


def plot_yield_distribution(df_clean: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df_clean["Yield"], bins=60, color="steelblue",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(df_clean["Yield"].mean(), color="red", linestyle="--",
                    label=f"Mean={df_clean['Yield'].mean():,.0f}")
    axes[0].axvline(df_clean["Yield"].median(), color="orange", linestyle="--",
                    label=f"Median={df_clean['Yield'].median():,.0f}")
    axes[0].set_title("Phân phối Yield (gốc)", fontweight="bold")
    axes[0].legend()

    axes[1].hist(np.log1p(df_clean["Yield"]), bins=60,
                 color="coral", edgecolor="white", alpha=0.85)
    axes[1].set_title("Phân phối log(Yield+1)", fontweight="bold")
    plt.suptitle("Phân phối biến mục tiêu", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "03_yield_distribution.png")


def plot_correlation(df_clean: pd.DataFrame, num_cols: list,
                      fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Rain", "Pest", "Temp", "Yield"]
    sns.heatmap(df_clean[num_cols].corr(), annot=True, fmt=".3f",
                cmap="coolwarm", square=True, linewidths=0.8,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Ma trận tương quan", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "04_correlation_matrix.png")


# ─── ASSOCIATION RULES ───────────────────────────────────────────────────────

def plot_association_rules(rules_fp: pd.DataFrame, rules_high: pd.DataFrame,
                            fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(rules_fp["support"], rules_fp["confidence"],
                          c=rules_fp["lift"], cmap="RdYlGn", alpha=0.7, s=50)
    plt.colorbar(sc, ax=axes[0], label="Lift")
    axes[0].set_xlabel("Support"); axes[0].set_ylabel("Confidence")
    axes[0].set_title("FP-Growth: Support vs Confidence", fontweight="bold")

    top8 = rules_high.head(8).reset_index(drop=True)
    labels = [", ".join(list(r["antecedents"]))[:35]
              for _, r in top8.iterrows()]
    axes[1].barh(range(len(top8)), top8["lift"],
                 color="steelblue", alpha=0.85)
    axes[1].set_yticks(range(len(top8)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].axvline(1.0, color="red", linestyle="--", label="Lift=1")
    axes[1].set_title("Top rules Yield cao (Lift)", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    save_fig(fig, fig_dir, "05_association_rules.png")


# ─── CLUSTERING ─────────────────────────────────────────────────────────────

def plot_elbow(inertias: dict, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(inertias.keys()), list(inertias.values()),
            "o-", color="steelblue", lw=2, markersize=8)
    ax.set_xlabel("Số cụm K"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Chọn K tối ưu", fontweight="bold")
    ax.set_xticks(list(inertias.keys()))
    plt.tight_layout()
    save_fig(fig, fig_dir, "06_elbow.png")


def plot_clusters(X_pca, labels, pca_var, profile, fig_dir: str) -> None:
    K = len(np.unique(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c in range(K):
        mask = labels == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=f"Cụm {c} (n={mask.sum()})",
                        alpha=0.45, s=18, color=COLORS[c])
    axes[0].set_xlabel(f"PC1 ({pca_var[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca_var[1]:.1%})")
    axes[0].set_title("PCA 2D — Phân cụm", fontweight="bold")
    axes[0].legend(fontsize=8)

    groups = [profile.loc[c, "Yield_avg"] for c in range(K)]
    axes[1].bar([f"Cụm {c}" for c in range(K)], groups,
                color=COLORS[:K], edgecolor="white", alpha=0.85)
    axes[1].set_ylabel("Yield TB (hg/ha)")
    axes[1].set_title("Yield trung bình theo cụm", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "07_clusters.png")


# ─── REGRESSION ──────────────────────────────────────────────────────────────

def plot_model_comparison(df_results: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, c) in zip(axes, [
        ("MAE",  "MAE (thấp=tốt)",  "#e74c3c"),
        ("RMSE", "RMSE (thấp=tốt)", "#e67e22"),
        ("R2",   "R² (cao=tốt)",    "#27ae60"),
    ]):
        d = df_results.sort_values(col, ascending=(col != "R2"))
        bars = ax.barh(d["Model"], d[col], color=c, alpha=0.82)
        for bar, val in zip(bars, d[col]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}" if val > 1 else f"{val:.4f}",
                    va="center", fontsize=9)
        ax.set_title(title, fontweight="bold")
    plt.suptitle("So sánh các mô hình Regression", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "08_model_comparison.png")


def plot_actual_vs_pred(y_test, y_pred, model_name: str, fig_dir: str) -> None:
    residuals = np.array(y_test) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=12, color="steelblue")
    lim = max(y_test.max(), np.max(y_pred))
    axes[0].plot([0, lim], [0, lim], "r--", lw=2, label="Perfect")
    axes[0].set_xlabel("Actual Yield"); axes[0].set_ylabel("Predicted Yield")
    axes[0].set_title("Actual vs Predicted", fontweight="bold")
    axes[0].legend()

    axes[1].hist(residuals, bins=60, color="coral", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="black", lw=2, linestyle="--")
    axes[1].axvline(residuals.mean(), color="blue", linestyle="--",
                    label=f"Mean={residuals.mean():,.0f}")
    axes[1].set_title("Phân phối Residuals", fontweight="bold")
    axes[1].legend()
    plt.suptitle(f"Phân tích lỗi — {model_name}", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "09_residuals.png")


def plot_rare_zone(df_zone: pd.DataFrame, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_z = COLORS[:len(df_zone)]

    bars0 = axes[0].bar(df_zone["Zone"], df_zone["MAE_abs"],
                         color=colors_z, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars0, df_zone["MAE_abs"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 50,
                     f"{val:,.0f}", ha="center", fontsize=8, fontweight="bold")
    axes[0].set_title("MAE tuyệt đối theo vùng Yield", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    bars1 = axes[1].bar(df_zone["Zone"], df_zone["MAE_rel_pct"],
                         color=colors_z, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars1, df_zone["MAE_rel_pct"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2,
                     f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    axes[1].set_title("Lỗi tương đối MAE/Mean (%)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    plt.suptitle("Phân tích lỗi vùng hiếm", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, fig_dir, "10_rare_zone_error.png")


# ─── TIME SERIES ─────────────────────────────────────────────────────────────

def plot_trend(year_mean: pd.DataFrame, trend_poly,
               trend_coef: float, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(year_mean["Year"], year_mean["Yield_mean"],
            "o-", color="steelblue", lw=2, markersize=6, label="Yield TB")
    ax.plot(year_mean["Year"], trend_poly(year_mean["Year"]),
            "r--", lw=2, label=f"Trend ({trend_coef:+.0f} hg/ha/năm)")
    ax.fill_between(year_mean["Year"], year_mean["Yield_mean"],
                    alpha=0.12, color="steelblue")
    ax.set_title("Xu hướng Yield theo năm (Trend Line)", fontweight="bold")
    ax.set_xlabel("Năm"); ax.set_ylabel("Yield TB (hg/ha)")
    ax.legend(); plt.tight_layout()
    save_fig(fig, fig_dir, "11_trend.png")


def plot_drift(y_early, y_late, years_early: str, years_late: str,
               pval: float, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(y_early, bins=50, alpha=0.6, color="steelblue",
            label=f"Năm đầu ({years_early})", edgecolor="white")
    ax.hist(y_late,  bins=50, alpha=0.6, color="coral",
            label=f"Năm sau ({years_late})", edgecolor="white")
    ax.set_title(f"Drift Detection: phân phối Yield theo thời gian\n(KS p={pval:.4f})",
                 fontweight="bold")
    ax.set_xlabel("Yield (hg/ha)"); ax.set_ylabel("Tần suất")
    ax.legend(); plt.tight_layout()
    save_fig(fig, fig_dir, "12_drift_detection.png")
