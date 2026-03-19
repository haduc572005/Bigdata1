"""
clustering.py — Phân cụm K-Means: Elbow, fit, profiling, PCA visualization.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class ClusterMiner:
    """
    Phân cụm vùng trồng cây theo điều kiện môi trường.

    Attributes
    ----------
    kmeans_   : KMeans model đã fit
    labels_   : nhãn cụm
    profile_  : DataFrame profiling các cụm
    inertias_ : dict {k: inertia} cho Elbow
    sil_score_: silhouette score
    dbi_score_: Davies-Bouldin index
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["clustering"]

    def elbow(self, X_scaled: np.ndarray) -> dict:
        """Tính inertia cho dải K để vẽ Elbow."""
        inertias = {}
        for k in self.cfg["k_range"]:
            km = KMeans(n_clusters=k, random_state=42,
                        n_init=self.cfg["n_init"])
            km.fit(X_scaled)
            inertias[k] = km.inertia_
            logger.debug(f"K={k}: inertia={km.inertia_:,.0f}")
        self.inertias_ = inertias
        return inertias

    def fit(self, df: pd.DataFrame) -> "ClusterMiner":
        feat_cols = self.cfg["features"]
        k         = self.cfg["k_best"]

        self.df_cluster_ = df[feat_cols].dropna().copy()
        self.scaler_c_   = StandardScaler()
        self.X_scaled_   = self.scaler_c_.fit_transform(self.df_cluster_)

        # Elbow
        self.elbow(self.X_scaled_)

        # Fit K-Means
        self.kmeans_ = KMeans(n_clusters=k, random_state=42,
                               n_init=self.cfg["n_init"])
        self.labels_ = self.kmeans_.fit_predict(self.X_scaled_)
        self.df_cluster_["Cluster"] = self.labels_

        # Metrics
        self.sil_score_ = silhouette_score(self.X_scaled_, self.labels_)
        self.dbi_score_ = davies_bouldin_score(self.X_scaled_, self.labels_)
        logger.info(f"K-Means K={k}: Silhouette={self.sil_score_:.4f}, "
                    f"DBI={self.dbi_score_:.4f}")

        # Profiling
        self.profile_ = (
            self.df_cluster_.groupby("Cluster")
            .agg(N=("Yield","count"),
                 Rain_avg=("average_rain_fall_mm_per_year","mean"),
                 Temp_avg=("avg_temp","mean"),
                 Pest_avg=("pesticides_log","mean"),
                 Yield_avg=("Yield","mean"),
                 Yield_std=("Yield","std"))
            .round(1)
        )
        logger.info(f"\n{self.profile_.to_string()}")

        # PCA
        pca = PCA(n_components=2, random_state=42)
        self.X_pca_    = pca.fit_transform(self.X_scaled_)
        self.pca_      = pca
        self.pca_var_  = pca.explained_variance_ratio_
        return self

    def get_cluster_names(self) -> dict:
        """Đặt tên mô tả cho từng cụm dựa trên profiling."""
        names = {}
        for c, row in self.profile_.iterrows():
            if row["Rain_avg"] > 1500:
                prefix = "Mưa nhiều"
            elif row["Rain_avg"] > 800:
                prefix = "Mưa TB"
            else:
                prefix = "Khô hạn"
            if row["Temp_avg"] > 25:
                suffix = "+ Nóng"
            elif row["Temp_avg"] > 15:
                suffix = "+ Ôn hòa"
            else:
                suffix = "+ Mát lạnh"
            names[c] = f"Cụm {c}: {prefix}{suffix}"
        return names
