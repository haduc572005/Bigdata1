# Crop Yield Prediction — Đề tài 7

**Môn học:** Dữ liệu lớn, Khai phá dữ liệu  
**Dataset:** [Kaggle: Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)  
**Nhóm:** *(Điền tên thành viên)*

---

## Mô tả đề tài

Xây dựng pipeline khai phá dữ liệu hoàn chỉnh để **dự báo năng suất cây trồng** (hg/ha) dựa trên điều kiện canh tác gồm lượng mưa, nhiệt độ, thuốc trừ sâu, loại cây và vùng địa lý.

---

## Cấu trúc thư mục

```
CROP_YIELD_PROJECT/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml          # Tham số: seed, split, paths, hyperparams
├── data/
│   ├── raw/                 # yield_df.csv (tải từ Kaggle — không commit)
│   └── processed/           # dữ liệu sau tiền xử lý (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py        # Đọc dữ liệu, kiểm tra schema
│   │   └── cleaner.py       # DataCleaner: xử lý NULL, outlier, encoding
│   ├── features/
│   │   └── builder.py       # FeatureBuilder: log transform, encoding, split, scale
│   ├── mining/
│   │   ├── association.py   # AssociationMiner: Apriori + FP-Growth
│   │   └── clustering.py    # ClusterMiner: K-Means, Elbow, Silhouette, PCA
│   ├── models/
│   │   ├── supervised.py    # Trainer: Baseline, Linear, Ridge, RF, XGBoost
│   │   └── forecasting.py   # TimeSeriesAnalyzer: trend, drift, time-split
│   ├── evaluation/
│   │   ├── metrics.py       # MAE, RMSE, R2, F1-bins, rare_zone_analysis
│   │   └── report.py        # Reporter: in bảng, lưu CSV
│   └── visualization/
│       └── plots.py         # Hàm vẽ dùng chung
├── scripts/
│   └── run_pipeline.py      # Chạy toàn bộ pipeline
└── outputs/
    ├── figures/             # Biểu đồ (.png)
    ├── tables/              # Bảng kết quả (.csv)
    ├── models/              # Model đã lưu
    └── reports/             # Báo cáo cuối
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### Cách 1: Chạy toàn bộ pipeline (khuyến nghị)

```bash
# 1. Tải dataset từ Kaggle, đặt file vào:
#    data/raw/yield_df.csv
# 2. Cập nhật đường dẫn trong configs/params.yaml nếu cần
# 3. Chạy pipeline:
python scripts/run_pipeline.py
```

Nếu không có file CSV, pipeline sẽ **tự tạo dữ liệu mô phỏng** (~2,500 bản ghi).

### Cách 2: Chạy từng notebook theo thứ tự

```
01_eda.ipynb               -> EDA & tiền xử lý
02_preprocess_feature.ipynb -> Feature engineering
03_mining_clustering.ipynb  -> Association Rules + Clustering
04_modeling.ipynb           -> Regression models + F1
05_evaluation_report.ipynb  -> Time-series + Kết luận
```

> **Lưu ý:** Chạy notebook từ thư mục `notebooks/` để import đúng `src/`.

---

## Pipeline khai phá dữ liệu

```
Data Source
    └─> Preprocessing (DataCleaner)
            └─> Feature/Representation (FeatureBuilder)
                    ├─> Mining (AssociationMiner: Apriori + FP-Growth)
                    ├─> Mining (ClusterMiner: K-Means K=4)
                    └─> Modeling (Trainer: Baseline/Linear/Ridge/RF/XGB)
                                └─> Evaluation (MAE, RMSE, R2, F1, Rare Zone)
                                            └─> Time Series (Trend, Drift, Split)
```

---

## Kết quả chính

| Mô hình | MAE (hg/ha) | RMSE | R² |
|---------|------------|------|-----|
| Baseline | ~16,500 | ~20,800 | 0.000 |
| Linear Regression | ~14,200 | ~17,900 | ~0.25 |
| Ridge | ~14,100 | ~17,800 | ~0.26 |
| **Random Forest** | **~8,500** | **~11,200** | **~0.71** |
| XGBoost | ~8,200 | ~10,800 | ~0.73 |

---

## Dataset

- **Nguồn:** [Kaggle - Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
- **File cần tải:** `yield_df.csv`
- **Đặt vào:** `data/raw/yield_df.csv`
- **Không commit dữ liệu lên GitHub** (đã có trong `.gitignore`)

---

## Yêu cầu tái lập (Reproducibility)

```bash
pip install -r requirements.txt
# Đặt yield_df.csv vào data/raw/ (hoặc để pipeline tự tạo dữ liệu mô phỏng)
python scripts/run_pipeline.py
# => outputs/figures/ và outputs/tables/ sẽ được tạo tự động
```
