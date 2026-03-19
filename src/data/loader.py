"""
loader.py — Đọc dữ liệu, chuẩn hoá tên cột, kiểm tra schema.

Dataset Kaggle thật (yield_df.csv) có header:
    ,Area,Item,Year,hg/ha_yield,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp
=> cột đầu là index số (0,1,2...) cần bỏ, target = "hg/ha_yield"
"""
import os
import logging
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "Area", "Item", "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes", "avg_temp", "Yield"
]

# Tất cả biến thể tên cột Kaggle -> tên chuẩn nội bộ
_RENAME = {
    "hg/ha_yield":  "Yield",
    "hg/ha yield":  "Yield",
    "hg_ha_yield":  "Yield",
    "yield":        "Yield",
    "Value":        "Yield",
    "area":         "Area",
    "item":         "Item",
    "year":         "Year",
}


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    path = cfg["paths"]["raw_data"]
    if os.path.exists(path):
        df = _read_csv_smart(path)
        logger.info(f"Loaded '{path}': {df.shape}")
    else:
        logger.warning(f"File '{path}' not found — generating simulated data.")
        df = _simulate(seed=cfg["project"]["seed"])

    df = _normalize_columns(df)
    _validate_schema(df)
    return df


def _read_csv_smart(path: str) -> pd.DataFrame:
    """Đọc CSV, tự phát hiện & bỏ cột index số thừa ở đầu."""
    df = pd.read_csv(path)
    first_col = df.columns[0]
    # Nếu cột đầu là dãy số nguyên liên tiếp (0,1,2,...) -> index thừa
    try:
        vals = pd.to_numeric(df[first_col], errors="coerce")
        n_valid = vals.notna().sum()
        if n_valid > len(df) * 0.9:
            is_seq = (vals.dropna().astype(int).sort_values().reset_index(drop=True)
                      == pd.Series(range(n_valid))).mean() > 0.9
            if is_seq:
                df = df.drop(columns=[first_col])
                logger.info(f"Dropped index column: '{first_col}'")
    except Exception:
        pass
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces, rename theo _RENAME, drop Unnamed cols."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in _RENAME and _RENAME[col] != col:
            rename_map[col] = _RENAME[col]
        elif col.lower() in _RENAME and col not in REQUIRED_COLS:
            target = _RENAME[col.lower()]
            if col != target:
                rename_map[col] = target

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed: {rename_map}")

    drop_cols = [c for c in df.columns
                 if str(c).lower().startswith("unnamed") or str(c).strip() == ""]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    logger.info(f"Columns: {list(df.columns)}")
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        suggestions = {c: [x for x in df.columns if c.lower() in x.lower()]
                       for c in missing}
        raise ValueError(
            f"\n[Schema Error] Missing: {missing}"
            f"\nAvailable: {list(df.columns)}"
            f"\nSuggestions: {suggestions}"
            f"\nTip: Kaggle CSV dùng 'hg/ha_yield' — đã auto-map thành 'Yield'."
            f"\n     Nếu vẫn lỗi, thêm tên cột thực vào _RENAME trong loader.py"
        )
    # Ép kiểu số
    for col in ["Year", "average_rain_fall_mm_per_year",
                "pesticides_tonnes", "avg_temp", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"Schema OK — {df.shape[0]:,} rows x {df.shape[1]} cols")


def _simulate(seed: int = 42, n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    AREAS = ["Albania","Algeria","Angola","Argentina","Australia",
             "Brazil","China","Egypt","France","Germany",
             "India","Indonesia","Italy","Japan","Kenya",
             "Mexico","Nigeria","Pakistan","Thailand","USA","Vietnam"]
    ITEMS = ["Maize","Potatoes","Rice, paddy","Sorghum","Soybeans",
             "Sweet potatoes","Wheat","Yams","Cassava","Sugarcane"]

    rain   = rng.uniform(200, 3000, n)
    pest   = rng.exponential(50000, n)
    temp   = rng.uniform(5, 35, n)
    yield_ = (10000 + rain * 5 + pest * 0.01
              - np.abs(temp - 20) * 300
              + rng.normal(0, 5000, n)).clip(500, 120000)

    idx_null = rng.choice(n, 80, replace=False)
    rain[idx_null[:20]]   = np.nan
    pest[idx_null[20:40]] = np.nan
    temp[idx_null[40:60]] = np.nan
    yield_[idx_null[60:]] = np.nan

    df = pd.DataFrame({
        "Area":  rng.choice(AREAS, n),
        "Item":  rng.choice(ITEMS, n),
        "Year":  rng.integers(1990, 2014, n),
        "average_rain_fall_mm_per_year": rain,
        "pesticides_tonnes": pest,
        "avg_temp": temp,
        "Yield": yield_,
    })
    df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)
    df.loc[df.sample(10, random_state=2).index, "Yield"] = -999
    df.loc[df.sample(5,  random_state=3).index, "pesticides_tonnes"] = -1
    logger.info(f"Simulated data: {df.shape}")
    return df
