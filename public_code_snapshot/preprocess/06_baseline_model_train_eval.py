# 06_baseline_model_train_eval.py
# Faster version: trains LR baseline + writes probability GeoTIFF using MANUAL inference (no sklearn per-pixel calls).
# Outputs:
#  - 03_Models/baseline_lr.joblib
#  - 03_Models/baseline_lr_metrics.txt
#  - 04_Results/maps/baseline_lr_proba.tif
#  - 04_Results/maps/baseline_lr_quicklook.png

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# Optional quicklook
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ----------------------------
# USER CONFIG
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

TRAIN_CSV     = PROJECT_ROOT / r"01_Data\Processed\tables\train_points_features.csv"
ALIGNED_DIR   = PROJECT_ROOT / r"01_Data\Processed\rasters_aligned"
HILLSHADE_TIF = ALIGNED_DIR / "hillshade_315_45_30m_aligned.tif"  # basemap only
OUT_TIF       = PROJECT_ROOT / r"04_Results\maps\baseline_lr_proba.tif"
OUT_PNG       = PROJECT_ROOT / r"04_Results\maps\baseline_lr_quicklook.png"
MODEL_DIR     = PROJECT_ROOT / r"03_Models"
MODEL_PKL     = MODEL_DIR / "baseline_lr.joblib"
METRICS_TXT   = MODEL_DIR / "baseline_lr_metrics.txt"

# Exclude hillshade from learning (cartography only)
EXCLUDE_FEATURES = {"hillshade_315_45_30m_aligned"}

# Raster prediction chunk
CHUNK = 2048  # increase for speed if RAM allows (1024/2048/4096)

OUT_NODATA = -9999.0


# ----------------------------
# Helpers
# ----------------------------
def find_raster_for_feature(aligned_dir: Path, feature_name: str) -> Path:
    exact = aligned_dir / f"{feature_name}.tif"
    if exact.exists():
        return exact
    hits = sorted(aligned_dir.glob(f"{feature_name}*.tif"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Raster for feature '{feature_name}' not found in: {aligned_dir}")

def iter_windows(width: int, height: int, chunk: int):
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off, row_off, w, h)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------
# Main
# ----------------------------
def main():
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)

    if "label" not in df.columns:
        raise ValueError("train_points_features.csv must contain a 'label' column (0/1).")

    y = df["label"].astype(int).values

    non_feature_cols = {"lon", "lat", "label", "source"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols and c not in EXCLUDE_FEATURES]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding non-feature columns and hillshade.")

    X = df[feature_cols].copy()

    # Baseline model pipeline
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    # CV metrics (sanity check)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    roc_list, ap_list = [], []
    for tr, te in skf.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        roc_list.append(roc_auc_score(y[te], p))
        ap_list.append(average_precision_score(y[te], p))

    roc_mean, roc_std = float(np.mean(roc_list)), float(np.std(roc_list))
    ap_mean, ap_std   = float(np.mean(ap_list)),  float(np.std(ap_list))

    # Fit final on ALL points
    pipe.fit(X, y)

    # Save
    joblib.dump({"pipeline": pipe, "feature_cols": feature_cols}, MODEL_PKL)
    METRICS_TXT.write_text(
        "\n".join([
            f"Rows: {len(df)} | Positives: {int(y.sum())} | Negatives: {int((y==0).sum())}",
            f"Features used ({len(feature_cols)}): {', '.join(feature_cols)}",
            f"5-fold ROC-AUC: {roc_mean:.4f} ± {roc_std:.4f}",
            f"5-fold AP (PR-AUC): {ap_mean:.4f} ± {ap_std:.4f}",
        ]),
        encoding="utf-8"
    )

    print(f"[OK] Model saved: {MODEL_PKL}")
    print(f"[OK] Metrics saved: {METRICS_TXT}")
    print(f"[INFO] Features used: {feature_cols}")

    # ----------------------------
    # Raster prediction (FAST manual inference)
    # ----------------------------
    # Extract trained params
    imputer: SimpleImputer = pipe.named_steps["imputer"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    med   = imputer.statistics_.astype("float32")  # (n_feat,)
    mean  = scaler.mean_.astype("float32")
    scale = scaler.scale_.astype("float32")
    w     = clf.coef_.reshape(-1).astype("float32")  # (n_feat,)
    b     = float(clf.intercept_[0])

    # Map feature columns -> aligned rasters
    feat_rasters = {f: find_raster_for_feature(ALIGNED_DIR, f) for f in feature_cols}

    # Use first feature raster as reference grid
    ref_path = feat_rasters[feature_cols[0]]
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        width, height = ref.width, ref.height

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=OUT_NODATA,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        driver="GTiff"
    )

    # Speed knobs for GDAL
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    os.environ.setdefault("GDAL_CACHEMAX", "1024")  # MB-ish, depends on GDAL build

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_CACHEMAX=1024):
        srcs = {f: rasterio.open(p) for f, p in feat_rasters.items()}
        try:
            with rasterio.open(OUT_TIF, "w", **out_profile) as dst:
                for idx, win in enumerate(iter_windows(width, height, CHUNK), start=1):

                    cols = []
                    full_mask = None

                    for f in feature_cols:
                        a = srcs[f].read(1, window=win, masked=True)
                        if full_mask is None:
                            full_mask = np.zeros(a.shape, dtype=bool)
                        full_mask |= np.ma.getmaskarray(a)
                        cols.append(a.filled(np.nan).astype("float32"))

                    # Stack -> (h,w,nf)
                    Xw = np.stack(cols, axis=-1)

                    # Impute NaNs (median per feature)
                    for j in range(Xw.shape[-1]):
                        np.nan_to_num(Xw[..., j], copy=False, nan=float(med[j]))

                    # Standardize
                    Xw = (Xw - mean) / scale

                    # Linear score + sigmoid
                    # (h,w,nf) dot (nf,) -> (h,w)
                    lin = np.tensordot(Xw, w, axes=([2], [0])) + b
                    pw = sigmoid(lin).astype("float32")

                    # Apply nodata mask (any feature nodata)
                    if full_mask is not None and full_mask.any():
                        pw[full_mask] = OUT_NODATA

                    dst.write(pw, 1, window=win)

                    if idx % 25 == 0:
                        print(f"[INFO] wrote {idx} windows...")

            print(f"[OK] Prospectivity raster saved: {OUT_TIF}")

        finally:
            for s in srcs.values():
                s.close()

    # ----------------------------
    # Optional quicklook PNG (hillshade + red proba overlay)
    # ----------------------------
    if HAS_PLT and HILLSHADE_TIF.exists():
        with rasterio.open(HILLSHADE_TIF) as hs, rasterio.open(OUT_TIF) as pr:
            scale_ds = max(hs.width // 1600, 1)
            out_w = hs.width // scale_ds
            out_h = hs.height // scale_ds

            hs_img = hs.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
            pr_img = pr.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")

            pr_img = np.where(pr_img == OUT_NODATA, np.nan, pr_img)

        fig = plt.figure(figsize=(12, 8), dpi=200)
        ax = plt.gca()
        ax.imshow(hs_img, cmap="gray")
        im = ax.imshow(pr_img, cmap="Reds", alpha=0.55, vmin=0.0, vmax=1.0)
        ax.set_title("Baseline logistic regression prospectivity (overlay on hillshade)")
        ax.set_axis_off()
        cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("P(copper)")
        plt.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUT_PNG, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Quicklook saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
