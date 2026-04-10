# 09_normalize_continuous_rasters.py
# Normalize continuous rasters to z-scores (mean/std) using fast sampling for stats.
# Outputs:
#  - 01_Data/Processed/rasters_norm/<name>_z.tif
#  - 01_Data/Processed/tables/raster_norm_stats.csv
#  - 01_Data/Processed/rasters_norm/quicklooks/<name>_z.png  (optional)

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

ALIGNED_DIR  = PROJECT_ROOT / r"01_Data\Processed\rasters_aligned"
FEATURE_DIR  = PROJECT_ROOT / r"01_Data\Processed\rasters_features"  # dist_to_fault + lithology here
OUT_DIR      = PROJECT_ROOT / r"01_Data\Processed\rasters_norm"
STATS_CSV    = PROJECT_ROOT / r"01_Data\Processed\tables\raster_norm_stats.csv"
QL_DIR       = OUT_DIR / "quicklooks"

OUT_NODATA = -9999.0
CHUNK = 1024

# Exclude from normalization (cartography or categorical)
EXCLUDE = {
    "hillshade_315_45_30m_aligned",     # basemap only
    "lithology_code_aligned",           # categorical classes
}

# If True: compute stats by sampling windows (fast). If False: full pass (slow).
FAST_STATS = True
N_SAMPLE_WINDOWS = 250  # adjust 150–500

# Clip z-scores to avoid extreme values
Z_CLIP = 6.0

# ----------------------------
# HELPERS
# ----------------------------
def iter_windows(width: int, height: int, chunk: int):
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off, row_off, w, h)

def list_input_rasters():
    rasters = []
    for d in [ALIGNED_DIR, FEATURE_DIR]:
        if d.exists():
            rasters.extend(sorted(d.glob("*.tif")))
    # de-dup by stem
    uniq = {}
    for p in rasters:
        uniq[p.stem] = p
    out = []
    for stem, p in sorted(uniq.items()):
        if stem in EXCLUDE:
            continue
        out.append(p)
    return out

def raster_stats_fast(src: rasterio.DatasetReader, n_windows: int):
    rng = np.random.default_rng(7)
    W, H = src.width, src.height

    # sample random windows
    means = []
    sq_means = []
    counts = []

    for _ in range(n_windows):
        w = min(CHUNK, W)
        h = min(CHUNK, H)
        col_off = int(rng.integers(0, max(1, W - w)))
        row_off = int(rng.integers(0, max(1, H - h)))
        win = Window(col_off, row_off, w, h)

        a = src.read(1, window=win, masked=True)
        if a.mask.all():
            continue
        x = a.compressed().astype("float64")
        if x.size == 0:
            continue
        means.append(x.mean())
        sq_means.append((x * x).mean())
        counts.append(x.size)

    if not counts:
        return np.nan, np.nan

    # weighted combine
    counts = np.array(counts, dtype="float64")
    means = np.array(means, dtype="float64")
    sq_means = np.array(sq_means, dtype="float64")

    wmean = (means * counts).sum() / counts.sum()
    wmean_sq = (sq_means * counts).sum() / counts.sum()
    var = max(0.0, wmean_sq - wmean * wmean)
    std = float(np.sqrt(var)) if var > 0 else 0.0
    return float(wmean), std

def raster_stats_full(src: rasterio.DatasetReader):
    # streaming Welford
    n = 0
    mean = 0.0
    M2 = 0.0
    for win in iter_windows(src.width, src.height, CHUNK):
        a = src.read(1, window=win, masked=True)
        if a.mask.all():
            continue
        x = a.compressed().astype("float64")
        if x.size == 0:
            continue
        for v in x:
            n += 1
            delta = v - mean
            mean += delta / n
            delta2 = v - mean
            M2 += delta * delta2
    if n < 2:
        return float(mean), 0.0
    var = M2 / (n - 1)
    return float(mean), float(np.sqrt(var))

def quicklook_png(tif_path: Path, png_path: Path, title: str):
    if not HAS_PLT:
        return
    with rasterio.open(tif_path) as src:
        scale = max(src.width // 1600, 1)
        out_w = max(1, src.width // scale)
        out_h = max(1, src.height // scale)

        arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        left, bottom, right, top = src.bounds

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8), dpi=200)
    ax = plt.gca()
    im = ax.imshow(arr, extent=[left, right, bottom, top], origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Longitude" if src.crs and src.crs.is_geographic else "X")
    ax.set_ylabel("Latitude" if src.crs and src.crs.is_geographic else "Y")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("z-score")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# MAIN
# ----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

    inputs = list_input_rasters()
    if not inputs:
        raise FileNotFoundError("No input rasters found in rasters_aligned / rasters_features.")

    stats_rows = []

    for in_path in inputs:
        name = in_path.stem
        out_path = OUT_DIR / f"{name}_z.tif"

        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            mean, std = raster_stats_fast(src, N_SAMPLE_WINDOWS) if FAST_STATS else raster_stats_full(src)

            if not np.isfinite(mean) or std == 0.0:
                print(f"[SKIP] {name}: cannot compute valid stats (mean={mean}, std={std})")
                continue

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

            print(f"[INFO] {name}: mean={mean:.6g}, std={std:.6g} -> {out_path.name}")

            with rasterio.open(out_path, "w", **out_profile) as dst:
                for win in iter_windows(src.width, src.height, CHUNK):
                    a = src.read(1, window=win, masked=True)

                    if a.mask.all():
                        dst.write(np.full((win.height, win.width), OUT_NODATA, dtype="float32"), 1, window=win)
                        continue

                    x = a.filled(np.nan).astype("float32")
                    z = (x - mean) / std
                    z = np.clip(z, -Z_CLIP, Z_CLIP)

                    # set nodata where any nan
                    z = np.where(np.isnan(z), OUT_NODATA, z).astype("float32")
                    dst.write(z, 1, window=win)

        stats_rows.append({"raster": name, "mean": mean, "std": std, "out_z": str(out_path)})

        # optional quicklook
        if HAS_PLT:
            quicklook_png(out_path, QL_DIR / f"{name}_z.png", f"{name} (z-score, downsampled)")

    pd.DataFrame(stats_rows).to_csv(STATS_CSV, index=False)
    print(f"[OK] Stats saved: {STATS_CSV}")
    print(f"[OK] Normalized rasters in: {OUT_DIR}")

if __name__ == "__main__":
    main()
