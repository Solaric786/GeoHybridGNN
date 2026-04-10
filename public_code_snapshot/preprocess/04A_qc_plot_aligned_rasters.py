# 04A_qc_plot_aligned_rasters.py
# QC plots for aligned rasters (GIS-style):
# - Hillshade -> grayscale + 2–98% stretch (no equalize)
# - Other rasters -> optional histogram equalize + "turbo" colormap

from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ----------------------------
# PATHS
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")
DEM_REF = PROJECT_ROOT / r"01_Data\Raw\rasters\dem_ref.tif"
ALIGNED_DIR = PROJECT_ROOT / r"01_Data\Processed\rasters_aligned"
OUT_DIR = PROJECT_ROOT / r"04_Results\preprocess\qc\rasters_aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# VIS SETTINGS
# ----------------------------
MAX_DIM = 1400
READ_RESAMPLING = Resampling.nearest

HILLSHADE_KEYS = ["hillshade", "hs_"]
CMAP_HILLSHADE = "gray"
CMAP_OTHER = "turbo"

STRETCH_PCTS = (2, 98)     # GIS-like contrast stretch
USE_EQUALIZE_FOR_OTHER = True
NBINS_EQ = 256
SHOW_COLORBAR = True


def short_crs(crs) -> str:
    if crs is None:
        return "CRS:None"
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    s = crs.to_string()
    m = re.search(r"EPSG[:=](\d+)", s)
    if m:
        return f"EPSG:{m.group(1)}"
    if "WGS 84" in s or "WGS84" in s:
        return "WGS84"
    return (s[:22] + "...") if len(s) > 25 else s


def read_downsample(src, band=1, max_dim=1400):
    h, w = src.height, src.width
    scale = max(h / max_dim, w / max_dim, 1.0)
    out_h = int(h / scale)
    out_w = int(w / scale)
    return src.read(band, out_shape=(out_h, out_w), resampling=READ_RESAMPLING).astype("float32")


def mask_nodata(arr, nodata):
    if nodata is None:
        return np.ma.masked_invalid(arr)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.ma.masked_invalid(arr)
    return np.ma.masked_equal(arr, nodata)


def percentile_stretch(arr_ma, pcts=(2, 98)):
    vals = arr_ma.compressed()
    if vals.size == 0:
        return arr_ma, None, None
    vmin, vmax = np.percentile(vals, pcts)
    return arr_ma, float(vmin), float(vmax)


def hist_equalize(arr_ma, nbins=256):
    """Return equalized array in [0,1], preserving mask."""
    if np.ma.isMaskedArray(arr_ma):
        mask = arr_ma.mask
        vals = arr_ma.compressed()
        base = arr_ma.filled(np.nan)
    else:
        mask = None
        base = np.asarray(arr_ma, dtype="float32")
        vals = base.ravel()

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return arr_ma

    hist, bin_edges = np.histogram(vals, bins=nbins, density=False)
    cdf = hist.cumsum().astype("float32")
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)

    base2 = np.asarray(base, dtype="float32")
    base2 = np.clip(base2, bin_edges[0], bin_edges[-1])
    idx = np.searchsorted(bin_edges[1:], base2)
    idx = np.clip(idx, 0, nbins - 1)

    out = cdf[idx].astype("float32")
    if mask is not None:
        out = np.ma.array(out, mask=mask)
    return out


def is_hillshade(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in HILLSHADE_KEYS)


def main():
    if not ALIGNED_DIR.exists():
        raise FileNotFoundError(f"Missing: {ALIGNED_DIR}")

    rasters = sorted(ALIGNED_DIR.glob("*.tif"))
    if not rasters:
        raise FileNotFoundError(f"No aligned rasters found: {ALIGNED_DIR}")

    # Read DEM CRS only for reference (not used in plot)
    dem_crs_short = None
    if DEM_REF.exists():
        with rasterio.open(DEM_REF) as d:
            dem_crs_short = short_crs(d.crs)

    for rp in rasters:
        with rasterio.open(rp) as src:
            arr = read_downsample(src, band=1, max_dim=MAX_DIM)
            arr = mask_nodata(arr, src.nodata)

            crs_s = short_crs(src.crs)
            if crs_s == "CRS:None" and dem_crs_short is not None:
                crs_s = f"{crs_s} (DEM={dem_crs_short})"

            title = f"{rp.name} | {crs_s} | {src.width}x{src.height}"

            plt.figure(figsize=(10, 7))

            if is_hillshade(rp.stem):
                arr_s, vmin, vmax = percentile_stretch(arr, STRETCH_PCTS)
                im = plt.imshow(arr_s, cmap=CMAP_HILLSHADE, vmin=vmin, vmax=vmax)
                cbar_label = "Hillshade"
                suffix = "_hs"
            else:
                if USE_EQUALIZE_FOR_OTHER:
                    arr_eq = hist_equalize(arr, NBINS_EQ)
                    im = plt.imshow(arr_eq, cmap=CMAP_OTHER, vmin=0, vmax=1)
                    cbar_label = "Equalized [0–1]"
                    suffix = "_eq"
                else:
                    arr_s, vmin, vmax = percentile_stretch(arr, STRETCH_PCTS)
                    im = plt.imshow(arr_s, cmap=CMAP_OTHER, vmin=vmin, vmax=vmax)
                    cbar_label = "Value (stretched)"
                    suffix = "_stretched"

            plt.title(title)
            plt.axis("off")

            if SHOW_COLORBAR:
                plt.colorbar(im, fraction=0.03, pad=0.02, label=cbar_label)

            out_png = OUT_DIR / f"{rp.stem}{suffix}.png"
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close()

            print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
