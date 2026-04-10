# 04_rasters_align_to_dem.py
# Align ALL rasters in 01_Data/Raw/rasters/ to dem_ref.tif grid
# Outputs: 01_Data/Processed/rasters_aligned/*_aligned.tif
# Report:  04_Results/preprocess/rasters_align_report.csv

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")
DEM_REF = PROJECT_ROOT / r"01_Data\Raw\rasters\dem_ref.tif"
RAW_RASTERS_DIR = PROJECT_ROOT / r"01_Data\Raw\rasters"

OUT_DIR = PROJECT_ROOT / r"01_Data\Processed\rasters_aligned"
REPORT_CSV = PROJECT_ROOT / r"04_Results\preprocess\rasters_align_report.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

RASTER_EXTS = {".tif", ".tiff", ".img"}

# heuristics for categorical rasters (use nearest)
CATEGORICAL_HINTS = ["lith", "geo", "class", "landcover", "lc", "unit", "code", "label", "id", "cat"]

def choose_resampling(path: Path) -> Resampling:
    name = path.stem.lower()
    return Resampling.nearest if any(h in name for h in CATEGORICAL_HINTS) else Resampling.bilinear

def is_integer_dtype(dtype_str: str) -> bool:
    return np.issubdtype(np.dtype(dtype_str), np.integer)

def transforms_close(a, b, tol=1e-9) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))

def pick_nodata(dtype_str: str):
    dt = np.dtype(dtype_str)
    if np.issubdtype(dt, np.floating):
        return -9999.0
    if np.issubdtype(dt, np.signedinteger):
        return -9999
    # unsigned int
    return 0

def main():
    if not DEM_REF.exists():
        raise FileNotFoundError(f"DEM reference not found: {DEM_REF}")

    with rasterio.open(DEM_REF) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    rasters = sorted([p for p in RAW_RASTERS_DIR.rglob("*") if p.suffix.lower() in RASTER_EXTS])
    if not rasters:
        raise FileNotFoundError(f"No rasters found under: {RAW_RASTERS_DIR}")

    rows = []

    for src_path in rasters:
        if src_path.resolve() == DEM_REF.resolve():
            continue

        resampling = choose_resampling(src_path)
        out_path = OUT_DIR / f"{src_path.stem}_aligned.tif"

        row = {
            "src": str(src_path),
            "dst": str(out_path),
            "resampling": resampling.name,
            "status": "OK"
        }

        try:
            with rasterio.open(src_path) as src:
                if src.crs is None:
                    raise ValueError("Source raster has no CRS. Fix CRS first.")

                src_dtype = src.dtypes[0]
                src_count = src.count

                # For bilinear on integer rasters -> output float32 to avoid truncation
                out_dtype = "float32" if (resampling != Resampling.nearest and is_integer_dtype(src_dtype)) else src_dtype

                dst_nodata = src.nodata if src.nodata is not None else pick_nodata(out_dtype)

                row.update({
                    "src_crs": str(src.crs),
                    "src_dtype": src_dtype,
                    "src_count": int(src_count),
                    "src_width": int(src.width),
                    "src_height": int(src.height),
                    "src_nodata": src.nodata,
                    "dst_dtype": out_dtype,
                    "dst_nodata": dst_nodata,
                })

                dst_profile = src.profile.copy()
                dst_profile.update({
                    "driver": "GTiff",
                    "crs": ref_crs,
                    "transform": ref_transform,
                    "width": ref_width,
                    "height": ref_height,
                    "count": src_count,
                    "dtype": out_dtype,
                    "nodata": dst_nodata,
                    "compress": "DEFLATE",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                })

                with rasterio.open(out_path, "w", **dst_profile) as dst:
                    for b in range(1, src_count + 1):
                        src_data = src.read(b)

                        # destination array (float32 if needed)
                        dst_data = np.zeros((ref_height, ref_width), dtype=np.dtype(out_dtype))

                        reproject(
                            source=src_data,
                            destination=dst_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            dst_nodata=dst_nodata,
                            resampling=resampling,
                        )
                        dst.write(dst_data, b)

            # Verify alignment
            with rasterio.open(out_path) as chk:
                row.update({
                    "aligned_crs_ok": (chk.crs == ref_crs),
                    "aligned_shape_ok": (chk.width == ref_width and chk.height == ref_height),
                    "aligned_transform_ok": transforms_close(tuple(chk.transform), tuple(ref_transform)),
                })

        except Exception as e:
            row["status"] = f"FAIL: {type(e).__name__}: {e}"

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_CSV, index=False)

    ok = (df["status"] == "OK").sum()
    fail = (df["status"] != "OK").sum()
    print(f"Aligned rasters saved in: {OUT_DIR}")
    print(f"Report saved: {REPORT_CSV}")
    print(f"OK: {ok} | FAIL: {fail}")

if __name__ == "__main__":
    main()
