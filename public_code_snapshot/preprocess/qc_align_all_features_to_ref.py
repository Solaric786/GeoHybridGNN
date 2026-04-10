# 02_Code/src/preprocess/90_qc_align_all_features_to_ref.py
# Ensures ALL feature rasters match the reference grid exactly (shape/transform/crs/nodata).
# Writes aligned copies to 01_Data/Processed/rasters_features_aligned/
# Also writes a CSV QC report.

from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

PROJECT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

REF = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_aligned\hillshade_315_45_30m_aligned.tif")

# Put *all* your feature rasters here (edit paths as needed)
FEATURES = [
    # Existing features
    Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_features\dist_to_fault_m_aligned.tif"),
    Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_aligned\lithology_aligned.tif"),
    # Sentinel / indices (examples)
    # Path(r"...\01_Data\Processed\rasters_aligned\s2_iron_oxide_aligned.tif"),
    # Path(r"...\01_Data\Processed\rasters_aligned\s2_clay_aligned.tif"),

    # NEW geochem (already aligned, but we still QC)
    Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_geochem_digitized\Cu_geochem_aligned_to_hillshade.tif"),
    Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_geochem_digitized\Mo_geochem_aligned_to_hillshade.tif"),
    Path(r"D:\Atif's_Science\Chagai_HybridCu_2026\01_Data\Processed\rasters_geochem_digitized\Ag_geochem_aligned_to_hillshade.tif"),
]

OUT_DIR = PROJECT / "01_Data/Processed/rasters_features_aligned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = OUT_DIR / "qc_alignment_report.csv"


def same_grid(a, b) -> bool:
    return (
        a.crs == b.crs
        and a.transform == b.transform
        and a.width == b.width
        and a.height == b.height
    )


def align_to_ref(src_path: Path, ref_path: Path, out_path: Path) -> None:
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        src_nodata = src.nodata
        # If nodata missing, choose a safe default
        if src_nodata is None:
            src_nodata = -9999.0

        dst = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        src_arr = src.read(1).astype(np.float32)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        profile = ref.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")

        with rasterio.open(out_path, "w", **profile) as ds:
            ds.write(np.where(np.isnan(dst), profile["nodata"], dst).astype(np.float32), 1)


def main():
    with rasterio.open(REF) as ref:
        pass

    rows = []
    for p in FEATURES:
        if not p.exists():
            rows.append([str(p), "MISSING", "", "", "", ""])
            continue

        with rasterio.open(REF) as ref, rasterio.open(p) as src:
            ok = same_grid(src, ref)
            rows.append([
                str(p),
                "OK" if ok else "ALIGN_NEEDED",
                str(src.crs),
                str(ref.crs),
                f"{src.width}x{src.height}",
                f"{ref.width}x{ref.height}",
            ])

        if not ok:
            out_path = OUT_DIR / f"{p.stem}_aligned_to_ref.tif"
            print(f"[ALIGN] {p.name} -> {out_path.name}")
            align_to_ref(p, REF, out_path)
        else:
            # still copy to unified folder (optional but keeps one place)
            out_path = OUT_DIR / f"{p.stem}_aligned_to_ref.tif"
            if not out_path.exists():
                out_path.write_bytes(p.read_bytes())

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "status", "src_crs", "ref_crs", "src_shape", "ref_shape"])
        w.writerows(rows)

    print(f"[DONE] QC report: {REPORT_CSV}")
    print(f"[DONE] Unified aligned folder: {OUT_DIR}")


if __name__ == "__main__":
    main()