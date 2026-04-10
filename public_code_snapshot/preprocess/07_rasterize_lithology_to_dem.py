# 07_rasterize_lithology_to_dem.py  (FIXED for your GPKG: layer="data", field="Name")

from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np

PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

DEM_REF      = PROJECT_ROOT / r"01_Data\Raw\rasters\dem_ref.tif"
GEOLOGY_GPKG = PROJECT_ROOT / r"01_Data\Processed\vectors_clipped\Export_Output_3_clip.gpkg"

GEOLOGY_LAYER = "data"     # <-- your layer
LITH_FIELD    = "Name"     # <-- 63 unique classes (good)

OUT_RASTER = PROJECT_ROOT / r"01_Data\Processed\rasters_features\lithology_code_aligned.tif"
CODEBOOK   = PROJECT_ROOT / r"01_Data\Processed\tables\lithology_codebook.csv"

NODATA = 0  # reserved nodata code

def main():
    OUT_RASTER.parent.mkdir(parents=True, exist_ok=True)
    CODEBOOK.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(DEM_REF) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_profile = ref.profile.copy()

    gdf = gpd.read_file(GEOLOGY_GPKG, layer=GEOLOGY_LAYER)
    if gdf.crs is None:
        raise ValueError(f"Geology has no CRS: {GEOLOGY_GPKG}")
    gdf = gdf.to_crs(ref_crs)
    gdf = gdf[gdf.geometry.notnull()].copy()

    if LITH_FIELD not in gdf.columns:
        raise ValueError(f"LITH_FIELD='{LITH_FIELD}' not found. Available: {list(gdf.columns)}")

    vals = gdf[LITH_FIELD].fillna("UNKNOWN").astype(str)
    uniq = sorted(vals.unique())
    if len(uniq) <= 2:
        raise ValueError(f"Field '{LITH_FIELD}' has only {len(uniq)} classes; choose a better field (e.g., SymbolID).")

    code_map = {name: i + 1 for i, name in enumerate(uniq)}  # start at 1
    gdf["_code"] = vals.map(code_map).astype("int32")

    shapes = ((geom, int(code)) for geom, code in zip(gdf.geometry, gdf["_code"]) if geom is not None)

    out = features.rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=NODATA,
        dtype="int32",
        all_touched=False
    )

    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="int32",
        count=1,
        nodata=NODATA,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(OUT_RASTER, "w", **profile) as dst:
        dst.write(out, 1)

    pd.DataFrame({"code": list(code_map.values()), "category": list(code_map.keys())}).to_csv(CODEBOOK, index=False)

    print(f"[OK] Lithology/Unit raster: {OUT_RASTER}")
    print(f"[OK] Codebook: {CODEBOOK}")
    print(f"[INFO] Field used: {LITH_FIELD} | classes: {len(code_map)} | layer: {GEOLOGY_LAYER}")

if __name__ == "__main__":
    main()
