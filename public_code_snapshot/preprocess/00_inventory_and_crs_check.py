from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
import pyproj


PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")
DATA_ROOT = PROJECT_ROOT / "01_Data"
OUT_DIR = PROJECT_ROOT / "04_Results" / "preprocess"


RASTER_EXT = {".tif", ".tiff"}
VECTOR_EXT = {".shp", ".geojson", ".gpkg"}


def raster_meta(p: Path) -> dict:
    with rasterio.open(p) as ds:
        return {
            "name": p.name,
            "path": str(p),
            "crs": str(ds.crs),
            "res": ds.res,
            "shape": f"{ds.height}x{ds.width}",
            "bands": ds.count,
            "dtype": str(ds.dtypes[0]),
            "nodata": ds.nodata,
        }


def vector_meta(p: Path) -> dict:
    gdf = gpd.read_file(p)
    geom_types = sorted(set(map(str, gdf.geom_type))) if len(gdf) else []
    return {
        "name": p.name,
        "path": str(p),
        "crs": str(gdf.crs),
        "n_features": int(len(gdf)),
        "geom_types": ";".join(geom_types),
        "columns": ";".join(list(gdf.columns)),
    }


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print("PYTHON EXE :", sys.executable)
    print("pyproj file :", pyproj.__file__)
    print("PROJ_LIB   :", os.environ.get("PROJ_LIB"))
    print("PROJ_DATA  :", os.environ.get("PROJ_DATA"))
    print("-" * 70)

    rasters = sorted([p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in RASTER_EXT])
    vectors = sorted([p for p in DATA_ROOT.rglob("*") if p.suffix.lower() in VECTOR_EXT])

    print(f"Found rasters: {len(rasters)}")
    print(f"Found vectors: {len(vectors)}")

    raster_rows = []
    for p in rasters:
        try:
            raster_rows.append(raster_meta(p))
        except Exception as e:
            raster_rows.append({"name": p.name, "path": str(p), "error": str(e)})

    vector_rows = []
    for p in vectors:
        try:
            vector_rows.append(vector_meta(p))
        except Exception as e:
            vector_rows.append({"name": p.name, "path": str(p), "error": str(e)})

    write_csv(raster_rows, OUT_DIR / "inspect_rasters.csv")
    write_csv(vector_rows, OUT_DIR / "inspect_vectors.csv")

    print("Saved:", OUT_DIR / "inspect_rasters.csv")
    print("Saved:", OUT_DIR / "inspect_vectors.csv")


if __name__ == "__main__":
    main()
