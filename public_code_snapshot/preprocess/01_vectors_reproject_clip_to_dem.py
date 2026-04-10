import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


def project_root() -> Path:
    # .../02_Code/src/preprocess/ -> go up to project root
    return Path(__file__).resolve().parents[3]


def find_vectors(vec_dir: Path):
    exts = ("*.shp", "*.SHP", "*.gpkg", "*.GPKG", "*.geojson", "*.GEOJSON")
    files = []
    for e in exts:
        files += list(vec_dir.rglob(e))
    return sorted(set(files))


def main():
    # Ensure we don't inherit PROJ from other installs
    os.environ.pop("PROJ_LIB", None)
    os.environ.pop("PROJ_DATA", None)
    os.environ.pop("GDAL_DATA", None)

    root = project_root()

    dem_path = root / "01_Data" / "Raw" / "rasters" / "dem_ref.tif"
    vec_in_dir = root / "01_Data" / "Raw" / "vectors"
    vec_out_dir = root / "01_Data" / "Processed" / "vectors_clipped"
    report_dir = root / "04_Results" / "preprocess"

    vec_out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not vec_in_dir.exists():
        raise FileNotFoundError(f"Vectors folder not found: {vec_in_dir}")

    with rasterio.open(dem_path) as ds:
        dem_crs = ds.crs
        b = ds.bounds  # left, bottom, right, top

    clip_poly = box(b.left, b.bottom, b.right, b.top)

    vector_files = find_vectors(vec_in_dir)
    print("DEM:", dem_path)
    print("VEC_DIR:", vec_in_dir)
    print("DEM CRS:", dem_crs)
    print("DEM BBOX:", b)
    print("Vectors in:", len(vector_files))

    rows = []

    for fp in vector_files:
        name = fp.name
        gdf = gpd.read_file(fp)

        if gdf.crs is None:
            rows.append({"file": name, "status": "SKIP (no CRS)", "in_path": str(fp)})
            print("[SKIP]", name, "(no CRS)")
            continue

        crs_in = str(gdf.crs)
        action = []

        if gdf.crs != dem_crs:
            gdf = gdf.to_crs(dem_crs)
            action.append("reproject")

        gdf2 = gdf.clip(clip_poly)
        action.append("clip_bbox")

        out_path = vec_out_dir / f"{fp.stem}_clip.gpkg"
        gdf2.to_file(out_path, driver="GPKG", layer="data")

        rows.append({
            "file": name,
            "status": "OK",
            "action": "+".join(action),
            "crs_in": crs_in,
            "crs_out": str(dem_crs),
            "n_in": len(gdf),
            "n_out": len(gdf2),
            "in_path": str(fp),
            "out_path": str(out_path),
        })
        print("[OK]", name, "->", out_path.name, f"({len(gdf)} -> {len(gdf2)})")

    report_path = report_dir / "01_vectors_reproject_clip_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
