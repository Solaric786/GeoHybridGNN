import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def pick_excel(vec_dir: Path) -> Path:
    xs = sorted(vec_dir.rglob("*.xlsx")) + sorted(vec_dir.rglob("*.xls"))
    if not xs:
        raise FileNotFoundError(f"No Excel file found under: {vec_dir}")
    for p in xs:
        if "copper" in p.name.lower():
            return p
    return xs[0]


def find_lon_lat_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    for a, b in [("longitude", "latitude"), ("lon", "lat"), ("x", "y")]:
        if a in cols and b in cols:
            return cols[a], cols[b]
    raise ValueError(f"Lon/Lat columns not found. Columns are: {list(df.columns)}")


def main():
    os.environ.pop("PROJ_LIB", None)
    os.environ.pop("PROJ_DATA", None)
    os.environ.pop("GDAL_DATA", None)

    root = project_root()

    dem_path = root / "01_Data" / "Raw" / "rasters" / "dem_ref.tif"
    vec_dir = root / "01_Data" / "Raw" / "vectors"
    out_dir = root / "01_Data" / "Processed" / "vectors_clipped"
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = pick_excel(vec_dir)
    df = pd.read_excel(excel_path)

    lon_col, lat_col = find_lon_lat_columns(df)

    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.loc[lon.notna() & lat.notna()].copy()
    lon = lon.loc[df.index]
    lat = lat.loc[df.index]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(lon, lat)],
        crs="EPSG:4326",
    )

    with rasterio.open(dem_path) as ds:
        dem_crs = ds.crs
        b = ds.bounds

    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)

    clip_poly = box(b.left, b.bottom, b.right, b.top)
    gdf2 = gdf.clip(clip_poly)

    out_path = out_dir / "copper_points_clip.shp"
    gdf2.to_file(out_path)

    print("Excel:", excel_path)
    print("Used columns:", lon_col, lat_col)
    print("Saved:", out_path, f"(n={len(gdf2)})")


if __name__ == "__main__":
    main()
