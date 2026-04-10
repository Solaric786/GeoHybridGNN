# 05_build_train_points_features.py
# Build training table by sampling aligned rasters at:
# - copper points (label=1)
# - background points (label=0)
#
# Inputs:
#   01_Data/Processed/vectors_clipped/copper_points_clip.shp
#   01_Data/Processed/rasters_aligned/*.tif
#   01_Data/Raw/rasters/dem_ref.tif
#
# Output:
#   01_Data/Processed/tables/train_points_features.csv

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

COPPER_SHP = PROJECT_ROOT / r"01_Data\Processed\vectors_clipped\copper_points_clip.shp"
DEM_REF = PROJECT_ROOT / r"01_Data\Raw\rasters\dem_ref.tif"
ALIGNED_DIR = PROJECT_ROOT / r"01_Data\Processed\rasters_aligned"

OUT_TABLE = PROJECT_ROOT / r"01_Data\Processed\tables\train_points_features.csv"
OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

# Optional polygon mask (Pakistan boundary). Leave as None to use DEM bbox.
MASK_VECTOR = None
# Example:
# MASK_VECTOR = PROJECT_ROOT / r"01_Data\Processed\vectors_clipped\Pakistan_Boundary_clip.gpkg"

N_BACKGROUND = 10000
MIN_DIST_METERS = 1000  # keep background at least this far from any copper point (set 0 to disable)
RANDOM_SEED = 7

# ----------------------------
# HELPERS
# ----------------------------
def list_aligned_rasters(folder: Path):
    rasters = sorted(folder.glob("*.tif"))
    if not rasters:
        raise FileNotFoundError(f"No aligned rasters found in: {folder}")
    return rasters

def sample_single_raster(src, coords):
    # returns array shape (n, bands)
    vals = np.array(list(src.sample(coords)))
    return vals

def sample_rasters(points_gdf: gpd.GeoDataFrame, rasters):
    coords = [(geom.x, geom.y) for geom in points_gdf.geometry]
    out = {}

    for rp in rasters:
        with rasterio.open(rp) as src:
            vals = sample_single_raster(src, coords)

            if vals.ndim == 2 and vals.shape[1] == 1:
                out[rp.stem] = vals[:, 0]
            else:
                for b in range(vals.shape[1]):
                    out[f"{rp.stem}_b{b+1}"] = vals[:, b]

    return pd.DataFrame(out)

def read_mask_polygon(mask_path: Path) -> gpd.GeoSeries:
    gdf = gpd.read_file(mask_path)
    if gdf.crs is None:
        raise ValueError(f"Mask vector has no CRS: {mask_path}")
    gdf = gdf.to_crs("EPSG:4326")
    # geopandas: unary_union deprecated -> use union_all()
    poly = gdf.geometry.union_all()
    return poly

def dem_bbox_polygon(dem_path: Path):
    with rasterio.open(dem_path) as dem:
        b = dem.bounds
    return box(b.left, b.bottom, b.right, b.top)

def is_valid_on_dem(dem_src, x, y):
    v = next(dem_src.sample([(x, y)]))[0]
    nd = dem_src.nodata
    if nd is None:
        return True
    if np.isnan(nd):
        return not np.isnan(v)
    return v != nd

def generate_background_points(poly, n, seed, dem_path: Path):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly.bounds

    pts = []
    max_tries = n * 500

    with rasterio.open(dem_path) as dem:
        tries = 0
        while len(pts) < n and tries < max_tries:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)

            if not poly.contains(p):
                tries += 1
                continue

            if not is_valid_on_dem(dem, x, y):
                tries += 1
                continue

            pts.append(p)
            tries += 1

    if len(pts) < n:
        raise RuntimeError(f"Could only sample {len(pts)} background points (requested {n}).")

    return pts

def apply_min_distance_filter(bg: gpd.GeoDataFrame, copper: gpd.GeoDataFrame, min_dist_m: float):
    if min_dist_m <= 0:
        bg["min_dist_m"] = np.nan
        return bg

    utm_crs = copper.estimate_utm_crs()
    copper_utm = copper.to_crs(utm_crs)
    bg_utm = bg.to_crs(utm_crs)

    copper_union = copper_utm.geometry.union_all()
    bg_utm["min_dist_m"] = bg_utm.geometry.apply(lambda g: g.distance(copper_union))

    keep = bg_utm["min_dist_m"] >= min_dist_m
    bg = bg.loc[keep.values].copy()
    bg["min_dist_m"] = bg_utm.loc[keep, "min_dist_m"].values
    return bg

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not COPPER_SHP.exists():
        raise FileNotFoundError(f"Missing copper shapefile: {COPPER_SHP}")
    if not DEM_REF.exists():
        raise FileNotFoundError(f"Missing DEM reference: {DEM_REF}")
    if not ALIGNED_DIR.exists():
        raise FileNotFoundError(f"Missing aligned raster folder: {ALIGNED_DIR}")

    # Load copper points
    copper = gpd.read_file(COPPER_SHP)
    if copper.crs is None:
        raise ValueError("Copper points have no CRS.")
    copper = copper.to_crs("EPSG:4326").copy()
    copper["label"] = 1
    copper["source"] = "copper"
    copper["lon"] = copper.geometry.x
    copper["lat"] = copper.geometry.y

    # Background sampling region
    if MASK_VECTOR is not None:
        poly = read_mask_polygon(Path(MASK_VECTOR))
    else:
        poly = dem_bbox_polygon(DEM_REF)

    # Generate background points (oversample slightly so distance filtering won't drop too many)
    oversample_n = int(N_BACKGROUND * 1.5) if MIN_DIST_METERS > 0 else N_BACKGROUND
    bg_pts = generate_background_points(poly, oversample_n, RANDOM_SEED, DEM_REF)

    bg = gpd.GeoDataFrame(geometry=bg_pts, crs="EPSG:4326")
    bg["label"] = 0
    bg["source"] = "background"
    bg["lon"] = bg.geometry.x
    bg["lat"] = bg.geometry.y

    # Distance filtering
    bg = apply_min_distance_filter(bg, copper, MIN_DIST_METERS)

    # Keep exactly N_BACKGROUND if possible
    if len(bg) >= N_BACKGROUND:
        bg = bg.sample(N_BACKGROUND, random_state=RANDOM_SEED).copy()
    else:
        print(f"Warning: background reduced to {len(bg)} after min-distance filtering.")

    # Load aligned rasters
    rasters = list_aligned_rasters(ALIGNED_DIR)

    # Combine points
    pts = pd.concat(
        [
            copper[["lon", "lat", "label", "source", "geometry"]],
            bg[["lon", "lat", "label", "source", "geometry"]],
        ],
        ignore_index=True,
    )
    pts_gdf = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:4326")

    # Sample rasters
    feats = sample_rasters(pts_gdf, rasters)

    # Final table
    out = pd.concat([pts_gdf.drop(columns=["geometry"]).reset_index(drop=True), feats], axis=1)
    out.to_csv(OUT_TABLE, index=False)

    print(f"Saved: {OUT_TABLE}")
    print(f"Rows: {len(out)} | Positives: {(out['label']==1).sum()} | Background: {(out['label']==0).sum()}")
    print(f"Rasters used: {len(rasters)}")

if __name__ == "__main__":
    main()
