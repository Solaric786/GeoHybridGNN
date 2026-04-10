# 08_distance_to_fault_raster_and_plot.py
# Creates distance-to-fault raster (GeoTIFF) + saves a quicklook PNG with proper map extent axes.
# Output:
#  - 01_Data/Processed/rasters_features/dist_to_fault_m_aligned.tif
#  - 04_Results/maps/dist_to_fault_quicklook.png

from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.enums import Resampling

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except Exception:
    HAS_GDAL = False

# ----------------------------
# USER CONFIG
# ----------------------------
PROJECT_ROOT = Path(r"D:\Atif's_Science\Chagai_HybridCu_2026")

DEM_REF = PROJECT_ROOT / r"01_Data\Raw\rasters\dem_ref.tif"

# If you already have a clipped/reprojected faults layer, point to it here.
# If left None, the script auto-finds something containing "fault" under vectors_clipped.
FAULTS_VECTOR = None  # e.g., PROJECT_ROOT / r"01_Data\Processed\vectors_clipped\faults_clip.gpkg"
FAULTS_LAYER = None   # e.g., "data" for GeoPackage; None -> first layer

OUT_DIST_TIF = PROJECT_ROOT / r"01_Data\Processed\rasters_features\dist_to_fault_m_aligned.tif"
OUT_PNG      = PROJECT_ROOT / r"04_Results\maps\dist_to_fault_quicklook.png"

# Temp mask (burn faults as 1 on DEM grid)
TMP_MASK_TIF = PROJECT_ROOT / r"01_Data\Processed\rasters_features\_tmp_fault_mask.tif"

# Control
RECOMPUTE_DISTANCE = False  # set True only if you want to rebuild the distance raster
ALL_TOUCHED = True          # rasterization option

# Quicklook downsample target (bigger -> slower, sharper)
MAX_W = 1600

# GDAL tuning (cache is in MB)
GDAL_CACHEMAX_MB = 1024

# ----------------------------
# Helpers
# ----------------------------
def auto_find_faults_vector(root: Path) -> Path:
    cand_dir = root / r"01_Data\Processed\vectors_clipped"
    if not cand_dir.exists():
        raise FileNotFoundError(f"Missing folder: {cand_dir}")

    patterns = [
        "*fault*clip*.gpkg", "*fault*clip*.shp",
        "*Fault*clip*.gpkg", "*Fault*clip*.shp",
        "*fault*.gpkg", "*fault*.shp",
        "*Fault*.gpkg", "*Fault*.shp",
    ]
    hits = []
    for pat in patterns:
        hits.extend(sorted(cand_dir.glob(pat)))
    hits = list(dict.fromkeys(hits))  # unique, keep order
    if not hits:
        raise FileNotFoundError(f"No faults file found in {cand_dir} (tried patterns {patterns})")
    return hits[0]

def is_geographic(crs) -> bool:
    try:
        return bool(crs and crs.is_geographic)
    except Exception:
        return False

def approx_m_per_degree(lat_deg: float) -> tuple[float, float]:
    # simple, good-enough approximation
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat

# ----------------------------
# Distance computation (fast + scalable via GDAL)
# ----------------------------
def build_distance_to_fault_gdal(dem_path: Path, faults_path: Path, out_tif: Path, tmp_mask: Path,
                                 layer_name: str | None = None) -> None:
    if not HAS_GDAL:
        raise RuntimeError("GDAL (osgeo) is not available in this environment.")

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_mask.parent.mkdir(parents=True, exist_ok=True)

    # Open DEM as template grid
    dem_ds = gdal.Open(str(dem_path))
    if dem_ds is None:
        raise FileNotFoundError(f"Cannot open DEM: {dem_path}")

    xsize, ysize = dem_ds.RasterXSize, dem_ds.RasterYSize
    gt = dem_ds.GetGeoTransform()
    proj = dem_ds.GetProjection()

    drv = gdal.GetDriverByName("GTiff")

    # 1) Rasterize faults to a Byte mask on DEM grid
    if tmp_mask.exists():
        tmp_mask.unlink()

    mask_ds = drv.Create(
        str(tmp_mask), xsize, ysize, 1, gdal.GDT_Byte,
        options=[
            "COMPRESS=LZW", "TILED=YES",
            "BLOCKXSIZE=256", "BLOCKYSIZE=256"
        ],
    )
    mask_ds.SetGeoTransform(gt)
    mask_ds.SetProjection(proj)
    mb = mask_ds.GetRasterBand(1)
    mb.SetNoDataValue(0)
    mb.Fill(0)

    vds = gdal.OpenEx(str(faults_path), gdal.OF_VECTOR)
    if vds is None:
        raise FileNotFoundError(f"Cannot open faults vector: {faults_path}")

    layer = vds.GetLayerByName(layer_name) if layer_name else vds.GetLayer(0)
    if layer is None:
        raise ValueError(f"Could not open layer '{layer_name}' in: {faults_path}")

    rasterize_opts = []
    if ALL_TOUCHED:
        rasterize_opts.append("ALL_TOUCHED=TRUE")

    err = gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1], options=rasterize_opts)
    if err != 0:
        raise RuntimeError("RasterizeLayer failed.")

    mask_ds.FlushCache()
    mask_ds = None
    vds = None

    # 2) Proximity (distance-to-nearest fault pixel)
    if out_tif.exists():
        out_tif.unlink()

    src_ds = gdal.Open(str(tmp_mask))
    dst_ds = drv.Create(
        str(out_tif), xsize, ysize, 1, gdal.GDT_Float32,
        options=[
            "COMPRESS=LZW", "TILED=YES",
            "BLOCKXSIZE=256", "BLOCKYSIZE=256"
        ],
    )
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)

    # DISTUNITS=GEO uses CRS units (meters for projected CRS).
    # If your DEM is EPSG:4326, this will be in degrees (still usable, but not meters).
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_CACHEMAX", str(int(GDAL_CACHEMAX_MB)))

    gdal.ComputeProximity(
        src_ds.GetRasterBand(1),
        dst_ds.GetRasterBand(1),
        options=["VALUES=1", "DISTUNITS=GEO"]
    )

    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds.FlushCache()
    src_ds = None
    dst_ds = None

# ----------------------------
# Quicklook plot (no recompute)
# ----------------------------
def save_quicklook_png(dist_tif: Path, out_png: Path) -> None:
    if not HAS_PLT:
        print("[WARN] matplotlib not available; skip PNG.")
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dist_tif) as src:
        crs = src.crs
        b = src.bounds
        width, height = src.width, src.height

        scale = max(width // MAX_W, 1)
        out_w = max(width // scale, 1)
        out_h = max(height // scale, 1)

        img = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            img = np.where(img == nodata, np.nan, img)

        extent = (b.left, b.right, b.bottom, b.top)

    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax = plt.gca()
    im = ax.imshow(img, extent=extent, origin="upper")
    ax.set_title("Distance to faults (quick view, downsampled)")

    if is_geographic(crs):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("distance (CRS units)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Quicklook saved: {out_png}")

# ----------------------------
# Main
# ----------------------------
def main():
    if FAULTS_VECTOR is None:
        faults_path = auto_find_faults_vector(PROJECT_ROOT)
    else:
        faults_path = Path(FAULTS_VECTOR)

    if (not OUT_DIST_TIF.exists()) or RECOMPUTE_DISTANCE:
        if not HAS_GDAL:
            raise RuntimeError("This script needs 'osgeo.gdal' for fast proximity on large rasters.")
        print(f"[INFO] Building distance raster from faults: {faults_path}")
        build_distance_to_fault_gdal(
            dem_path=DEM_REF,
            faults_path=faults_path,
            out_tif=OUT_DIST_TIF,
            tmp_mask=TMP_MASK_TIF,
            layer_name=FAULTS_LAYER
        )
        print(f"[OK] Distance-to-fault raster saved: {OUT_DIST_TIF}")
    else:
        print(f"[OK] Distance raster already exists (skip compute): {OUT_DIST_TIF}")

    save_quicklook_png(OUT_DIST_TIF, OUT_PNG)

if __name__ == "__main__":
    main()
