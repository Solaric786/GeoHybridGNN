from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_hillshade_downsample(path: Path, max_dim: int = 4000):
    with rasterio.open(path) as ds:
        extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)

        scale = max(ds.width / max_dim, ds.height / max_dim, 1.0)
        out_w = int(ds.width / scale)
        out_h = int(ds.height / scale)

        hs = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)

    hs = np.nan_to_num(hs)
    hs = hs.clip(0, 255).astype(np.uint8) if hs.max() > 1.0 else (hs * 255.0).clip(0, 255).astype(np.uint8)
    return hs, extent


def find_pakistan_boundary(search_dirs: list[Path]) -> Path | None:
    # looks for filenames like: pakistan*, *pak*adm0*, *pak*bound*, *country*boundary*, etc.
    keys = ("pakistan", "pak")
    hints = ("bound", "boundary", "adm0", "outline", "country")
    exts = (".shp", ".gpkg", ".geojson")

    for d in search_dirs:
        if not d.exists():
            continue
        for ext in exts:
            for p in d.rglob(f"*{ext}"):
                name = p.name.lower()
                if any(k in name for k in keys) and any(h in name for h in hints):
                    return p
    return None


def main():
    root = project_root()

    hillshade = root / "01_Data" / "Raw" / "rasters" / "hillshade_315_45_30m.tif"
    copper = root / "01_Data" / "Processed" / "vectors_clipped" / "copper_points_clip.shp"
    faults = root / "01_Data" / "Processed" / "vectors_clipped" / "gem_active_faults_crs4326_clip.gpkg"
    geology = root / "01_Data" / "Processed" / "vectors_clipped" / "Export_Output_3_clip.gpkg"
    tectonics = root / "01_Data" / "Processed" / "vectors_clipped" / "WEP_PRVG_clip.gpkg"

    out_dir = root / "04_Results" / "preprocess" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "03_qc_overlay_usgs_hillshade.png"

    hs, extent = read_hillshade_downsample(hillshade, max_dim=4000)
    bbox_geom = box(extent[0], extent[2], extent[1], extent[3])

    gdf_cu = gpd.read_file(copper)
    gdf_faults = gpd.read_file(faults) if faults.exists() else None
    gdf_geo = gpd.read_file(geology) if geology.exists() else None
    gdf_tect = gpd.read_file(tectonics) if tectonics.exists() else None

    # Pakistan boundary (optional)
    pak_path = find_pakistan_boundary([
        root / "01_Data" / "Processed" / "vectors_clipped",
        root / "01_Data" / "Raw" / "vectors",
    ])
    gdf_pak = gpd.read_file(pak_path) if pak_path else None

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(hs, extent=extent, origin="upper", cmap="gray")

    ax.set_title("QC Overlay: Hillshade + Geology + Tectonics + Faults + Copper Points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # geology boundaries
    if gdf_geo is not None and len(gdf_geo) > 0:
        gdf_geo.boundary.plot(ax=ax, color="orange", linewidth=0.45, label="Geology (boundaries)", zorder=3)

    # tectonic boundaries
    if gdf_tect is not None and len(gdf_tect) > 0:
        gdf_tect.boundary.plot(ax=ax, color="purple", linewidth=0.9, label="Tectonic boundaries", zorder=4)

    # faults
    if gdf_faults is not None and len(gdf_faults) > 0:
        gdf_faults.plot(ax=ax, color="deepskyblue", linewidth=0.9, label="Faults", zorder=5)

    # Pakistan boundary (BOLD)
    if gdf_pak is not None and len(gdf_pak) > 0:
        if gdf_pak.crs is None:
            # if missing CRS, assume EPSG:4326 since your project is in lon/lat
            gdf_pak = gdf_pak.set_crs("EPSG:4326", allow_override=True)
        if gdf_pak.crs != "EPSG:4326":
            gdf_pak = gdf_pak.to_crs("EPSG:4326")

        # clip to bbox so it doesn't draw huge geometries
        gdf_pak = gdf_pak[gdf_pak.intersects(bbox_geom)]
        if len(gdf_pak) > 0:
            gdf_pak.boundary.plot(ax=ax, color="black", linewidth=2.4, label="Pakistan boundary", zorder=6)

    # copper points
    if len(gdf_cu) > 0:
        gdf_cu.plot(
            ax=ax, color="red", markersize=45, edgecolor="black", linewidth=0.35,
            label="Copper points", zorder=7
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print("Saved QC overlay:", out_png)
    if pak_path:
        print("Pakistan boundary used:", pak_path)
    else:
        print("Pakistan boundary: NOT FOUND (skipped)")


if __name__ == "__main__":
    main()
