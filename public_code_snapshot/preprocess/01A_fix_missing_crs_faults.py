from pathlib import Path
import geopandas as gpd


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main():
    root = project_root()
    vec_dir = root / "01_Data" / "Raw" / "vectors"

    # find any shapefile containing these keywords
    candidates = []
    for p in vec_dir.rglob("*.shp"):
        n = p.name.lower()
        if "gem" in n and "fault" in n and "active" in n:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No GEM active faults shapefile found under: {vec_dir}")

    # pick the shortest name (usually the main one)
    src = sorted(candidates, key=lambda x: len(x.name))[0]
    print("Found:", src)

    gdf = gpd.read_file(src)

    if gdf.crs is not None:
        print("CRS already exists:", gdf.crs)
        return

    # Your DEM is EPSG:4326, so we assign the same
    gdf = gdf.set_crs(epsg=4326, allow_override=True)

    out = src.with_name(src.stem + "_crs4326.shp")
    gdf.to_file(out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
