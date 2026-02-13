"""
Worcester Age Category Map (POLYGONS ONLY) — geoplot + geopandas

Reads TWO paths from CLI arguments:
  1) --boundary  -> Worcester boundary map (shp/geojson/gpkg/etc)
  2) --dataset   -> Your dataset (geojson/shp/gpkg/parquet/csv) containing:
                     - 'age_category' column with values like 0.0/1.0/2.0/3.0/4.0
                     - 'geometry' column (if CSV/Parquet) as WKT or GeoJSON string

Rules:
  - Ignore rows where age_category == 0.0
  - Plot ONLY classes 1.0, 2.0, 3.0, 4.0
  - Legend labels: Class 1, Class 2, Class 3, Class 4
  - Colors:
      Class 1 = red    (#E41A1C)
      Class 2 = purple (#984EA3)
      Class 3 = blue   (#377EB8)
      Class 4 = green  (#4DAF4A)

Output:
  - saves: worcester_age_categories.pdf
  - shows plot

Usage:
  python map_maker.py --boundary <path_to_boundary> --dataset <path_to_dataset>

Install:
  pip install geopandas geoplot pyarrow shapely matplotlib
"""

import os
import json
import argparse

import pandas as pd
import geopandas as gpd
import geoplot
import geoplot.crs as gcrs

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from shapely import wkt
from shapely.geometry import shape


# ----------------------------
# Helpers
# ----------------------------
def _first_non_null(series: pd.Series):
    for v in series:
        if v is not None and (not (isinstance(v, float) and pd.isna(v))) and (not pd.isna(v)):
            return v
    return None


def parse_geometry_column(df: pd.DataFrame, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with a 'geometry' column to a GeoDataFrame.
    Accepts geometry as:
      - shapely geometries (already)
      - WKT strings (e.g., "POLYGON (...)")
      - GeoJSON strings (e.g., '{"type":"Polygon","coordinates":[...]}')
    """
    if geometry_col not in df.columns:
        raise ValueError(f"Expected a '{geometry_col}' column in DATASET_PATH, but it was not found.")

    sample = _first_non_null(df[geometry_col])

    # If it's already a shapely geometry, just wrap as GeoDataFrame
    if hasattr(sample, "geom_type"):
        return gpd.GeoDataFrame(df, geometry=geometry_col)

    # If it's a string, decide WKT vs GeoJSON
    if isinstance(sample, str):
        s = sample.strip()

        if s.startswith("{") or s.startswith("["):
            # GeoJSON string per row
            def _to_geom(v):
                if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
                    return None
                obj = json.loads(v) if isinstance(v, str) else v
                return shape(obj)

            geoms = df[geometry_col].apply(_to_geom)
        else:
            # WKT per row
            def _to_geom(v):
                if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
                    return None
                return wkt.loads(v)

            geoms = df[geometry_col].apply(_to_geom)

        return gpd.GeoDataFrame(df.drop(columns=[geometry_col]), geometry=geoms)

    raise TypeError(
        f"Unsupported geometry type in '{geometry_col}'. "
        f"Got sample type: {type(sample)}. Use WKT/GeoJSON strings or shapely geometries."
    )


def read_dataset_any(path: str) -> gpd.GeoDataFrame:
    """
    Reads a dataset that may be:
      - geospatial file: .shp/.geojson/.gpkg/...  -> gpd.read_file
      - parquet: .parquet -> try gpd.read_parquet, else pd.read_parquet and parse geometry column
      - csv: .csv -> pd.read_csv and parse geometry column
    """
    ext = os.path.splitext(path.lower())[1]

    if ext in [".shp", ".geojson", ".json", ".gpkg", ".geopackage", ".fgb", ".gml", ".kml"]:
        return gpd.read_file(path)

    if ext in [".parquet", ".pq"]:
        # geopandas can read parquet if it was written as a GeoParquet
        try:
            return gpd.read_parquet(path)
        except Exception:
            df = pd.read_parquet(path)
            return parse_geometry_column(df, geometry_col="geometry")

    if ext == ".csv":
        df = pd.read_csv(path)
        return parse_geometry_column(df, geometry_col="geometry")

    # Fallback: try geopandas read_file
    return gpd.read_file(path)


def coerce_age_category_to_int(series: pd.Series) -> pd.Series:
    """
    Converts values like 1.0/2.0/'3.0'/'Class 4' -> integers 1..4 when possible.
    Converts None/NaN values to -1 for visualization.
    Returns pandas Int64 (nullable) series.
    """
    def _to_int(v):
        if pd.isna(v):
            return -1
        s = str(v).strip().lower().replace("_", " ").replace("-", " ")
        # Common patterns: "1.0", "2", "class 3"
        if "class" in s:
            s = s.replace("class", "").strip()
        try:
            # handles "1.0" -> 1
            x = float(s)
            return int(x)
        except Exception:
            return -1

    return series.apply(_to_int).astype("Int64")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Worcester age category map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python map_maker.py --boundary /path/to/boundary.geojson --dataset /path/to/dataset.parquet
        """,
    )
    parser.add_argument(
        "--boundary",
        required=True,
        help="Path to Worcester boundary map (geojson/shp/gpkg/etc)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset (geojson/shp/gpkg/parquet/csv) with age_category and geometry columns",
    )

    args = parser.parse_args()

    boundary_path = args.boundary
    dataset_path = args.dataset

    if not os.path.exists(boundary_path):
        raise SystemExit(f"ERROR: Boundary file not found: {boundary_path}")
    if not os.path.exists(dataset_path):
        raise SystemExit(f"ERROR: Dataset file not found: {dataset_path}")

    worcester_map = gpd.read_file(boundary_path)
    gdf = read_dataset_any(dataset_path)

    if "age_category" not in gdf.columns:
        raise ValueError("DATASET_PATH must contain an 'age_category' column.")
    if gdf.geometry is None:
        raise ValueError("Dataset does not have a valid geometry column after reading/parsing.")

    # --- Enforce: classes -1 (unknown), 1, 2, 3, 4; ignore 0 and anything else ---
    gdf = gdf.copy()
    gdf["age_class"] = coerce_age_category_to_int(gdf["age_category"])

    # Keep -1 (unknown), and 1..4, drop 0
    gdf = gdf[gdf["age_class"].isin([-1, 1, 2, 3, 4])].copy()

    # Count number of records per class for debugging
    class_counts = gdf["age_class"].value_counts(dropna=False).sort_index()
    print("Record counts by age_class:")
    for c in [-1, 1, 2, 3, 4]:
        print(f"  Class {c}: {class_counts.get(c, 0)}")
    import sys
    sys.exit(0)


    if len(gdf) == 0:
        raise ValueError(
            "After filtering, no rows remain with age_category in {-1, 1.0, 2.0, 3.0, 4.0}. "
            "Check your 'age_category' values."
        )

    # CRS alignment
    if worcester_map.crs is None:
        print("WARNING: Worcester boundary CRS is missing. Plot will still run, but CRS alignment may be wrong.")
    if gdf.crs is None and worcester_map.crs is not None:
        print("WARNING: Dataset CRS is missing; assuming it matches Worcester boundary CRS.")
        gdf = gdf.set_crs(worcester_map.crs, allow_override=True)
    if worcester_map.crs is not None and gdf.crs is not None and worcester_map.crs != gdf.crs:
        gdf = gdf.to_crs(worcester_map.crs)

    # Optional: clip to Worcester boundary
    try:
        gdf = gpd.clip(gdf, worcester_map)
    except Exception:
        pass

    # Fixed mapping: class -> color + legend label
    class_fill_colors = {
        -1: "#CCCCCC",  # gray (unknown)
        1: "#E41A1C",  # red
        2: "#984EA3",  # purple
        3: "#377EB8",  # blue
        4: "#4DAF4A",  # green
    }
    class_labels = {
        -1: "Unknown",
        1: "Class 1",
        2: "Class 2",
        3: "Class 3",
        4: "Class 4",
    }

    proj = gcrs.AlbersEqualArea()

    # Base map
    ax = geoplot.polyplot(
        worcester_map,
        edgecolor="#29272D",
        linewidth=0.4,
        projection=proj,
        figsize=(10, 10),
    )

    legend_patches = []

    for c in [-1, 1, 2, 3, 4]:
        sub = gdf[gdf["age_class"] == c]

        # Skip if no polygons for that class
        if len(sub) == 0:
            continue

        # (Optional) fix invalid geometries that can break plotting/clipping
        try:
            sub = sub.copy()
            sub["geometry"] = sub.geometry.buffer(0)
        except Exception:
            pass

        geoplot.polyplot(
            sub,
            projection=proj,
            edgecolor=class_fill_colors[c],
            facecolor=class_fill_colors[c],
            linewidth=0.2,
            ax=ax,
        )

        # Only add to legend if not "Unknown" (c != -1)
        if c != -1:
            legend_patches.append(mpatches.Patch(color=class_fill_colors[c], label=class_labels[c]))

    plt.title("")
    plt.legend(
        handles=legend_patches,
        ncol=4,
        loc="lower center",
        frameon=True,
    )

    out_file = "worcester_building_age_map.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to: {out_file}")



if __name__ == "__main__":
    main()
