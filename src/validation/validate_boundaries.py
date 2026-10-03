import geopandas as gpd
from pathlib import Path


# ============================================================
# PATH
# ============================================================

INPUT_PATH = Path(
    "data/raw/boundaries/coastal_ap_districts.geojson.gpkg"
)


# ============================================================
# LOAD LAYER
# ============================================================

gdf = gpd.read_file(INPUT_PATH)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 70)
print("COASTAL ANDHRA PRADESH BOUNDARY VALIDATION")
print("=" * 70)

print("\nFile:")
print(INPUT_PATH)

print("\nShape:")
print(gdf.shape)

print("\nColumns:")
print(gdf.columns.tolist())

print("\nCRS:")
print(gdf.crs)


# ============================================================
# DISTRICT COLUMN
# ============================================================

print("\n" + "=" * 70)
print("DISTRICT VALUES")
print("=" * 70)

# Show likely district-related columns
district_columns = [
    column
    for column in gdf.columns
    if "district" in column.lower()
]

print("District-related columns:")
print(district_columns)

if not district_columns:
    raise ValueError(
        "No district-related column found."
    )

district_column = district_columns[0]

print("\nUsing district column:")
print(district_column)

print("\nDistricts:")
print(
    gdf[district_column]
    .sort_values()
    .to_string(index=False)
)


# ============================================================
# EXPECTED FIVE DISTRICTS
# ============================================================

expected_districts = {
    "Srikakulam",
    "Vizianagaram",
    "Visakhapatnam",
    "Anakapalli",
    "Kakinada",
}

actual_districts = set(
    gdf[district_column]
    .dropna()
    .astype(str)
    .str.strip()
)


print("\n" + "=" * 70)
print("EXPECTED 5-DISTRICT CHECK")
print("=" * 70)

print("Expected:")
for district in sorted(expected_districts):
    print(" -", district)

print("\nFound:")
for district in sorted(actual_districts):
    print(" -", district)


missing = expected_districts - actual_districts
extra = actual_districts - expected_districts


print("\nMissing expected districts:")
print(sorted(missing))

print("\nUnexpected districts:")
print(sorted(extra))


# ============================================================
# GEOMETRY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("GEOMETRY VALIDATION")
print("=" * 70)

print(
    "Empty geometries:",
    gdf.geometry.is_empty.sum()
)

print(
    "Missing geometries:",
    gdf.geometry.isna().sum()
)

print(
    "Invalid geometries:",
    (~gdf.geometry.is_valid).sum()
)


# ============================================================
# GEOMETRY TYPES
# ============================================================

print("\nGeometry types:")

print(
    gdf.geometry.geom_type.value_counts()
    .to_string()
)


# ============================================================
# BOUNDS
# ============================================================

print("\n" + "=" * 70)
print("BOUNDING BOX")
print("=" * 70)

minx, miny, maxx, maxy = gdf.total_bounds

print("Minimum longitude:", minx)
print("Minimum latitude: ", miny)
print("Maximum longitude:", maxx)
print("Maximum latitude: ", maxy)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)