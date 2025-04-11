import math
import pyproj
import geopandas as gpd
from shapely.ops import unary_union

def estimate_utm_crs(gdf_wgs84):
    """Estimates the UTM CRS for a GeoDataFrame in WGS84.

    Args:
        gdf_wgs84 (gpd.GeoDataFrame): Input GeoDataFrame in EPSG:4326.

    Returns:
        str: The estimated UTM CRS string (e.g., 'EPSG:32614') or a Proj string.
             Returns a default UTM CRS if estimation fails.
    """
    print("Estimating UTM CRS...")
    try:
        if gdf_wgs84.empty or gdf_wgs84.geometry.iloc[0] is None:
            raise ValueError("Input GeoDataFrame is empty or has no geometry.")

        # Ensure input is WGS84
        if gdf_wgs84.crs != "EPSG:4326":
            print(f"  Warning: Input GDF CRS is {gdf_wgs84.crs}, expected EPSG:4326. Attempting conversion.")
            gdf_wgs84 = gdf_wgs84.to_crs("EPSG:4326")

        # Use unary_union to handle multi-part geometries and get a representative point
        unified_geom = unary_union(gdf_wgs84.geometry)
        if unified_geom.is_empty:
            raise ValueError("Unified geometry is empty after unary_union.")

        # Get centroid of the unified geometry
        center_lon, center_lat = unified_geom.centroid.x, unified_geom.centroid.y

        # Calculate UTM zone
        utm_zone = math.floor((center_lon + 180) / 6) + 1

        # Construct Proj string
        crs_proj_str = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        if center_lat < 0:
            crs_proj_str += " +south"

        # Try to get EPSG code
        try:
            crs = pyproj.CRS(crs_proj_str)
            epsg_code = crs.to_epsg()
            if epsg_code:
                print(f"-> Estimated UTM CRS (EPSG:{epsg_code})")
                return f"EPSG:{epsg_code}"
            else:
                print(f"-> Estimated UTM CRS (Proj): {crs_proj_str}")
                return crs_proj_str
        except Exception as crs_e:
            print(f"  Warning: Could not convert Proj string to EPSG ({crs_e}). Using Proj string.")
            return crs_proj_str

    except Exception as e:
        default_crs = 'EPSG:32614' # Default UTM zone 14N (covers parts of Texas)
        print(f"Warning: UTM estimation failed ({e}). Defaulting to {default_crs}.")
        return default_crs 