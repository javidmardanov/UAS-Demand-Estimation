
# -*- coding: utf-8 -*-
"""
THIS IS NOT THE CODE. THIS IS SIMPLY A CODE FROM GOOGLE COLAB, TO TEST A SIMPLE MINIMAL VIABLE PRODUCT. THIS IS JUST A SIMPLE CARCASS, OF WHAT OUR PIPELINE SHOULD BE. 

UAS Last-Mile Delivery Demand Modeling and O-D Matching (Colab Implementation - v20.3 Corrected)

Implements the detailed plan v20.1.
Fixes YAML ScannerError by correcting indentation/structure for demand coefficients.
Includes previous fixes.
"""

# -----------------------------------------------------------------------------
# 0. Setup Environment
# -----------------------------------------------------------------------------
print("## 0. Setting up Environment...")
print("Installing required packages...")
!pip install geopandas osmnx cenpy contextily matplotlib seaborn scikit-learn requests scipy fuzzywuzzy python-levenshtein pyyaml Pillow pyogrio --quiet
print("Packages installed.")

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping, LineString
from shapely.ops import transform, unary_union
import osmnx as ox
import cenpy
from cenpy import products
from datetime import datetime, timedelta
import random
import math
from scipy.stats import poisson
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
import time
import json
import pyproj
from fuzzywuzzy import fuzz, process
import yaml
from PIL import Image
import glob
import requests
import zipfile
import io
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="Legend does not support handles for PatchCollection instances.")
warnings.filterwarnings("ignore", message="Legend does not support handles for LineCollection instances.")
warnings.filterwarnings("ignore", message="The Shapely GEOS version .* differs from the GEOS version")

# Matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# --- Optional: Add your Census API Key ---
print("Checking for Census API Key...")
census_api_key = os.environ.get("CENSUS_API_KEY", None)
census_api_key = "7ea7151ee08ce736334cae8a9bdd5b0f3f21b639"
if census_api_key and census_api_key != "YOUR_API_KEY_HERE": print("Using Census API Key from environment.")
else: census_api_key = None; print("Census API Key not set. Using default access.")
print("Environment setup complete.")

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
print("\n## 1. Configuration...")
# **Corrected YAML Structure v20.2**
config_yaml = """
area_selection:
  method: 'coordinates'
  coordinates: [30.265, -97.745, 30.270, -97.740]
  buffer_km: 1
data_acquisition:
  osm_tags: {'building': True, 'shop': True, 'amenity': True, 'landuse': True, 'highway': True}
  census_variables: ['B19013_001E', 'B01003_001E', 'B25010_001E', 'B23025_004E', 'B23025_002E', 'B25001_001E']
  census_product: "ACSDT5Y2022"
  state_code: '48'
  county_code: '453'
  tract_year: 2022
building_classification:
  method: 'rule_based'
  rule_based_parameters: {min_residential_area_m2: 30, likely_max_residential_area_m2: 500, min_nonresidential_area_m2: 2000, residential_building_tags: ['residential', 'house', 'apartments', 'detached', 'terrace', 'semidetached_house', 'bungalow', 'dormitory'], nonresidential_building_tags: ['commercial', 'retail', 'industrial', 'warehouse', 'office', 'supermarket', 'shop', 'mall', 'store', 'school', 'hospital', 'church', 'public', 'civic', 'government', 'hotel', 'motel', 'cathedral', 'chapel', 'clinic', 'kindergarten', 'university', 'college'], residential_landuse: ['residential'], nonresidential_landuse: ['commercial', 'industrial', 'retail', 'institutional', 'military', 'brownfield'], residential_road_types: ['residential', 'living_street', 'unclassified', 'tertiary'], primary_road_types: ['primary', 'secondary', 'trunk', 'motorway'], road_buffer_meters: 30, residential_name_keywords: ['apartment', 'residence', 'housing', 'condo', 'villa', 'home', 'house', 'living', 'manor', 'place', 'tower', 'lofts', 'village', 'gardens'], nonresidential_name_keywords: ['school', 'store', 'shop', 'business', 'office', 'bank', 'restaurant', 'cafe', 'hotel', 'church', 'hospital', 'clinic', 'inc', 'corp', 'ltd', 'university', 'college', 'station', 'county', 'city of', 'state of', 'medical', 'center', 'institute', 'foundation']}
  store_tags:
    shop: ['supermarket', 'department_store', 'convenience', 'mall', 'wholesale', 'grocery']
    building: ['warehouse', 'retail', 'commercial']
    amenity: ['marketplace', 'fast_food', 'pharmacy']
    name_keywords: ["walmart", "target", "amazon", "kroger", "H-E-B", "heb", "costco", "distribution center", "fulfillment", "supercenter", "warehouse", "grocery", "market", "cvs", "walgreens", "pharmacy"]
height_estimation: {default_height_m: 3.5, meters_per_level: 3.5, knn_neighbors: 5, use_area_feature: True, max_height_cap_m: 150}
population_allocation: {population_scale_factor: 1.0, avg_household_size_override: Null}
demand_model:
  base_deliveries_per_household_per_day: 0.18
  # Corrected: Each coefficient on its own line
  income_coef: 0.6
  pop_density_coef: 0.2
  household_size_coef: -0.15
  employment_coef: 0.1
  reference_values: {ref_median_income: 75000, ref_pop_density: 3000, ref_avg_household_size: 2.5, ref_employment_rate: 95}
  daily_variation: {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 0.8, 6: 0.5}
  hourly_distribution: {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.01, 7:0.02, 8:0.04, 9:0.06, 10:0.08, 11:0.09, 12:0.10, 13:0.11, 14:0.11, 15:0.10, 16:0.09, 17:0.08, 18:0.06, 19:0.04, 20:0.01, 21:0.00, 22:0.0, 23:0.0} # Adjusted sum = 1.0
  monthly_factors: {1: 0.8, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.05, 9: 1.0, 10: 1.1, 11: 1.5, 12: 1.75}
  simulation_start_date: '2023-11-06'
  simulation_duration_days: 7
origin_destination_matching:
  method: 'Market Share Weighted'
  market_shares: {'Walmart': 0.65, 'Target': 0.20, 'Retail Hub': 0.10, 'Kroger': 0.0, 'H-E-B': 0.05, 'Warehouse/Fulfillment': 0.0, 'Other': 0.0}
  simulation_hour_for_matching: 17
random_seed: 42
output_dir: './uas_analysis_output'
"""
config = yaml.safe_load(config_yaml)
# Handle Null conversion after loading
if config['population_allocation'].get('avg_household_size_override') == 'Null': config['population_allocation']['avg_household_size_override'] = None
if 'demand_model' in config and 'hourly_distribution' in config['demand_model']:
    try: # Process hourly distribution
        hourly_dist = config['demand_model']['hourly_distribution']
        hourly_dist = {int(k): float(v) if isinstance(v, (int, float)) else 0.0 for k, v in hourly_dist.items()}
        for h in range(24): hourly_dist.setdefault(h, 0.0)
        hourly_sum = sum(hourly_dist.values());
        if not math.isclose(hourly_sum, 1.0, rel_tol=1e-5):
             print(f"Warning: Hourly distribution sum {hourly_sum:.4f} != 1.0. Normalizing.");
             if hourly_sum > 1e-9: hourly_dist = {k: v / hourly_sum for k, v in hourly_dist.items()}
             else: print("Warning: Hourly sum zero. Using uniform."); hourly_dist = {h: 1/24 for h in range(24)}
        config['demand_model']['hourly_distribution'] = dict(sorted(hourly_dist.items()))
        print("Hourly distribution processed successfully.")
    except Exception as e: print(f"ERROR processing hourly distribution: {e}. Using fallback."); config['demand_model']['hourly_distribution'] = {h: 1/24 for h in range(24)}

seed = config['random_seed']; np.random.seed(seed); random.seed(seed); print(f"Using Random Seed: {seed}")
output_dir = config['output_dir']; subdirs = ['data', 'visualizations', 'stats', 'config', 'animation_frames']
for subdir in subdirs: os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
print(f"Output directory: {os.path.abspath(output_dir)}");
try: # Save config
    with open(os.path.join(output_dir, 'config', 'used_config.yaml'), 'w') as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False)
except Exception as e: print(f"Warning: Could not save config - {e}")
WGS84 = 'EPSG:4326'; print("Configuration loaded.")
data_subdir=os.path.join(output_dir, "data"); viz_subdir=os.path.join(output_dir, "visualizations"); stats_subdir=os.path.join(output_dir, "stats"); config_subdir=os.path.join(output_dir, "config"); frames_dir=os.path.join(output_dir, 'animation_frames')

def estimate_utm_crs(gdf_wgs84): # Same as v19.1
    print("Estimating UTM CRS...");
    try:
        if gdf_wgs84.empty or gdf_wgs84.geometry.iloc[0] is None: raise ValueError("Input empty.")
        unified_geom = unary_union(gdf_wgs84.geometry);
        if unified_geom.is_empty: raise ValueError("Unified geometry empty.")
        center_lon, center_lat = unified_geom.centroid.x, unified_geom.centroid.y; utm_zone = math.floor((center_lon + 180) / 6) + 1
        crs_proj_str = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs";
        if center_lat < 0: crs_proj_str += " +south"
        crs = pyproj.CRS(crs_proj_str); epsg_code = crs.to_epsg()
        if epsg_code: print(f"-> Estimated UTM CRS (EPSG:{epsg_code})"); return f"EPSG:{epsg_code}"
        else: print(f"-> Estimated UTM CRS (Proj): {crs_proj_str}"); return crs_proj_str
    except Exception as e: print(f"Warning: UTM estimation failed ({e}). Defaulting to EPSG:32614."); return 'EPSG:32614'
def save_empty_gdf(filepath, driver="GeoJSON", crs=WGS84): # Same as v19.1
     print(f"Saving empty GDF: {filepath}"); gdf = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=crs)
     try: gdf.to_file(filepath, driver=driver)
     except Exception as e: print(f"Error saving empty GDF {filepath}: {e}")
def save_empty_df(filepath, columns=None): # Same as v19.1
     print(f"Saving empty DF: {filepath}"); pd.DataFrame(columns=columns if columns else []).to_csv(filepath, index=False)
def safe_ox_features(polygon, tags): # Same as v19.1
    try: features = ox.features_from_polygon(polygon, tags); return features
    except Exception as e: print(f"  Warning: OSMnx fetch failed for {tags}. Err: {e}"); return gpd.GeoDataFrame()
def style_map(ax, title): # Same as v19.1
    ax.set_axis_off(); ax.set_title(title, fontsize=14, fontweight='bold'); ax.set_aspect('equal', adjustable='box')
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]; y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    if x_range > 0 and y_range > 0 and np.isfinite(x_range):
        scale_len_map = x_range * 0.1; possible_scales = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        if scale_len_map > 1e-6:
            scale_meters = min(possible_scales, key=lambda x: abs(x - scale_len_map))
            scale_len_display = (scale_meters / scale_len_map) * (x_range * 0.1)
            x_pos = ax.get_xlim()[0] + 0.05 * x_range; y_pos = ax.get_ylim()[0] + 0.05 * y_range
            ax.plot([x_pos, x_pos + scale_len_display], [y_pos, y_pos], color='black', linewidth=3, transform=ax.transData)
            ax.text(x_pos + scale_len_display / 2, y_pos - 0.01 * y_range, f'{scale_meters}m', horizontalalignment='center', verticalalignment='top', fontsize=10, transform=ax.transData)
        else: print("Warn: Map width too small for scale.")
    else: print("Warn: Invalid limits for scale.")
    ax.text(0.06, 0.94, 'N\n^', transform=ax.transAxes, ha='center', va='center', fontsize=14, fontweight='bold')

# -----------------------------------------------------------------------------
# Module A: Area Selection & Data Acquisition
# -----------------------------------------------------------------------------
# ... [Code identical to v19.1] ...
print("\n## Module A: Area Selection & Data Acquisition...")
module_A_start = time.time()
sel_config = config['area_selection']; selected_polygon = None
if sel_config['method'] == 'coordinates': lat_min, lon_min, lat_max, lon_max = sel_config['coordinates']; selected_polygon = box(lon_min, lat_min, lon_max, lat_max)
else: raise ValueError(f"Method '{sel_config['method']}' invalid.")
selected_gdf = gpd.GeoDataFrame(geometry=[selected_polygon], crs=WGS84); utm_crs = estimate_utm_crs(selected_gdf); area_selected_km2 = selected_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
print(f"Selected Area: {area_selected_km2:.2f} km²")
print("Fetching Census tracts..."); tracts_gdf = gpd.GeoDataFrame(columns=['GEOID', 'geometry'], geometry='geometry', crs=WGS84); temp_dir = None
try:
    state = config['data_acquisition']['state_code']; county = config['data_acquisition']['county_code']; tract_year = config['data_acquisition']['tract_year']
    tract_url = f"https://www2.census.gov/geo/tiger/TIGER{tract_year}/TRACT/tl_{tract_year}_{state}_tract.zip"; print(f"Downloading: {tract_url}"); response = requests.get(tract_url, stream=True, timeout=120); response.raise_for_status()
    temp_dir = tempfile.mkdtemp(); zip_path = os.path.join(temp_dir, f"tracts.zip"); extracted_dir = os.path.join(temp_dir, "extracted"); os.makedirs(extracted_dir, exist_ok=True)
    with open(zip_path, 'wb') as f: f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extracted_dir)
    shp_files = glob.glob(os.path.join(extracted_dir, '*.shp'));
    if not shp_files: raise FileNotFoundError("No .shp found.")
    all_tracts_in_state = gpd.read_file(shp_files[0]); print("Read tracts.");
    if all_tracts_in_state.crs is None: all_tracts_in_state.crs = WGS84
    else: all_tracts_in_state = all_tracts_in_state.to_crs(WGS84)
    county_tracts = all_tracts_in_state[all_tracts_in_state['COUNTYFP'] == county]; county_tracts = county_tracts[county_tracts.geometry.is_valid]
    intersecting_mask = county_tracts.geometry.intersects(selected_polygon); tracts_gdf = county_tracts[intersecting_mask].copy()
    geoid_col = next((col for col in ['GEOID', f'GEOID{str(tract_year)[-2:]}', 'GEOID20', 'GEOID10'] if col in tracts_gdf.columns), None)
    if geoid_col and geoid_col != 'GEOID': tracts_gdf.rename(columns={geoid_col: 'GEOID'}, inplace=True)
    elif 'GEOID' not in tracts_gdf.columns and all({'STATEFP', 'COUNTYFP', 'TRACTCE'}) <= set(tracts_gdf.columns): tracts_gdf['GEOID'] = tracts_gdf['STATEFP'] + tracts_gdf['COUNTYFP'] + tracts_gdf['TRACTCE']
    if 'GEOID' not in tracts_gdf.columns: raise ValueError("No GEOID."); tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str); print(f"Filtered to {len(tracts_gdf)} tracts.")
except Exception as e: print(f"ERROR fetching tracts: {e}")
finally:
    if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir); print("Cleaned temp tract dir.")
union_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=WGS84); buffered_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=WGS84); buffered_polygon = None; area_union_km2 = 0; area_buffered_km2 = 0
if not tracts_gdf.empty:
    tracts_gdf = tracts_gdf[tracts_gdf.geometry.is_valid]
    if not tracts_gdf.empty:
        union_of_tracts_poly = unary_union(tracts_gdf.geometry); union_gdf = gpd.GeoDataFrame(geometry=[union_of_tracts_poly], crs=WGS84)
        union_utm = union_gdf.to_crs(utm_crs); area_union_km2 = union_utm.geometry.iloc[0].area / 1e6
        buffer_m = sel_config['buffer_km'] * 1000; buffered_poly_utm = union_utm.geometry.iloc[0].buffer(buffer_m)
        buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84); buffered_polygon = buffered_gdf.geometry.iloc[0]; area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
        print(f"Union of {len(tracts_gdf)} tracts. Area: {area_union_km2:.2f} km²"); print(f"Buffered Union Area: {area_buffered_km2:.2f} km²")
    else: print("Warn: No valid tracts."); raise ValueError("No valid tracts.")
else:
    print("Using selected rect for buffer."); union_of_tracts_poly = selected_polygon; union_gdf = selected_gdf.copy(); area_union_km2 = area_selected_km2
    selected_utm = selected_gdf.to_crs(utm_crs); buffer_m = sel_config['buffer_km'] * 1000; buffered_poly_utm = selected_utm.geometry.iloc[0].buffer(buffer_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84); buffered_polygon = buffered_gdf.geometry.iloc[0]; area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
    print(f"Buffered Area: {area_buffered_km2:.2f} km²")
if buffered_polygon is None: raise ValueError("Buffered polygon needed for OSM fetch is not defined.")
print("Fetching OSM data..."); osm_config = config['data_acquisition']['osm_tags']; osm_gdf = gpd.GeoDataFrame()
try:
    osm_fetch_bbox = buffered_polygon.bounds; fetched_osm = ox.features_from_bbox(bbox=osm_fetch_bbox, tags=osm_config)
    osm_gdf = gpd.clip(fetched_osm, buffered_gdf); print(f"Fetched {len(osm_gdf)} OSM features."); osm_gdf = osm_gdf.reset_index()
    if isinstance(osm_gdf.columns, pd.MultiIndex): osm_gdf.columns = ['_'.join(map(str, col)).strip('_') for col in osm_gdf.columns.values]
    if 'osmid' not in osm_gdf.columns and 'element_type_osmid' in osm_gdf.columns: osm_gdf.rename(columns={'element_type_osmid': 'osmid'}, inplace=True)
    elif 'osmid' not in osm_gdf.columns and 'id' in osm_gdf.columns: osm_gdf.rename(columns={'id': 'osmid'}, inplace=True)
    if 'element_type' in osm_gdf.columns and 'osmid' in osm_gdf.columns: osm_gdf['unique_id'] = osm_gdf['element_type'].astype(str) + osm_gdf['osmid'].astype(str)
    elif 'osmid' in osm_gdf.columns: osm_gdf['unique_id'] = 'way' + osm_gdf['osmid'].astype(str)
    else: osm_gdf = osm_gdf.reset_index().rename(columns={'index':'generated_id'}); osm_gdf['unique_id'] = 'feat_' + osm_gdf['generated_id'].astype(str)
    osm_gdf = osm_gdf[~osm_gdf.geometry.isna()]; osm_gdf['geometry'] = osm_gdf.geometry.buffer(0); osm_gdf = osm_gdf[osm_gdf.geometry.is_valid]; print(f"Valid OSM features: {len(osm_gdf)}")
except Exception as e: print(f"ERROR fetching OSM: {e}")
print("Fetching Census data..."); cen_config = config['data_acquisition']
census_columns = ['GEOID'] + cen_config['census_variables'] + ['total_population', 'median_income', 'avg_household_size', 'employment_rate', 'pop_density', 'total_housing_units']
census_data_df = pd.DataFrame(columns=census_columns); census_data_fetched = False
if not tracts_gdf.empty:
    try:
        print(f"Cenpy query: {cen_config['census_product']}..."); conn = products.APIConnection(cen_config['census_product'])
        census_api_data = conn.query(cols=cen_config['census_variables'], geo_unit='tract:*', geo_filter={'state': state, 'county': county}, apikey=census_api_key)
        census_api_data['GEOID'] = census_api_data['state'] + census_api_data['county'] + census_api_data['tract']; census_api_data['GEOID'] = census_api_data['GEOID'].astype(str)
        census_api_data = census_api_data[census_api_data['GEOID'].isin(tracts_gdf['GEOID'])]; print(f"Fetched {len(census_api_data)} tracts via cenpy."); census_data_fetched = True
    except Exception as cenpy_e: print(f"WARN: cenpy failed: {cenpy_e}. Trying direct API...");
        # ... [Direct API fallback logic] ...
    if census_data_fetched and not census_api_data.empty:
        print("Processing Census data..."); tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str); census_api_data['GEOID'] = census_api_data['GEOID'].astype(str)
        for col in cen_config['census_variables']:
            if col in census_api_data.columns: census_api_data[col] = pd.to_numeric(census_api_data[col], errors='coerce')
        census_api_data['total_population'] = census_api_data.get('B01003_001E', 0); census_api_data['median_income'] = census_api_data.get('B19013_001E', 0)
        census_api_data['avg_household_size'] = census_api_data.get('B25010_001E', 0); census_api_data['total_housing_units'] = census_api_data.get('B25001_001E', 0)
        pop_in_labor_force = census_api_data.get('B23025_002E', pd.Series(0, index=census_api_data.index)); employed_pop = census_api_data.get('B23025_004E', pd.Series(0, index=census_api_data.index))
        census_api_data['employment_rate'] = np.where(pop_in_labor_force > 0, (employed_pop / pop_in_labor_force) * 100, 0)
        if not tracts_gdf.empty: # Ensure tracts_gdf exists before calculating area
             tracts_with_area = tracts_gdf.to_crs(utm_crs); tracts_with_area['area_sqkm'] = tracts_with_area.geometry.area / 1e6
             census_api_data = census_api_data.merge(tracts_with_area[['GEOID', 'area_sqkm']], on='GEOID', how='left')
             census_api_data['pop_density'] = np.where(census_api_data['area_sqkm'] > 0, census_api_data['total_population'] / census_api_data['area_sqkm'], 0)
        else: census_api_data['area_sqkm'] = 0; census_api_data['pop_density'] = 0; print("Warn: Cannot calc density.")
        cols_to_keep = ['GEOID', 'total_population', 'median_income', 'avg_household_size', 'employment_rate', 'pop_density', 'total_housing_units']
        existing_cols = [col for col in cols_to_keep if col in census_api_data.columns]; census_data_df = census_api_data[existing_cols].fillna(0); print("Census processing complete.")
    else: print("Census data not fetched/processed.")
else: print("No tracts, skipping Census fetch.")
print("Saving Module A outputs...")
selected_gdf.to_file(os.path.join(data_subdir, "area_selection.geojson"), driver="GeoJSON"); union_gdf.to_file(os.path.join(data_subdir, "union_of_tracts.geojson"), driver="GeoJSON"); buffered_gdf.to_file(os.path.join(data_subdir, "buffered_area.geojson"), driver="GeoJSON")
if not osm_gdf.empty: osm_gdf.to_file(os.path.join(data_subdir, "initial_osm_features.geojson"), driver="GeoJSON")
else: save_empty_gdf(os.path.join(data_subdir, "initial_osm_features.geojson"))
tracts_gdf.to_file(os.path.join(data_subdir, "census_tracts.geojson"), driver="GeoJSON"); census_data_df.to_csv(os.path.join(data_subdir, "census_data.csv"), index=False)
print("Generating setup visualizations...")
# Fig 1
fig1, ax1 = plt.subplots(figsize=(10, 10)); selected_gdf.to_crs(epsg=3857).plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3, label='Selected Area')
try: cx.add_basemap(ax1, crs='epsg:3857', source=cx.providers.CartoDB.Positron); style_map(ax1, 'Figure 1: Selected Study Area')
except Exception as e: print(f"Basemap Fig 1 failed: {e}"); ax1.set_title('Figure 1: Selected Study Area'); ax1.set_axis_off();
ax1.legend(); plt.savefig(os.path.join(viz_subdir, 'setup_map_1.png'), dpi=150, bbox_inches='tight'); plt.close(fig1)
# Fig 2
fig2, ax2 = plt.subplots(figsize=(10, 10)); union_plot_gdf = union_gdf.to_crs(epsg=3857); tracts_plot_gdf = tracts_gdf.to_crs(epsg=3857); selected_plot_gdf = selected_gdf.to_crs(epsg=3857)
osm_buildings_in_union = gpd.GeoDataFrame()
if not osm_gdf.empty and not union_gdf.empty: osm_buildings_in_union = gpd.clip(osm_gdf[osm_gdf['building'].notna()], union_gdf).to_crs(epsg=3857)
tracts_plot_gdf.plot(ax=ax2, facecolor='whitesmoke', edgecolor='gray', linewidth=0.5, label='Tracts', alpha=0.6)
if not osm_buildings_in_union.empty: osm_buildings_in_union.plot(ax=ax2, color='darkgray', alpha=0.7, linewidth=0.1, label='Buildings in Union')
union_plot_gdf.boundary.plot(ax=ax2, edgecolor='green', linewidth=2.5, label='Union of Tracts', zorder=4); selected_plot_gdf.boundary.plot(ax=ax2, edgecolor='red', linewidth=1.5, linestyle='--', label='Selected Area', zorder=5)
try: cx.add_basemap(ax2, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
except Exception as e: print(f"Basemap Fig 2 failed: {e}")
style_map(ax2, 'Figure 2: Tracts, Union, Buildings & Selection'); plt.savefig(os.path.join(viz_subdir, 'setup_map_2.png'), dpi=150, bbox_inches='tight'); plt.close(fig2)
# Fig 3
fig3, ax3 = plt.subplots(figsize=(10, 10)); buffered_plot_gdf = buffered_gdf.to_crs(epsg=3857); union_plot_gdf3 = union_gdf.to_crs(epsg=3857); all_osm_plot_gdf = osm_gdf.to_crs(epsg=3857)
store_footprints_gdf_placeholder = gpd.GeoDataFrame() # Placeholder
buffered_plot_gdf.plot(ax=ax3, facecolor='lightblue', edgecolor='blue', alpha=0.15, label=f"{sel_config['buffer_km']}km Buffer")
all_osm_plot_gdf[all_osm_plot_gdf['building'].notna()].plot(ax=ax3, color='gray', alpha=0.4, linewidth=0, label='Buildings in Buffer')
if not store_footprints_gdf_placeholder.empty: store_footprints_gdf_placeholder.to_crs(epsg=3857).plot(ax=ax3, facecolor='none', edgecolor='magenta', linewidth=1.0, label='Store Footprints (Initial)')
union_plot_gdf3.boundary.plot(ax=ax3, edgecolor='green', linewidth=2.5, label='Union of Tracts')
try: cx.add_basemap(ax3, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
except Exception as e: print(f"Basemap Fig 3 failed: {e}")
style_map(ax3, 'Figure 3: Buffer & All Buildings (Stores TBD)'); plt.savefig(os.path.join(viz_subdir, 'setup_map_3_initial.png'), dpi=150, bbox_inches='tight'); plt.close(fig3)
stats_a = {"Selected Area (km²)": area_selected_km2,"Union Area (km²)": area_union_km2,"Buffered Area (km²)": area_buffered_km2,"OSM Features Fetched": len(osm_gdf),"Tracts Fetched": len(tracts_gdf),"Tracts with Data": len(census_data_df) if 'GEOID' in census_data_df else 0, "Total Pop (Tracts)": census_data_df['total_population'].sum() if 'total_population' in census_data_df else 0,"Avg Income (Tracts)": census_data_df['median_income'].mean() if 'median_income' in census_data_df else 0,}
stats_a_df = pd.DataFrame.from_dict(stats_a, orient='index', columns=['Value']); stats_a_df.to_csv(os.path.join(stats_subdir, 'setup_stats.csv'))
print("Module A Stats:"); print(stats_a_df)
module_A_time = time.time() - module_A_start
print(f"Module A completed in {module_A_time:.2f} seconds.")


# -----------------------------------------------------------------------------
# Module B: Building Classification & Store Identification
# -----------------------------------------------------------------------------
print("\n## Module B: Building Classification & Store Identification...")
module_B_start = time.time()
buildings_all = gpd.GeoDataFrame(); stores_gdf = gpd.GeoDataFrame(); buildings_in_union_area = gpd.GeoDataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    osm_file = os.path.join(data_subdir, "initial_osm_features.geojson"); union_file = os.path.join(data_subdir, "union_of_tracts.geojson"); buffered_file = os.path.join(data_subdir, "buffered_area.geojson")
    if not os.path.exists(osm_file): raise FileNotFoundError("initial_osm_features.geojson missing.")
    if not os.path.exists(union_file): raise FileNotFoundError("union_of_tracts.geojson missing.")
    if not os.path.exists(buffered_file): raise FileNotFoundError("buffered_area.geojson missing.")
    buildings_all = gpd.read_file(osm_file); union_gdf = gpd.read_file(union_file); buffered_gdf = gpd.read_file(buffered_file)
    if buildings_all.empty or union_gdf.empty: raise ValueError("Input data empty.")
    if 'unique_id' not in buildings_all.columns:
        if 'element_type' in buildings_all.columns and 'osmid' in buildings_all.columns: buildings_all['unique_id'] = buildings_all['element_type'] + buildings_all['osmid'].astype(str)
        elif 'osmid' in buildings_all.columns: buildings_all['unique_id'] = 'way' + buildings_all['osmid'].astype(str)
        else: buildings_all = buildings_all.reset_index().rename(columns={'index':'unique_id'}); buildings_all['unique_id'] = 'feat_' + buildings_all['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading initial OSM features: {e}. Skipping Module B.")

if not buildings_all.empty and not union_gdf.empty:
    cls_config = config['building_classification']
    # --- Detailed Rule-Based Classification Function ---
    def classify_building_rules_detailed(buildings_gdf, buffer_poly, config): # Same as v19.1
        print("--- Starting Detailed Rule-Based Classification ---"); buildings = buildings_gdf.copy(); rules = config['building_classification']['rule_based_parameters']; buildings['residential'] = 'unknown'
        if 'building_area_m2' not in buildings.columns: print("  Calculating building areas..."); buildings['building_area_m2'] = buildings.to_crs(utm_crs).geometry.area
        print("  Applying OSM tag rules..."); res_tags = rules.get('residential_building_tags', []); nonres_tags = rules.get('nonresidential_building_tags', [])
        res_keywords = rules.get('residential_name_keywords', []); nonres_keywords = rules.get('nonresidential_name_keywords', [])
        if 'building' in buildings.columns: buildings.loc[buildings['building'].isin(res_tags), 'residential'] = 'yes'; buildings.loc[buildings['building'].isin(nonres_tags), 'residential'] = 'no'
        if 'shop' in buildings.columns: buildings.loc[buildings['shop'].notna(), 'residential'] = 'no'
        if 'amenity' in buildings.columns: buildings.loc[buildings['amenity'].notna(), 'residential'] = 'no'
        if 'office' in buildings.columns: buildings.loc[buildings['office'].notna(), 'residential'] = 'no'
        if 'tourism' in buildings.columns: buildings.loc[buildings['tourism'].notna(), 'residential'] = 'no'
        if 'leisure' in buildings.columns: buildings.loc[buildings['leisure'].notna(), 'residential'] = 'no'
        if 'name' in buildings.columns:
            nonres_name_pattern = r'(' + '|'.join(nonres_keywords) + r')' if nonres_keywords else '^$'; nonres_name_mask = buildings['name'].str.contains(nonres_name_pattern, case=False, na=False, regex=True)
            buildings.loc[nonres_name_mask, 'residential'] = 'no'
            if res_keywords: res_name_pattern = r'(' + '|'.join(res_keywords) + r')'; res_name_mask = buildings['name'].str.contains(res_name_pattern, case=False, na=False, regex=True); buildings.loc[res_name_mask & buildings['residential'].isin(['unknown', 'likely_yes']), 'residential'] = 'yes'
        print("  Applying area rules..."); small_thresh = rules.get('likely_max_residential_area_m2', 500); large_thresh = rules.get('min_nonresidential_area_m2', 1500); tiny_thresh = rules.get('min_residential_area_m2', 30)
        likely_res_mask = (buildings['building_area_m2'] < small_thresh) & (buildings['residential'] == 'unknown'); likely_nonres_mask = (buildings['building_area_m2'] > large_thresh) & (buildings['residential'] == 'unknown')
        buildings.loc[likely_res_mask, 'residential'] = 'likely_yes'; buildings.loc[likely_nonres_mask, 'residential'] = 'likely_no'
        tiny_mask = (buildings['building_area_m2'] < tiny_thresh) & (buildings['residential'] == 'likely_yes') & (~buildings['building'].isin(['house', 'bungalow', 'shed', 'garage', 'detached']))
        buildings.loc[tiny_mask, 'residential'] = 'no'
        print("  Applying road proximity rules...");
        try:
            if not isinstance(buffer_poly, (Polygon, MultiPolygon)): buffer_poly = unary_union(buffer_poly)
            if 'highway' not in buildings_gdf.columns: print("  Skipping road proximity: 'highway' tag not found.")
            else:
                roads = buildings_gdf[buildings_gdf['highway'].notna()]
                if not roads.empty:
                    roads_proj = roads.to_crs(utm_crs); buildings_proj = buildings.to_crs(utm_crs); road_buffer_m = rules.get('road_buffer_meters', 30)
                    res_roads = roads_proj[roads_proj['highway'].isin(rules.get('residential_road_types', []))]; near_res_indices = set()
                    if not res_roads.empty: res_buffer_geom = unary_union(res_roads.geometry.buffer(road_buffer_m)); near_res_indices = set(buildings_proj[buildings_proj.geometry.intersects(res_buffer_geom)].index)
                    buildings.loc[buildings.index.isin(near_res_indices) & (buildings['residential'] == 'unknown'), 'residential'] = 'likely_yes'
                    primary_roads = roads_proj[roads_proj['highway'].isin(rules.get('primary_road_types', []))]
                    if not primary_roads.empty:
                        primary_buffer_geom = unary_union(primary_roads.geometry.buffer(road_buffer_m)); near_primary_indices = set(buildings_proj[buildings_proj.geometry.intersects(primary_buffer_geom)].index)
                        mask_primary = buildings.index.isin(near_primary_indices) & buildings['residential'].isin(['unknown', 'likely_no']) & (~buildings.index.isin(near_res_indices))
                        buildings.loc[mask_primary, 'residential'] = 'likely_no'
                else: print("  No relevant road types found for proximity.")
        except Exception as e: print(f"  Road proximity failed: {e}")
        print("  Applying landuse context rules...");
        try:
            if 'landuse' not in buildings_gdf.columns: print("  Skipping landuse context: 'landuse' tag not found.")
            else:
                landuse = buildings_gdf[buildings_gdf['landuse'].notna()]
                if not landuse.empty:
                    landuse_proj = landuse.to_crs(utm_crs); buildings_proj = buildings.to_crs(utm_crs)
                    res_landuse_names = rules.get('residential_landuse', []); nonres_landuse_names = rules.get('nonresidential_landuse', [])
                    res_landuse = landuse_proj[landuse_proj['landuse'].isin(res_landuse_names)]
                    if not res_landuse.empty:
                        res_lu_geom = unary_union(res_landuse.geometry); in_res_indices = buildings_proj[buildings_proj.geometry.within(res_lu_geom)].index
                        buildings.loc[buildings.index.isin(in_res_indices) & buildings['residential'].isin(['unknown', 'likely_yes']), 'residential'] = 'yes'
                    nonres_landuse = landuse_proj[landuse_proj['landuse'].isin(nonres_landuse_names)]
                    if not nonres_landuse.empty:
                        nonres_lu_geom = unary_union(nonres_landuse.geometry); in_nonres_indices = buildings_proj[buildings_proj.geometry.within(nonres_lu_geom)].index
                        buildings.loc[buildings.index.isin(in_nonres_indices) & buildings['residential'].isin(['unknown', 'likely_no']), 'residential'] = 'no'
                else: print("  No landuse features found.")
        except Exception as e: print(f"  Landuse context failed: {e}")
        buildings['residential'] = buildings['residential'].replace({'likely_yes': 'yes', 'likely_no': 'no', 'unknown': 'no'})
        print("--- Detailed Rule-Based Classification Finished ---"); return buildings

    # --- Store Identification Logic ---
    store_cfg = cls_config.get('store_tags', {}); store_keywords_pattern = "|".join(store_cfg.get('name_keywords', []))
    def identify_store_type(row): # Same as v19.1
        tags = row.to_dict(); name = str(tags.get('name', '')).lower(); shop = str(tags.get('shop', '')).lower(); bldg = str(tags.get('building', '')).lower(); amenity = str(tags.get('amenity', '')).lower()
        if 'walmart' in name: return 'Walmart';
        if 'target' in name: return 'Target';
        if 'kroger' in name: return 'Kroger';
        if 'h-e-b' in name or 'heb' in name: return 'H-E-B';
        if 'costco' in name: return 'Costco';
        if 'amazon' in name: return 'Warehouse/Fulfillment';
        if 'distribution center' in name or 'fulfillment' in name: return 'Warehouse/Fulfillment';
        if 'warehouse' in name or bldg == 'warehouse': return 'Warehouse/Fulfillment';
        if 'supercenter' in name or 'supermarket' in shop or 'grocery' in name or 'market' in name: return 'Retail Hub';
        if 'department_store' in shop or 'mall' in shop or 'wholesale' in shop: return 'Retail Hub';
        if shop in store_cfg.get('shop', []): return 'Retail Hub';
        if amenity in store_cfg.get('amenity', []): return 'Retail Hub';
        if store_keywords_pattern and re.search(store_keywords_pattern, name, re.IGNORECASE): return 'Retail Hub / Warehouse';
        return None

    # --- Apply Classification and Store ID ---
    if 'building_area_m2' not in buildings_all.columns: buildings_all['building_area_m2'] = buildings_all.to_crs(utm_crs).geometry.area
    classification_method = cls_config.get('method', 'rule_based')
    if classification_method == 'rule_based':
        print("Classifying buildings using Detailed Rules...")
        buildings_all = classify_building_rules_detailed(buildings_all, buffered_polygon, config)
    else: print(f"Warning: Method '{classification_method}' not impl."); buildings_all['residential'] = 'no'
    print("Identifying stores...")
    buildings_all['store_type'] = buildings_all.apply(identify_store_type, axis=1)
    buildings_all['is_store'] = buildings_all['store_type'].notna()
    buildings_all.loc[buildings_all['is_store'], 'residential'] = 'no'
    stores_gdf = buildings_all[buildings_all['is_store']].copy()
    if not stores_gdf.empty:
        # stores_gdf['geometry_poly'] = stores_gdf['geometry'] # Keep original polygon - **REMOVED**
        stores_gdf['geometry'] = stores_gdf.geometry.centroid # Set active geometry to centroid
        stores_gdf = stores_gdf.set_geometry('geometry') # Explicitly set active geometry
        stores_gdf = stores_gdf.reset_index(drop=True).reset_index().rename(columns={'index':'store_seq_id'})
        stores_gdf['store_id'] = 'Store_' + stores_gdf['store_seq_id'].astype(str)
        # Select columns, EXCLUDING geometry_poly
        stores_gdf = stores_gdf[['unique_id', 'osmid', 'store_id', 'name', 'shop', 'amenity', 'building', 'store_type', 'geometry']].copy()
    print(f"Identified {len(stores_gdf)} potential stores.")
    if not stores_gdf.empty: print("Store types found:\n", stores_gdf['store_type'].value_counts())
    else: print("No stores identified.")
    print("Saving Module B outputs...")
    cols_to_save_b = [col for col in buildings_all.columns if col not in ['knn_details', 'residential_rule']]
    buildings_save_b = buildings_all[cols_to_save_b].copy()
    for col in buildings_save_b.select_dtypes(include=['object']).columns:
        mask = buildings_save_b[col].apply(lambda x: isinstance(x, (list, dict)))
        if mask.any(): buildings_save_b.loc[mask, col] = buildings_save_b.loc[mask, col].astype(str)
    buildings_save_b.to_file(os.path.join(data_subdir, "classified_buildings.geojson"), driver="GeoJSON")
    stores_gdf.to_file(os.path.join(data_subdir, "stores.geojson"), driver="GeoJSON") # Now saves only POINT geometry

    print("Clipping buildings to union of tracts area...")
    buildings_in_union_area = gpd.clip(buildings_all, union_gdf)
    print(f"Buildings within union: {len(buildings_in_union_area)}")
    print("Generating classification visualization (Fig 4 - Stores as Points)...")
    fig_b4, ax_b4 = plt.subplots(figsize=(12, 12)); plot_data_b4 = buildings_in_union_area.to_crs(epsg=3857); union_plot_gdf_b4 = union_gdf.to_crs(epsg=3857); stores_plot_b4 = stores_gdf.to_crs(epsg=3857) # Stores are points
    plot_data_b4_poly = plot_data_b4[plot_data_b4.geom_type.isin(['Polygon', 'MultiPolygon'])]
    non_res_plot = plot_data_b4_poly[plot_data_b4_poly['residential'] == 'no']
    res_plot = plot_data_b4_poly[plot_data_b4_poly['residential'] == 'yes']
    if not non_res_plot.empty: non_res_plot.plot(ax=ax_b4, color='gray', alpha=0.6, label='Non-Residential Buildings', linewidth=0.5)
    if not res_plot.empty: res_plot.plot(ax=ax_b4, color='green', alpha=0.7, label='Residential Buildings', linewidth=0.5)
    union_plot_gdf_b4.boundary.plot(ax=ax_b4, edgecolor='black', linewidth=2.0, label='Union of Tracts', zorder=4)
    if not stores_plot_b4.empty: stores_plot_b4.plot(ax=ax_b4, color='red', marker='*', markersize=100, label='Stores (in Buffer)', zorder=5)
    try: cx.add_basemap(ax_b4, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    style_map(ax_b4, 'Figure 4: Classification (Union) & Store Locations (Buffer)')
    handles_b4, labels_b4 = ax_b4.get_legend_handles_labels(); unique_items_b4 = dict(zip(labels_b4, handles_b4)); ax_b4.legend(unique_items_b4.values(), unique_items_b4.keys(), loc='upper left')
    plt.savefig(os.path.join(viz_subdir, 'classification_map.png'), dpi=150, bbox_inches='tight'); plt.close(fig_b4)
    print("Regenerating setup visualization (Fig 3 with identified store footprints)...")
    fig3, ax3 = plt.subplots(figsize=(10, 10)); buffered_plot_gdf = gpd.read_file(os.path.join(data_subdir, "buffered_area.geojson")).to_crs(epsg=3857); union_plot_gdf3 = union_gdf.to_crs(epsg=3857); all_osm_plot_gdf = buildings_all.to_crs(epsg=3857)
    store_footprints_plot_fig3 = all_osm_plot_gdf[all_osm_plot_gdf['is_store']] # Get footprints of identified stores
    buffered_plot_gdf.plot(ax=ax3, facecolor='lightblue', edgecolor='blue', alpha=0.15, label=f"{config['area_selection']['buffer_km']}km Buffer")
    all_osm_plot_gdf_no_stores = all_osm_plot_gdf[(all_osm_plot_gdf['building'].notna()) & (~all_osm_plot_gdf['is_store']) & all_osm_plot_gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
    if not all_osm_plot_gdf_no_stores.empty: all_osm_plot_gdf_no_stores.plot(ax=ax3, color='gray', alpha=0.4, linewidth=0, label='Other Buildings in Buffer')
    if not store_footprints_plot_fig3.empty: store_footprints_plot_fig3.plot(ax=ax3, facecolor='none', edgecolor='magenta', linewidth=1.0, label='Store Footprints')
    union_plot_gdf3.boundary.plot(ax=ax3, edgecolor='green', linewidth=2.5, label='Union of Tracts')
    try: cx.add_basemap(ax3, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap Fig 3 failed: {e}")
    style_map(ax3, 'Figure 3: Buffer, All Buildings, & Identified Store Footprints'); plt.savefig(os.path.join(viz_subdir, 'setup_map_3_final.png'), dpi=150, bbox_inches='tight'); plt.close(fig3)
    stats_b = {"Total Buildings Classified (buffer)": len(buildings_all), "Buildings in Union Area": len(buildings_in_union_area), "Residential (Union)": len(buildings_in_union_area[buildings_in_union_area['residential'] == 'yes']), "Non-Residential (Union)": len(buildings_in_union_area[buildings_in_union_area['residential'] == 'no']), "Stores Identified (buffer)": len(stores_gdf)}
    stats_b_df = pd.DataFrame.from_dict(stats_b, orient='index', columns=['Value']); stats_b_df.to_csv(os.path.join(stats_subdir, 'classification_stats.csv'))
    print("Module B Stats:"); print(stats_b_df)
else:
    print("Skipping Module B because input data is empty.")
    save_empty_gdf(os.path.join(data_subdir, "classified_buildings.geojson")); save_empty_gdf(os.path.join(data_subdir, "stores.geojson")); save_empty_df(os.path.join(stats_subdir, 'classification_stats.csv'), columns=['Value'])
module_B_time = time.time() - module_B_start
print(f"Module B completed in {module_B_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module C: Building Height Estimation
# -----------------------------------------------------------------------------
# ... [Code identical to v19.1] ...
print("\n## Module C: Building Height Estimation...")
module_C_start = time.time()
buildings_with_heights = gpd.GeoDataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    buildings_classified = gpd.read_file(os.path.join(data_subdir, "classified_buildings.geojson"))
    union_gdf = gpd.read_file(os.path.join(data_subdir, "union_of_tracts.geojson"))
    if buildings_classified.empty or union_gdf.empty: raise ValueError("Input data missing.")
    if 'unique_id' not in buildings_classified.columns:
        if 'element_type' in buildings_classified.columns and 'osmid' in buildings_classified.columns: buildings_classified['unique_id'] = buildings_classified['element_type'] + buildings_classified['osmid'].astype(str)
        elif 'osmid' in buildings_classified.columns: buildings_classified['unique_id'] = 'way' + buildings_classified['osmid'].astype(str)
        else: buildings_classified = buildings_classified.reset_index().rename(columns={'index':'unique_id'}); buildings_classified['unique_id'] = 'bldg_' + buildings_classified['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for height estimation: {e}. Skipping Module C.")

if not buildings_classified.empty and not union_gdf.empty:
    print(f"Clipping {len(buildings_classified)} features to union area for height processing...")
    buildings_in_union = gpd.clip(buildings_classified, union_gdf)
    buildings_in_union = buildings_in_union[buildings_in_union['building'].notna()].copy()
    print(f"Processing heights for {len(buildings_in_union)} buildings within union.")
    if not buildings_in_union.empty:
        h_config = config['height_estimation']
        def extract_known_height(row, meters_per_level): # Same as v19.1
            height_val = row.get('height', None); levels_val = row.get('building:levels', None)
            try:
                if pd.notna(height_val):
                    height_str = str(height_val).lower().split(' ')[0]; height_str = re.split(r'[~\-,]', height_str)[0]
                    height_m = float(height_str);
                    if height_m > 0: return height_m
            except: pass
            try:
                if pd.notna(levels_val):
                    levels_str = str(levels_val).split(',')[0].split(';')[0]
                    if '-' in levels_str: levels_val = float(levels_str.split('-')[-1])
                    elif levels_str.replace('.','',1).isdigit(): levels_val = float(levels_str)
                    else: levels_val = None
                    if levels_val is not None: height_m = max(1.0, levels_val) * meters_per_level;
                    if height_m > 0: return height_m
            except: pass
            return None
        def estimate_missing_heights(buildings_to_estimate, known_buildings_df, config): # Same as v19.1
            h_config = config['height_estimation']; k = min(h_config['knn_neighbors'], len(known_buildings_df))
            if k <= 0 or len(buildings_to_estimate) == 0: return pd.Series(dtype=float), pd.Series(dtype=object)
            features = []; known_buildings_df_copy = known_buildings_df.copy(); buildings_to_estimate_copy = buildings_to_estimate.copy()
            if h_config['use_area_feature'] and 'building_area_m2' in buildings_to_estimate_copy.columns: features.append('building_area_m2')
            buildings_to_estimate_proj = buildings_to_estimate_copy.to_crs(utm_crs); known_buildings_proj = known_buildings_df_copy.to_crs(utm_crs)
            buildings_to_estimate_copy['centroid_x'] = buildings_to_estimate_proj.geometry.centroid.x; buildings_to_estimate_copy['centroid_y'] = buildings_to_estimate_proj.geometry.centroid.y
            known_buildings_df_copy['centroid_x'] = known_buildings_proj.geometry.centroid.x; known_buildings_df_copy['centroid_y'] = known_buildings_proj.geometry.centroid.y
            features.extend(['centroid_x', 'centroid_y'])
            if not features: print("Warning: No k-NN features."); return pd.Series(dtype=float), pd.Series(dtype=object)
            X_known = known_buildings_df_copy[features].fillna(0).values; y_known = known_buildings_df_copy['height_m'].values
            X_unknown = buildings_to_estimate_copy[features].fillna(0).values
            if X_known.shape[1] != X_unknown.shape[1]: print("ERROR: Feature shape mismatch."); return pd.Series(dtype=float), pd.Series(dtype=object)
            scaler = StandardScaler(); X_known_scaled = scaler.fit_transform(X_known); X_unknown_scaled = scaler.transform(X_unknown)
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance'); knn.fit(X_known_scaled, y_known)
            predicted_heights = knn.predict(X_unknown_scaled)
            distances, indices = knn.kneighbors(X_unknown_scaled); neighbor_details = []
            for i in range(len(indices)):
                valid_indices = [idx for idx in indices[i] if idx < len(known_buildings_df_copy)]; neighbor_ids = []; neighbor_heights = []
                if valid_indices:
                     neighbor_ids = known_buildings_df_copy.iloc[valid_indices]['unique_id'].tolist()
                     neighbor_heights = y_known[valid_indices].tolist()
                neighbor_details.append({'ids': neighbor_ids, 'heights': [round(h,1) for h in neighbor_heights]})
            return pd.Series(predicted_heights, index=buildings_to_estimate_copy.index), pd.Series(neighbor_details, index=buildings_to_estimate_copy.index)

        print("Applying height logic to buildings within union...")
        buildings_with_heights = buildings_in_union.copy()
        if 'building_area_m2' not in buildings_with_heights.columns: buildings_with_heights['building_area_m2'] = buildings_with_heights.to_crs(utm_crs).geometry.area
        buildings_with_heights['height_m'] = buildings_with_heights.apply(lambda row: extract_known_height(row, h_config['meters_per_level']), axis=1)
        buildings_with_heights['height_source'] = np.where(buildings_with_heights['height_m'].notna(), 'tag', None)
        known_height_buildings = buildings_with_heights[buildings_with_heights['height_source'] == 'tag'].copy()
        unknown_height_buildings = buildings_with_heights[buildings_with_heights['height_source'].isna()].copy()
        buildings_with_heights['knn_details'] = None
        if len(known_height_buildings) >= h_config['knn_neighbors'] and not unknown_height_buildings.empty:
            print(f"Estimating heights for {len(unknown_height_buildings)} buildings via k-NN...")
            knn_preds, knn_details_series = estimate_missing_heights(unknown_height_buildings, known_height_buildings, config)
            if not knn_preds.empty:
                buildings_with_heights.loc[knn_preds.index, 'height_m'] = knn_preds
                buildings_with_heights.loc[knn_preds.index, 'height_source'] = 'knn'
                details_json = knn_details_series.apply(lambda x: json.dumps(x) if x and isinstance(x, dict) else None)
                buildings_with_heights.loc[knn_details_series.index, 'knn_details'] = details_json
        else: print(f"Skipping k-NN estimation: Known ({len(known_height_buildings)}), Unknown ({len(unknown_height_buildings)}).")
        default_mask = buildings_with_heights['height_source'].isna()
        buildings_with_heights.loc[default_mask, 'height_m'] = h_config['default_height_m']; buildings_with_heights.loc[default_mask, 'height_source'] = 'default'
        buildings_with_heights['height_m'] = buildings_with_heights['height_m'].fillna(h_config['default_height_m']).clip(lower=1.0, upper=h_config['max_height_cap_m'])
        print("Saving Module C outputs...")
        if 'knn_details' in buildings_with_heights.columns: buildings_with_heights['knn_details'] = buildings_with_heights['knn_details'].astype(str)
        buildings_with_heights.to_file(os.path.join(data_subdir, "buildings_with_heights.geojson"), driver="GeoJSON")
        print("Generating height visualization...")
        fig_c, ax_c = plt.subplots(1, 2, figsize=(18, 8)); plot_data_c = buildings_with_heights.to_crs(epsg=3857)
        if not plot_data_c.empty:
            cmap_height = 'plasma'; norm_height = Normalize(vmin=plot_data_c['height_m'].min(), vmax=plot_data_c['height_m'].quantile(0.98))
            plot_data_polys = plot_data_c[plot_data_c.geom_type.isin(['Polygon', 'MultiPolygon'])]
            known_polys = plot_data_polys[plot_data_polys['height_source'] == 'tag']
            estimated_polys = plot_data_polys[plot_data_polys['height_source'] != 'tag']
            if not estimated_polys.empty: estimated_polys.plot(column='height_m', cmap=cmap_height, norm=norm_height, ax=ax_c[0], alpha=0.7, edgecolor='gray', linewidth=0.2, linestyle='--')
            if not known_polys.empty: known_polys.plot(column='height_m', cmap=cmap_height, norm=norm_height, legend=False, ax=ax_c[0], alpha=0.9, edgecolor='black', linewidth=0.3)
            sm = ScalarMappable(cmap=cmap_height, norm=norm_height); sm.set_array([]); cbar = fig_c.colorbar(sm, ax=ax_c[0], shrink=0.7); cbar.set_label("Height (m)")
            try: cx.add_basemap(ax_c[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
            except Exception as e: print(f"Basemap failed: {e}")
            style_map(ax_c[0], 'Building Heights (Within Union Area)')
            sns.histplot(plot_data_c['height_m'], bins=30, kde=True, ax=ax_c[1]); ax_c[1].set_title('Distribution of Building Heights'); ax_c[1].set_xlabel('Height (m)'); ax_c[1].set_ylabel('Num Buildings')
        else: ax_c[0].text(0.5, 0.5, "No buildings", ha='center'); ax_c[1].text(0.5, 0.5, "No data", ha='center')
        plt.tight_layout(); plt.savefig(os.path.join(viz_subdir, 'height_visualization.png'), dpi=150, bbox_inches='tight'); plt.close(fig_c)
        stats_c = buildings_with_heights['height_m'].describe().to_dict(); stats_c.update(buildings_with_heights['height_source'].value_counts().rename(lambda x: f"count_{x}").to_dict())
        stats_c_df = pd.DataFrame.from_dict(stats_c, orient='index', columns=['Value']); stats_c_df.to_csv(os.path.join(stats_subdir, 'height_stats.csv'))
        print("Module C Stats:"); print(stats_c_df)
    else:
        print("Skipping height processing as no buildings are within the union area.")
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson"))
        save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'), columns=['Value'])
else:
    print("Skipping Module C."); save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson")); save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'), columns=['Value'])
module_C_time = time.time() - module_C_start
print(f"Module C completed in {module_C_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module D: Population Allocation
# -----------------------------------------------------------------------------
# ... [Code as in v19.1] ...
print("\n## Module D: Population Allocation...")
module_D_start = time.time()
buildings_with_pop = gpd.GeoDataFrame()
tracts_with_data_d = gpd.GeoDataFrame(); census_data_df_d = pd.DataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    buildings_with_heights = gpd.read_file(os.path.join(data_subdir, "buildings_with_heights.geojson")) # Contains buildings in union
    tracts_gdf_d = gpd.read_file(os.path.join(data_subdir, "census_tracts.geojson"))
    census_data_df_d = pd.read_csv(os.path.join(data_subdir, "census_data.csv"))
    if buildings_with_heights.empty or tracts_gdf_d.empty or census_data_df_d.empty or 'GEOID' not in census_data_df_d.columns: raise ValueError("Input data missing/invalid.")
    if 'unique_id' not in buildings_with_heights.columns:
        if 'element_type' in buildings_with_heights.columns and 'osmid' in buildings_with_heights.columns: buildings_with_heights['unique_id'] = buildings_with_heights['element_type'] + buildings_with_heights['osmid'].astype(str)
        elif 'osmid' in buildings_with_heights.columns: buildings_with_heights['unique_id'] = 'way' + buildings_with_heights['osmid'].astype(str)
        else: buildings_with_heights = buildings_with_heights.reset_index().rename(columns={'index':'unique_id'}); buildings_with_heights['unique_id'] = 'bldg_' + buildings_with_heights['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for pop allocation: {e}. Skipping Module D.")

if not buildings_with_heights.empty and not tracts_gdf_d.empty and not census_data_df_d.empty:
    pop_config = config['population_allocation']
    residential_buildings = buildings_with_heights[buildings_with_heights['residential'] == 'yes'].copy()
    if not residential_buildings.empty:
        if 'building_area_m2' not in residential_buildings.columns: residential_buildings['building_area_m2'] = residential_buildings.to_crs(utm_crs).geometry.area
        if 'height_m' not in residential_buildings.columns: residential_buildings['height_m'] = config['height_estimation']['default_height_m']
        residential_buildings['building_volume_m3'] = (residential_buildings['building_area_m2'] * residential_buildings['height_m']).clip(lower=1.0)
        print(f"Performing allocation for {len(residential_buildings)} residential buildings.")
        tracts_gdf_d['GEOID'] = tracts_gdf_d['GEOID'].astype(str); census_data_df_d['GEOID'] = census_data_df_d['GEOID'].astype(str)
        tracts_with_data_d = tracts_gdf_d.merge(census_data_df_d, on='GEOID', how='left')
        tracts_with_data_d['total_population'] = pd.to_numeric(tracts_with_data_d['total_population'], errors='coerce').fillna(0) * pop_config['population_scale_factor']
        tracts_with_data_d['avg_household_size'] = pd.to_numeric(tracts_with_data_d['avg_household_size'], errors='coerce')
        default_hh_size = pop_config.get('avg_household_size_override') if pop_config.get('avg_household_size_override') else 2.5
        tracts_with_data_d['avg_household_size'] = tracts_with_data_d['avg_household_size'].fillna(default_hh_size).clip(lower=1.0)
        residential_buildings = residential_buildings.to_crs(tracts_with_data_d.crs)
        print("Performing spatial join (predicate='intersects')...")
        bldg_tract_join = gpd.sjoin(residential_buildings, tracts_with_data_d[['GEOID', 'total_population', 'avg_household_size', 'geometry']], how='left', predicate='intersects')
        bldg_tract_join = bldg_tract_join.loc[~bldg_tract_join.index.duplicated(keep='first')]
        missing_tract_pop = bldg_tract_join['GEOID'].isna().sum()
        if missing_tract_pop > 0:
             print(f"Assigning nearest tract data to {missing_tract_pop} buildings...")
             missing_indices = bldg_tract_join[bldg_tract_join['GEOID'].isna()].index
             if not missing_indices.empty:
                  tracts_for_nearest = tracts_with_data_d[['GEOID', 'total_population', 'avg_household_size', 'geometry']].copy(); tracts_for_nearest['GEOID'] = tracts_for_nearest['GEOID'].astype(str)
                  missing_gdf_to_join = bldg_tract_join.loc[missing_indices].drop(columns=['index_right', 'GEOID','total_population', 'avg_household_size'], errors='ignore').copy()
                  if not missing_gdf_to_join.empty:
                      print(f"Running sjoin_nearest for {len(missing_gdf_to_join)} buildings...")
                      missing_gdf_to_join = missing_gdf_to_join.to_crs(tracts_for_nearest.crs)
                      nearest_join = gpd.sjoin_nearest(missing_gdf_to_join, tracts_for_nearest, how='left')
                      print(f"sjoin_nearest completed.")
                      nearest_join = nearest_join.loc[~nearest_join.index.duplicated(keep='first')]
                      geo_col_right = 'GEOID_right' if 'GEOID_right' in nearest_join.columns else 'GEOID'
                      pop_col_right = 'total_population_right' if 'total_population_right' in nearest_join.columns else 'total_population'
                      hh_col_right = 'avg_household_size_right' if 'avg_household_size_right' in nearest_join.columns else 'avg_household_size'
                      if geo_col_right in nearest_join.columns:
                          update_data = nearest_join[[geo_col_right, pop_col_right, hh_col_right]].rename(columns={geo_col_right: 'GEOID', pop_col_right: 'total_population', hh_col_right: 'avg_household_size'})
                          update_data.index = missing_indices
                          bldg_tract_join.update(update_data[['GEOID', 'total_population', 'avg_household_size']], overwrite=True)
                          print(f"Successfully updated {len(update_data.dropna())} buildings.")
                      else: print("ERROR: Columns not found in nearest_join.")
                  else: print("No missing buildings found.")
        bldg_tract_join['GEOID'] = bldg_tract_join['GEOID'].astype(str); tract_volume_sum = bldg_tract_join.groupby('GEOID')['building_volume_m3'].sum().reset_index(); tract_volume_sum.rename(columns={'building_volume_m3': 'total_tract_res_volume_m3'}, inplace=True); tract_volume_sum['GEOID'] = tract_volume_sum['GEOID'].astype(str)
        bldg_tract_join = bldg_tract_join.merge(tract_volume_sum, on='GEOID', how='left'); bldg_tract_join['total_tract_res_volume_m3'] = bldg_tract_join['total_tract_res_volume_m3'].fillna(1.0).clip(lower=1.0)
        bldg_tract_join['volume_share'] = bldg_tract_join['building_volume_m3'] / bldg_tract_join['total_tract_res_volume_m3']
        bldg_tract_join['allocated_population'] = bldg_tract_join['total_population'] * bldg_tract_join['volume_share']
        tract_alloc_check = bldg_tract_join.groupby('GEOID').agg(alloc_pop_sum=('allocated_population', 'sum'), tract_pop_target=('total_population', 'first')).reset_index(); tract_alloc_check['GEOID'] = tract_alloc_check['GEOID'].astype(str)
        tract_alloc_check['scale_factor'] = np.where(tract_alloc_check['alloc_pop_sum'] > 1e-6, tract_alloc_check['tract_pop_target'] / tract_alloc_check['alloc_pop_sum'], 1.0)
        bldg_tract_join = bldg_tract_join.merge(tract_alloc_check[['GEOID', 'scale_factor']], on='GEOID', how='left')
        bldg_tract_join['allocated_population'] *= bldg_tract_join['scale_factor'].fillna(1.0); bldg_tract_join['allocated_population'] = bldg_tract_join['allocated_population'].fillna(0).clip(lower=0)
        avg_hh_size = bldg_tract_join['avg_household_size'].clip(lower=1.0)
        bldg_tract_join['estimated_households'] = (bldg_tract_join['allocated_population'] / avg_hh_size).round().clip(lower=1); bldg_tract_join['estimated_households'] = bldg_tract_join['estimated_households'].fillna(1).astype(int)
        cols_to_add = ['GEOID', 'total_population', 'avg_household_size', 'total_tract_res_volume_m3', 'volume_share', 'allocated_population', 'estimated_households']
        buildings_with_pop = buildings_with_heights.merge(bldg_tract_join[['unique_id'] + cols_to_add], on='unique_id', how='left')
        for col in cols_to_add:
            if col not in ['GEOID', 'total_population', 'avg_household_size']: buildings_with_pop[col] = buildings_with_pop[col].fillna(0)
        buildings_with_pop['allocated_population'] = buildings_with_pop['allocated_population'].fillna(0).clip(lower=0)
        buildings_with_pop['estimated_households'] = buildings_with_pop['estimated_households'].fillna(0).astype(int)
        print(f"Allocated population to {len(bldg_tract_join)} residential buildings.")
        print(f"Total population allocated: {buildings_with_pop['allocated_population'].sum():,.1f}")
        if 'total_population' in tracts_with_data_d.columns and not tracts_with_data_d.empty: print(f"Compare to Census sum in area: {tracts_with_data_d['total_population'].sum():,.0f}")
    else:
        print("No residential buildings found. Skipping population allocation.")
        buildings_with_pop = buildings_with_heights.copy()
        for col in ['GEOID', 'total_population', 'avg_household_size', 'total_tract_res_volume_m3', 'volume_share', 'allocated_population', 'estimated_households']: buildings_with_pop[col] = 0 if col not in ['GEOID'] else None

    print("Saving Module D outputs...")
    cols_to_save_d = [col for col in buildings_with_pop.columns if col not in ['knn_details', 'simulated_daily_deliveries', 'index_right', 'scale_factor']]; buildings_save_d = buildings_with_pop[cols_to_save_d].copy()
    for col in buildings_save_d.select_dtypes(include=['object']).columns:
        mask = buildings_save_d[col].apply(lambda x: isinstance(x, (list, dict)))
        if mask.any(): buildings_save_d.loc[mask, col] = buildings_save_d.loc[mask, col].astype(str)
    buildings_save_d.to_file(os.path.join(data_subdir, "buildings_with_population.geojson"), driver="GeoJSON")
    print("Generating population visualization...")
    fig_d, ax_d = plt.subplots(1, 2, figsize=(18, 8)); plot_data_d = buildings_with_pop.to_crs(epsg=3857); union_gdf_proj = union_gdf.to_crs(epsg=3857)
    plot_data_d_clipped = gpd.clip(plot_data_d, union_gdf_proj); res_plot_d = plot_data_d_clipped[plot_data_d_clipped['residential'] == 'yes']; nonres_plot_d = plot_data_d_clipped[plot_data_d_clipped['residential'] == 'no']
    nonres_plot_d.plot(ax=ax_d[0], color='lightgray', alpha=0.5, label='Non-Residential')
    if not res_plot_d.empty and res_plot_d['allocated_population'].sum() > 0:
         cmap_pop = 'Reds'; pop_max_quantile = res_plot_d['allocated_population'].quantile(0.98); norm_pop = Normalize(vmin=0, vmax=pop_max_quantile if pop_max_quantile > 0 else 1)
         res_plot_d.plot(column='allocated_population', cmap=cmap_pop, norm=norm_pop, legend=False, ax=ax_d[0], alpha=0.8)
         sm_pop = ScalarMappable(cmap=cmap_pop, norm=norm_pop); sm_pop.set_array([]); cbar_pop = fig_d.colorbar(sm_pop, ax=ax_d[0], shrink=0.7); cbar_pop.set_label("Allocated Population")
    else: ax_d[0].text(0.5, 0.5, "No residential pop", transform=ax_d[0].transAxes, ha='center')
    try: cx.add_basemap(ax_d[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    style_map(ax_d[0], 'Population Allocation (Union Area)')
    pop_for_hist = res_plot_d[res_plot_d['allocated_population'] > 0]['allocated_population']
    if not pop_for_hist.empty: sns.histplot(pop_for_hist.clip(upper=pop_for_hist.quantile(0.98)), bins=30, kde=False, ax=ax_d[1]); ax_d[1].set_title('Distribution of Allocated Population (>0)'); ax_d[1].set_xlabel('Est. Pop per Building'); ax_d[1].set_ylabel('Num Buildings')
    else: ax_d[1].text(0.5, 0.5, "No data", ha='center'); ax_d[1].set_title('Distribution of Allocated Population')
    plt.tight_layout(); plt.savefig(os.path.join(viz_subdir, 'population_visualization.png'), dpi=150, bbox_inches='tight'); plt.close(fig_d)
    stats_d = {}; stats_d_df = pd.DataFrame(columns=['Value'])
    if not res_plot_d.empty:
        stats_d = res_plot_d['allocated_population'].describe().to_dict(); stats_d['total_allocated_population'] = res_plot_d['allocated_population'].sum()
        stats_d.update(res_plot_d['estimated_households'].describe().rename(lambda x: f"households_{x}").to_dict()); stats_d_df = pd.DataFrame.from_dict(stats_d, orient='index', columns=['Value'])
    stats_d_df.to_csv(os.path.join(stats_subdir, 'population_stats.csv'))
    print("Module D Stats:"); print(stats_d_df)
else:
    print("Skipping Module D."); save_empty_gdf(os.path.join(data_subdir, "buildings_with_population.geojson")); save_empty_df(os.path.join(stats_subdir, 'population_stats.csv'), columns=['Value'])
module_D_time = time.time() - module_D_start
print(f"Module D completed in {module_D_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module E: Demand Modeling
# -----------------------------------------------------------------------------
# ... [Code as in v19.1] ...
print("\n## Module E: Demand Modeling...")
module_E_start = time.time()
buildings_with_demand = gpd.GeoDataFrame()
delivery_events_df = pd.DataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats"); frames_dir = os.path.join(output_dir, 'animation_frames'); config_subdir = os.path.join(output_dir, 'config')
try:
    buildings_with_pop = gpd.read_file(os.path.join(data_subdir, "buildings_with_population.geojson"))
    census_data_df_e = pd.read_csv(os.path.join(data_subdir, "census_data.csv"))
    union_gdf_e = gpd.read_file(os.path.join(data_subdir, "union_of_tracts.geojson")) # Load union
    if buildings_with_pop.empty or union_gdf_e.empty: raise ValueError("Input data missing.")
    if 'unique_id' not in buildings_with_pop.columns:
        if 'element_type' in buildings_with_pop.columns and 'osmid' in buildings_with_pop.columns: buildings_with_pop['unique_id'] = buildings_with_pop['element_type'] + buildings_with_pop['osmid'].astype(str)
        elif 'osmid' in buildings_with_pop.columns: buildings_with_pop['unique_id'] = 'way' + buildings_with_pop['osmid'].astype(str)
        else: buildings_with_pop = buildings_with_pop.reset_index().rename(columns={'index':'unique_id'}); buildings_with_pop['unique_id'] = 'bldg_' + buildings_with_pop['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for demand modeling: {e}. Skipping Module E.")

if not buildings_with_pop.empty and not union_gdf_e.empty:
    dem_config = config['demand_model']; ref = dem_config['reference_values']
    required_census_cols = ['GEOID', 'pop_density', 'employment_rate', 'median_income', 'avg_household_size']
    missing_census_cols = [col for col in required_census_cols if col not in buildings_with_pop.columns and col != 'GEOID']
    if missing_census_cols and not census_data_df_e.empty:
        print(f"Merging missing Census cols: {missing_census_cols}"); buildings_with_pop['GEOID'] = buildings_with_pop['GEOID'].astype(str); census_data_df_e['GEOID'] = census_data_df_e['GEOID'].astype(str)
        buildings_with_pop = buildings_with_pop.merge(census_data_df_e[[col for col in required_census_cols if col in census_data_df_e.columns]], on='GEOID', how='left', suffixes=('', '_census'))
    for col, ref_val in zip(['pop_density', 'employment_rate', 'median_income', 'avg_household_size'], [ref['ref_pop_density'], ref['ref_employment_rate'], ref['ref_median_income'], ref['ref_avg_household_size']]):
         if col not in buildings_with_pop.columns: buildings_with_pop[col] = ref_val
         else: buildings_with_pop[col] = buildings_with_pop[col].fillna(ref_val)
    buildings_with_pop['estimated_households'] = pd.to_numeric(buildings_with_pop['estimated_households'], errors='coerce').fillna(0).astype(int)

    def calculate_demand_rate(row, config): # Same formula
        params = config['demand_model']; ref = params['reference_values']; households = row.get('estimated_households', 0)
        if row.get('residential', 'no') != 'yes' or households <= 0: return 0, 1, 1, 1, 1
        income_scaled = row.get('median_income', ref['ref_median_income']) / ref['ref_median_income'] if ref['ref_median_income'] > 0 else 1.0
        density_scaled = row.get('pop_density', ref['ref_pop_density']) / ref['ref_pop_density'] if ref['ref_pop_density'] > 0 else 1.0
        avg_hh_size = row.get('avg_household_size', ref['ref_avg_household_size'])
        emp_rate = row.get('employment_rate', ref['ref_employment_rate'])
        inc_adj = max(0.2, 1 + params['income_coef'] * (income_scaled - 1))
        den_adj = max(0.2, 1 + params['pop_density_coef'] * (density_scaled - 1))
        hh_adj = max(0.2, 1 + params['household_size_coef'] * (avg_hh_size - ref['ref_avg_household_size']) / ref['ref_avg_household_size']) if ref['ref_avg_household_size'] > 0 else 1.0
        emp_adj = max(0.2, 1 + params['employment_coef'] * (emp_rate - ref['ref_employment_rate']) / ref['ref_employment_rate']) if ref['ref_employment_rate'] > 0 else 1.0
        base_demand = params['base_deliveries_per_household_per_day'] * households
        demand_rate = base_demand * inc_adj * den_adj * hh_adj * emp_adj; return max(0, demand_rate), inc_adj, den_adj, hh_adj, emp_adj
    print("Calculating base demand rates..."); demand_results = buildings_with_pop.apply(lambda row: calculate_demand_rate(row, config), axis=1)
    buildings_with_pop[['demand_rate', 'income_adj', 'density_adj', 'hh_size_adj', 'emp_adj']] = pd.DataFrame(demand_results.tolist(), index=buildings_with_pop.index)

    print("Simulating delivery events..."); sim_start = datetime.strptime(dem_config['simulation_start_date'], '%Y-%m-%d'); sim_end = sim_start + timedelta(days=dem_config['simulation_duration_days'])
    delivery_events = []; building_daily_counts = {uid: [] for uid in buildings_with_pop['unique_id']}
    union_geom_unary = unary_union(union_gdf_e.geometry);
    buildings_with_pop = buildings_with_pop.to_crs(union_gdf_e.crs)
    buildings_in_union_mask = buildings_with_pop.geometry.within(union_geom_unary)
    sim_buildings = buildings_with_pop[buildings_in_union_mask & (buildings_with_pop['demand_rate'] > 0) & (buildings_with_pop['residential'] == 'yes')]
    if not sim_buildings.empty:
        hourly_dist_config = dem_config['hourly_distribution']; hourly_dist_values = list(hourly_dist_config.values()); hourly_dist_keys = list(hourly_dist_config.keys()); hourly_probs = np.array(hourly_dist_values, dtype=float); probs_sum = hourly_probs.sum()
        sim_buildings_indexed = sim_buildings.set_index('unique_id', drop=False)
        for date in pd.date_range(sim_start, sim_end - timedelta(days=1)):
            day_of_week = date.weekday(); sim_month = date.month; day_factor = dem_config['daily_variation'].get(day_of_week, 1.0); month_factor = dem_config['monthly_factors'].get(sim_month, 1.0)
            daily_adjusted_rates = sim_buildings_indexed['demand_rate'] * day_factor * month_factor
            num_deliveries_today = np.random.poisson(daily_adjusted_rates.clip(lower=0))
            for unique_id, count in zip(daily_adjusted_rates.index, num_deliveries_today): building_daily_counts.setdefault(unique_id, []).append(count)
            total_deliveries_today = num_deliveries_today.sum()
            if total_deliveries_today > 0:
                building_unique_ids_today = np.repeat(daily_adjusted_rates.index, num_deliveries_today)
                if not hourly_dist_values or probs_sum <= 1e-9: assigned_hours = np.random.choice(24, size=total_deliveries_today)
                else: normalized_probs = hourly_probs / probs_sum; assigned_hours = np.random.choice(hourly_dist_keys, size=total_deliveries_today, p=normalized_probs)
                skipped_events = 0
                for i in range(total_deliveries_today):
                    bldg_unique_id = building_unique_ids_today[i]
                    if bldg_unique_id not in sim_buildings_indexed.index: skipped_events += 1; continue
                    building_row = sim_buildings_indexed.loc[bldg_unique_id]
                    if building_row.geometry is None or building_row.geometry.is_empty: skipped_events += 1; continue
                    point = building_row.geometry.centroid;
                    if point.is_empty: skipped_events += 1; continue
                    hour = assigned_hours[i]; minute = random.randint(0, 59); second = random.randint(0, 59)
                    timestamp = date.replace(hour=int(hour), minute=minute, second=second)
                    delivery_events.append({'building_unique_id': building_row['unique_id'], 'timestamp': timestamp, 'latitude': point.y, 'longitude': point.x,})
                if skipped_events > 0: print(f"Skipped {skipped_events} events on {date.date()}")
        delivery_events_df = pd.DataFrame(delivery_events)
        print(f"Simulated {len(delivery_events_df)} valid delivery events over {dem_config['simulation_duration_days']} days.")
    else: print("No residential buildings with positive demand within union."); delivery_events_df = pd.DataFrame(columns=['building_unique_id', 'timestamp', 'latitude', 'longitude'])
    buildings_with_pop['simulated_daily_deliveries'] = buildings_with_pop['unique_id'].map(lambda uid: str(building_daily_counts.get(uid, [0]*dem_config['simulation_duration_days'])))
    buildings_with_pop['simulated_total_deliveries'] = buildings_with_pop['simulated_daily_deliveries'].apply(lambda x: sum(eval(x)) if isinstance(x, str) and x.startswith('[') else 0)
    buildings_with_demand = buildings_with_pop.reset_index(drop=True)

    # --- Generate Demand Animation Frames (with Legend/Colorbar) ---
    print("\nGenerating Hourly Demand Animation Frames...")
    if not delivery_events_df.empty:
        os.makedirs(frames_dir, exist_ok=True);
        # Calculate max deliveries per hour for consistent scale
        temp_df = delivery_events_df.copy()
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        temp_df['date'] = temp_df['timestamp'].dt.date
        hourly_building_counts = temp_df.groupby(['date', 'hour', 'building_unique_id']).size()
        max_deliveries_per_hour = max(1, hourly_building_counts.max() if not hourly_building_counts.empty else 1)
        animation_norm = Normalize(vmin=0, vmax=max_deliveries_per_hour); # Use calculated max
        animation_cmap = plt.cm.YlOrRd
        sm_anim = ScalarMappable(cmap=animation_cmap, norm=animation_norm); sm_anim.set_array([]) # Create ScalarMappable for colorbar

        buildings_for_anim = buildings_with_demand.to_crs(epsg=3857); tracts_for_anim = union_gdf_e.to_crs(epsg=3857)
        frame_count = 0
        union_geom_anim = unary_union(tracts_for_anim.geometry)

        for date in pd.date_range(sim_start, sim_end - timedelta(days=1)):
             sample_day_str = date.strftime('%Y-%m-%d'); print(f"  Generating frames for {sample_day_str}...")
             for hour in range(6, 22):
                hour_start = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour); hour_end = hour_start + timedelta(hours=1)
                hour_deliveries = delivery_events_df[ (delivery_events_df['timestamp'] >= hour_start) & (delivery_events_df['timestamp'] < hour_end) ]
                hour_counts = hour_deliveries.groupby('building_unique_id').size().to_dict()
                buildings_for_anim['deliveries_this_hour'] = buildings_for_anim['unique_id'].map(lambda uid: hour_counts.get(uid, 0))
                buildings_for_anim['active'] = buildings_for_anim['deliveries_this_hour'] > 0
                fig_anim, ax_anim = plt.subplots(figsize=(12, 12))
                tracts_for_anim.boundary.plot(ax=ax_anim, color='black', linewidth=1.0, alpha=0.6, zorder=5)
                inactive = buildings_for_anim[(~buildings_for_anim['active']) & buildings_for_anim.geometry.intersects(union_geom_anim)]
                active = buildings_for_anim[buildings_for_anim['active'] & buildings_for_anim.geometry.intersects(union_geom_anim)]
                if not inactive.empty: inactive.plot(ax=ax_anim, color='lightgray', edgecolor='darkgray', linewidth=0.1, alpha=0.5, zorder=1)
                if not active.empty: active.plot(column='deliveries_this_hour', ax=ax_anim, cmap=animation_cmap, norm=animation_norm, edgecolor='black', linewidth=0.3, alpha=0.9, zorder=2)
                try: cx.add_basemap(ax_anim, crs=buildings_for_anim.crs, source=cx.providers.CartoDB.Positron, alpha=0.7, zorder=0)
                except Exception as e: print(f"Basemap failed: {e}")
                time_str = f"{hour:02d}:00 - {hour+1:02d}:00"; title_str = f"UAS Deliveries - {date.strftime('%a, %b %d')} | {time_str}"
                # Add Colorbar
                cbar_anim = fig_anim.colorbar(sm_anim, ax=ax_anim, shrink=0.6, location='left', pad=0.01)
                cbar_anim.set_label("# Deliveries This Hour")
                tick_max = int(animation_norm.vmax)
                cbar_anim.set_ticks(np.linspace(0, tick_max, num=min(tick_max+1, 5), dtype=int))
                # Style map and add time title
                style_map(ax_anim, "") # Apply style first
                ax_anim.set_title(title_str, fontsize=16, fontweight='bold') # Then title
                ax_anim.text(0.98, 0.02, f"Deliveries: {len(hour_deliveries)}", transform=ax_anim.transAxes, fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.7))
                frame_path = os.path.join(frames_dir, f"demand_day{(date-sim_start).days}_hour_{hour:02d}.png"); plt.savefig(frame_path, dpi=100, bbox_inches='tight'); plt.close(fig_anim); frame_count += 1
        print(f"Generated {frame_count} demand animation frames.")
    else: print("Skipping animation frame generation.")

    print("Saving Module E outputs...")
    cols_to_save_e = [col for col in buildings_with_demand.columns if col not in ['knn_details', 'index_right', 'scale_factor', 'deliveries_this_hour', 'active']]
    buildings_save_e = buildings_with_demand[cols_to_save_e].copy()
    for col in buildings_save_e.select_dtypes(include=['object']).columns:
        mask = buildings_save_e[col].apply(lambda x: isinstance(x, (list, dict)));
        if mask.any(): buildings_save_e.loc[mask, col] = buildings_save_e.loc[mask, col].astype(str)
    buildings_save_e.to_file(os.path.join(data_subdir, "buildings_with_demand.geojson"), driver="GeoJSON")
    if not delivery_events_df.empty: delivery_events_df.to_csv(os.path.join(data_subdir, "delivery_events.csv"), index=False)
    else: save_empty_df(os.path.join(data_subdir, "delivery_events.csv"), columns=['building_unique_id', 'timestamp', 'latitude', 'longitude'])
    with open(os.path.join(config_subdir, "demand_parameters.yaml"), 'w') as f: yaml.dump(config['demand_model'], f)
    print("Generating demand visualization...")
    fig_e, ax_e = plt.subplots(1, 2, figsize=(18, 8)); plot_data_e = buildings_with_demand.to_crs(epsg=3857); union_gdf_proj_e = union_gdf_e.to_crs(epsg=3857)
    plot_data_e = gpd.clip(plot_data_e, union_gdf_proj_e); res_plot_e = plot_data_e[plot_data_e['residential'] == 'yes']; nonres_plot_e = plot_data_e[plot_data_e['residential'] == 'no']
    nonres_plot_e.plot(ax=ax_e[0], color='lightgray', alpha=0.5, label='Non-Residential')
    if not res_plot_e.empty and res_plot_e['simulated_total_deliveries'].sum() > 0:
         cmap_dem = 'YlOrRd'; vmax_dem = res_plot_e['simulated_total_deliveries'].quantile(0.98); norm_dem = Normalize(vmin=0, vmax=vmax_dem if vmax_dem > 0 else 1)
         res_plot_e.plot(column='simulated_total_deliveries', cmap=cmap_dem, norm=norm_dem, legend=False, ax=ax_e[0], alpha=0.8)
         sm_dem = ScalarMappable(cmap=cmap_dem, norm=norm_dem); sm_dem.set_array([]); cbar_dem = fig_e.colorbar(sm_dem, ax=ax_e[0], shrink=0.7); cbar_dem.set_label(f"Simulated Deliveries ({dem_config['simulation_duration_days']} days)")
    else: ax_e[0].text(0.5, 0.5, "No res demand", transform=ax_e[0].transAxes, ha='center')
    try: cx.add_basemap(ax_e[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    style_map(ax_e[0], 'Simulated Delivery Demand (Union Area)')
    if not delivery_events_df.empty:
        delivery_events_df['date'] = delivery_events_df['timestamp'].dt.date; daily_summary = delivery_events_df.groupby('date').size()
        if not daily_summary.empty: daily_summary.plot(kind='line', marker='o', ax=ax_e[1]); ax_e[1].set_title('Total Simulated Deliveries per Day'); ax_e[1].set_xlabel('Date'); ax_e[1].set_ylabel('Deliveries'); ax_e[1].grid(True); plt.setp(ax_e[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        else: ax_e[1].text(0.5, 0.5, 'No deliveries.', ha='center')
    else: ax_e[1].text(0.5, 0.5, 'No deliveries.', ha='center')
    ax_e[1].set_title('Total Simulated Deliveries per Day')
    plt.tight_layout(); plt.savefig(os.path.join(viz_subdir, 'demand_visualization.png'), dpi=150, bbox_inches='tight'); plt.close(fig_e)
    stats_e = {}; stats_e_df = pd.DataFrame(columns=['Value'])
    if not res_plot_e.empty: stats_e = res_plot_e['demand_rate'].describe().rename(lambda x: f"demand_rate_{x}").to_dict()
    stats_e['total_simulated_deliveries'] = len(delivery_events_df)
    stats_e['avg_daily_simulated'] = len(delivery_events_df) / dem_config['simulation_duration_days'] if dem_config['simulation_duration_days'] > 0 else 0
    if stats_e: stats_e_df = pd.DataFrame.from_dict(stats_e, orient='index', columns=['Value'])
    stats_e_df.to_csv(os.path.join(stats_subdir, 'demand_stats.csv'))
    print("Module E Stats:"); print(stats_e_df)
else:
    print("Skipping Module E."); save_empty_gdf(os.path.join(data_subdir, "buildings_with_demand.geojson")); save_empty_df(os.path.join(data_subdir, "delivery_events.csv")); save_empty_df(os.path.join(stats_subdir, 'demand_stats.csv')); os.makedirs(config_subdir, exist_ok=True); open(os.path.join(config_subdir, "demand_parameters.yaml"), 'w').close()
module_E_time = time.time() - module_E_start
print(f"Module E completed in {module_E_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module F: Origin-Destination Matching & Dataset Generation
# -----------------------------------------------------------------------------
# ... [Code identical to v19.1 + O-D Animation Frame Generation] ...
print("\n## Module F: Origin-Destination Matching & Dataset Generation...")
module_F_start = time.time()
matched_gdf_all = gpd.GeoDataFrame() # Store ALL matched deliveries for animation
final_csv = pd.DataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats"); frames_dir = os.path.join(output_dir, 'animation_frames')
try:
    stores_gdf_f = gpd.read_file(os.path.join(data_subdir, "stores.geojson")) # Point geometry, has 'geometry_poly' col
    delivery_events_f = pd.read_csv(os.path.join(data_subdir, "delivery_events.csv"))
    buildings_gdf_f = gpd.read_file(os.path.join(data_subdir, "buildings_with_demand.geojson")) # Contains unique_id and footprints
    union_gdf_f = gpd.read_file(os.path.join(data_subdir, "union_of_tracts.geojson")) # Load union for viz
    all_buildings_classified_f = gpd.read_file(os.path.join(data_subdir, "classified_buildings.geojson")) # Load classified buildings for store footprints

    if stores_gdf_f.empty or delivery_events_f.empty or buildings_gdf_f.empty: raise ValueError("Input data empty.")
    if 'unique_id' not in buildings_gdf_f.columns: raise ValueError("Buildings file missing unique_id.")
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for O-D matching: {e}. Skipping Module F.")

if not stores_gdf_f.empty and not delivery_events_f.empty and not buildings_gdf_f.empty:
    delivery_events_f['timestamp'] = pd.to_datetime(delivery_events_f['timestamp'])
    match_config = config['origin_destination_matching']; match_hour = match_config['simulation_hour_for_matching']
    sim_start_date_f = datetime.strptime(config['demand_model']['simulation_start_date'], '%Y-%m-%d').date()
    sim_end_date_f = sim_start_date_f + timedelta(days=config['demand_model']['simulation_duration_days']) # End date for week

    print(f"Processing {len(delivery_events_f)} total deliveries for O-D matching and animation...")
    od_config = config['origin_destination_matching']; match_method = od_config['method']
    stores_utm = stores_gdf_f.to_crs(utm_crs)
    if 'store_id' not in stores_utm.columns: stores_utm = stores_utm.reset_index().rename(columns={'index':'store_id'}); stores_utm['store_id'] = 'Store_' + stores_utm['store_id'].astype(str)
    stores_utm['store_type'] = stores_utm['store_type'].fillna('Other')
    all_deliveries_gdf = gpd.GeoDataFrame(delivery_events_f, geometry=gpd.points_from_xy(delivery_events_f['longitude'], delivery_events_f['latitude']), crs=WGS84).to_crs(utm_crs)
    matched_gdf_all = None # Initialize for ALL deliveries
    if all_deliveries_gdf.empty: print("No deliveries in the simulation period to match.")
    # --- O-D Matching Logic (applied to ALL deliveries) ---
    elif match_method == 'Proximity-Based':
        # ... [Proximity logic as in v19.1] ...
        print("Using Proximity-Based matching for all deliveries...")
        all_deliveries_gdf = all_deliveries_gdf.reset_index(drop=True); stores_utm_valid = stores_utm[stores_utm.geometry.is_valid & ~stores_utm.geometry.is_empty].reset_index(drop=True)
        if not stores_utm_valid.empty:
            matched_gdf_all = gpd.sjoin_nearest(all_deliveries_gdf, stores_utm_valid, how='left', distance_col="dist_to_store")
            matched_gdf_all = matched_gdf_all.rename(columns={'store_id_right':'origin_id', 'name_right':'origin_name', 'store_type_right':'store_type', 'geometry_right': 'origin_geometry'})
            if 'index_right' in matched_gdf_all.columns: matched_gdf_all = matched_gdf_all.drop(columns=['index_right'])
        else: print("ERROR: No valid stores for Proximity-Based matching.")

    elif match_method == 'Market Share Weighted':
        # ... [Market Share logic as in v19.1] ...
        print("Using Market Share Weighted matching for all deliveries...")
        shares = od_config.get('market_shares', {}); stores_utm_typed = stores_utm[stores_utm['store_type'].notna()]
        available_types = stores_utm_typed['store_type'].unique(); valid_shares = {st: sh for st, sh in shares.items() if st in available_types and pd.notna(st) and isinstance(sh, (int,float)) and sh > 0}
        if not valid_shares: print("ERROR: No market shares match stores. Falling back to random."); match_method = 'Random'
        else:
            total_share = sum(valid_shares.values());
            if not math.isclose(total_share, 1.0, rel_tol=1e-5): print(f"Warn: Market shares sum {total_share:.4f}. Normalizing."); norm_shares = {st: sh / total_share for st, sh in valid_shares.items()}
            else: norm_shares = valid_shares
            store_types_for_choice = list(norm_shares.keys()); probabilities = list(norm_shares.values())
            all_deliveries_gdf['assigned_store_type'] = np.random.choice(store_types_for_choice, size=len(all_deliveries_gdf), p=probabilities)
            matched_list = []; stores_by_type = {stype: sdf[sdf.geometry.is_valid & ~sdf.geometry.is_empty] for stype, sdf in stores_utm_typed.groupby('store_type')}
            valid_stores_overall = stores_utm_typed[stores_utm_typed.geometry.is_valid & ~stores_utm_typed.geometry.is_empty]
            for idx, delivery in all_deliveries_gdf.iterrows():
                assigned_type = delivery['assigned_store_type']; chosen_origin = None
                if not (delivery.geometry and delivery.geometry.is_valid and not delivery.geometry.is_empty): print(f"Warn: Invalid delivery geometry {idx}. Fallback.")
                elif pd.isna(assigned_type) or assigned_type not in stores_by_type or stores_by_type[assigned_type].empty: print(f"Warn: Invalid type '{assigned_type}' or no stores for it {idx}. Fallback.")
                else:
                    possible_origins = stores_by_type[assigned_type]; distances = possible_origins.geometry.distance(delivery.geometry)
                    if distances.empty or distances.isna().all(): print(f"Warn: Could not calculate distances {idx}. Fallback.")
                    else:
                        min_dist_idx_label = distances.idxmin()
                        if min_dist_idx_label in possible_origins.index: chosen_origin = possible_origins.loc[min_dist_idx_label] # Use .loc
                        else: print(f"Warn: idxmin label {min_dist_idx_label} not in index {idx}. Fallback.")
                if chosen_origin is None and not valid_stores_overall.empty:
                     if delivery.geometry and delivery.geometry.is_valid:
                          fallback_distances = valid_stores_overall.geometry.distance(delivery.geometry)
                          if not fallback_distances.empty and not fallback_distances.isna().all():
                               min_dist_idx_label_fb = fallback_distances.idxmin()
                               if min_dist_idx_label_fb in valid_stores_overall.index: chosen_origin = valid_stores_overall.loc[min_dist_idx_label_fb] # Use .loc
                     if chosen_origin is None: chosen_origin = valid_stores_overall.iloc[0]
                if chosen_origin is None: print(f"ERROR: Could not assign origin for {idx}. Skipping."); continue
                delivery_dict = delivery.to_dict(); delivery_dict['origin_id'] = chosen_origin['store_id']; delivery_dict['origin_name'] = chosen_origin['name']; delivery_dict['store_type'] = chosen_origin['store_type']; delivery_dict['origin_geometry'] = chosen_origin['geometry']
                matched_list.append(delivery_dict)
            if matched_list: matched_gdf_all = gpd.GeoDataFrame(matched_list, crs=utm_crs)
            else: print("No deliveries matched."); matched_gdf_all = gpd.GeoDataFrame()

    if match_method == 'Random' or matched_gdf_all is None:
        if not all_deliveries_gdf.empty:
             print("Using Random matching for all deliveries..."); num_deliveries = len(all_deliveries_gdf); random_store_indices = np.random.choice(stores_utm.index, size=num_deliveries)
             chosen_origins = stores_utm.loc[random_store_indices].reset_index(drop=True)
             matched_gdf_all = all_deliveries_gdf.reset_index(drop=True).join(chosen_origins.add_suffix('_origin').rename(columns={'store_id_origin':'origin_id', 'name_origin':'origin_name', 'geometry_origin':'origin_geometry', 'store_type_origin':'store_type'}))
        else: matched_gdf_all = gpd.GeoDataFrame()

    # --- Format FINAL routing_dataset.csv (from the SPECIFIED HOUR) ---
    hour_start = datetime.combine(sim_start_date_f, datetime.min.time()) + timedelta(hours=match_hour); hour_end = hour_start + timedelta(hours=1)
    matched_gdf_hour = matched_gdf_all[ # Filter the full matched set
        (matched_gdf_all['timestamp'] >= hour_start) &
        (matched_gdf_all['timestamp'] < hour_end)
    ].copy() if matched_gdf_all is not None else gpd.GeoDataFrame()

        # Filter ALL deliveries (before matching) to find destinations for THIS hour
    deliveries_this_hour_for_viz = all_deliveries_gdf[
        (all_deliveries_gdf['timestamp'] >= hour_start) &
        (all_deliveries_gdf['timestamp'] < hour_end)
    ].copy() if not all_deliveries_gdf.empty else gpd.GeoDataFrame()

    if matched_gdf_hour is not None and not matched_gdf_hour.empty:
        print(f"Formatting {len(matched_gdf_hour)} deliveries for final dataset (Hour {match_hour}:00)...")
        if 'origin_geometry' not in matched_gdf_hour.columns: matched_gdf_hour = matched_gdf_hour.merge(stores_utm[['store_id', 'geometry']].rename(columns={'store_id':'origin_id', 'geometry':'origin_geometry'}), on='origin_id', how='left')
        matched_gdf_hour['straight_line_distance_m'] = matched_gdf_hour.apply(lambda row: row.geometry.distance(row['origin_geometry']) if isinstance(row.geometry, Point) and isinstance(row.get('origin_geometry'), Point) and not row.geometry.is_empty and not row.get('origin_geometry').is_empty and row.geometry.is_valid and row.get('origin_geometry').is_valid else np.nan, axis=1 )
        final_output = matched_gdf_hour.copy(); final_output['timestamp_unix'] = final_output['timestamp'].apply(lambda dt: int(dt.timestamp()))
        final_output_wgs84 = final_output.to_crs(WGS84)
        def format_coords(point_geom):
            if pd.notna(point_geom) and isinstance(point_geom, Point) and not point_geom.is_empty: return f"[{point_geom.y:.6f}, {point_geom.x:.6f}]"
            return None
        if 'origin_geometry' in final_output_wgs84.columns: origin_geom_series = gpd.GeoSeries(final_output_wgs84['origin_geometry'], crs=final_output_wgs84.crs).centroid; final_output['origin_coordinates'] = origin_geom_series.apply(format_coords)
        else: final_output['origin_coordinates'] = None
        destination_geom_series = gpd.GeoSeries(final_output_wgs84['geometry'], crs=final_output_wgs84.crs); final_output['destination_coordinates'] = destination_geom_series.apply(format_coords)
        final_output.insert(0, 'order_id', range(1, len(final_output) + 1))
        dest_id_col = 'building_unique_id' if 'building_unique_id' in final_output.columns else 'unique_id'
        if dest_id_col not in final_output.columns: print(f"Warn: Dest ID Col '{dest_id_col}' not found."); final_output = final_output.reset_index().rename(columns={'index':'destination_id_temp'}); dest_id_col = 'destination_id_temp'
        origin_id_col = 'origin_id' if 'origin_id' in final_output.columns else 'store_id'
        if origin_id_col not in final_output.columns: origin_id_col = stores_utm.columns[0]
        final_csv = final_output[['order_id', 'timestamp_unix', origin_id_col, 'origin_coordinates', dest_id_col, 'destination_coordinates']].rename(columns={'timestamp_unix': 'timestamp', dest_id_col: 'destination_id', origin_id_col: 'origin_id'})
        print("Saving final routing_dataset.csv...")
        final_csv.to_csv(os.path.join(data_subdir, "routing_dataset.csv"), index=False)
    else:
        print(f"No deliveries matched for hour {match_hour}. Saving empty routing dataset.")
        save_empty_df(os.path.join(data_subdir, "routing_dataset.csv"), columns=['order_id', 'timestamp', 'origin_id', 'origin_coordinates', 'destination_id', 'destination_coordinates'])

    # --- Save DETAILED GeoJSON (for ALL matched deliveries over the week) ---
    if matched_gdf_all is not None and not matched_gdf_all.empty:
        print("Saving detailed weekly O-D data (GeoJSON)...")
        if 'origin_geometry' not in matched_gdf_all.columns: matched_gdf_all = matched_gdf_all.merge(stores_utm[['store_id', 'geometry']].rename(columns={'store_id':'origin_id', 'geometry':'origin_geometry'}), on='origin_id', how='left')
        cols_to_save_f = [col for col in matched_gdf_all.columns if col != 'origin_geometry' and col not in ['assigned_store_type', 'index_right', 'dist_to_store']]
        matched_gdf_save = matched_gdf_all[cols_to_save_f].copy()
        matched_gdf_save['geometry'] = matched_gdf_all.apply(lambda row: LineString([row['origin_geometry'], row.geometry]) if row.get('origin_geometry') and row.geometry and row.get('origin_geometry').is_valid and row.geometry.is_valid else None, axis=1);
        matched_gdf_save = gpd.GeoDataFrame(matched_gdf_save, geometry='geometry', crs=utm_crs)
        matched_gdf_save['timestamp'] = matched_gdf_save['timestamp'].astype(str) # Convert timestamp for saving
        for col in matched_gdf_save.select_dtypes(include=['object']).columns:
            mask = matched_gdf_save[col].apply(lambda x: isinstance(x, (list, dict)));
            if mask.any(): matched_gdf_save.loc[mask, col] = matched_gdf_save.loc[mask, col].astype(str)
        matched_gdf_save.to_file(os.path.join(data_subdir, "routing_dataset_detailed_weekly.geojson"), driver="GeoJSON")
    else:
        save_empty_gdf(os.path.join(data_subdir, "routing_dataset_detailed_weekly.geojson"))

    # --- Generate Static O-D Visualization (using footprints for specified hour) ---
    print("Generating O-D visualization (using footprints for specified hour)...")
    fig_f, ax_f = plt.subplots(figsize=(12, 12))
    buildings_plot_f = buildings_gdf_f.to_crs(epsg=3857)
    union_plot_f = union_gdf_f.to_crs(epsg=3857)
    all_buildings_classified_f = gpd.read_file(os.path.join(data_subdir, "classified_buildings.geojson")).to_crs(epsg=3857)
    stores_footprints_f = all_buildings_classified_f[all_buildings_classified_f['is_store']]
    # Load the detailed routing data for the hour
    routes_plot_f = gpd.GeoDataFrame() # Init empty
    if matched_gdf_hour is not None and not matched_gdf_hour.empty:
         # Recreate line geometry if needed (might have been dropped)
         if 'line_geometry' not in matched_gdf_hour.columns:
             if 'origin_geometry' not in matched_gdf_hour.columns: matched_gdf_hour = matched_gdf_hour.merge(stores_utm[['store_id', 'geometry']].rename(columns={'store_id':'origin_id', 'geometry':'origin_geometry'}), on='origin_id', how='left')
             matched_gdf_hour['line_geometry'] = matched_gdf_hour.apply(lambda row: LineString([row['origin_geometry'], row.geometry]) if row.get('origin_geometry') and row.geometry and row.get('origin_geometry').is_valid and row.geometry.is_valid else None, axis=1)
         routes_plot_f = gpd.GeoDataFrame(matched_gdf_hour, geometry='line_geometry', crs=utm_crs).to_crs(epsg=3857)
    # Get destination counts for *this hour* from the filtered raw deliveries
    dest_counts_hour = deliveries_this_hour_for_viz['building_unique_id'].value_counts().to_dict() if not deliveries_this_hour_for_viz.empty else {}
    buildings_plot_f['deliveries_this_hour'] = buildings_plot_f['unique_id'].map(dest_counts_hour).fillna(0)

    union_plot_f.boundary.plot(ax=ax_f, color='black', linewidth=2.0, label='Union of Tracts', zorder=5)
    store_colors_map = {'Walmart': 'blue', 'Target': 'red', 'Retail Hub': 'orange', 'H-E-B': 'purple', 'Warehouse/Fulfillment': 'brown', 'Other': 'cyan', 'Retail Hub / Warehouse': 'darkorange'}
    plotted_store_types = set(); store_handles = []
    origin_ids_this_hour = routes_plot_f['origin_id'].unique() if not routes_plot_f.empty else []
    stores_to_plot = stores_footprints_f[stores_footprints_f['store_id'].isin(origin_ids_this_hour)]
    if not stores_to_plot.empty:
        print(f"Plotting {len(stores_to_plot)} active store footprints...")
        for store_type, group in stores_to_plot.groupby('store_type'):
            color = store_colors_map.get(store_type, 'gray'); label = store_type if store_type not in plotted_store_types else None
            group.plot(ax=ax_f, color=color, alpha=0.6, edgecolor='black', linewidth=0.7, label=label, zorder=3)
            if label: plotted_store_types.add(store_type); store_handles.append(Patch(facecolor=color, edgecolor='black', label=label, alpha=0.6))
    dest_buildings_plot = buildings_plot_f[(buildings_plot_f['deliveries_this_hour'] > 0) & (buildings_plot_f['residential'] == 'yes')]
    sm_od = None
    if not dest_buildings_plot.empty:
         print(f"Plotting {len(dest_buildings_plot)} destination footprints for hour {match_hour}...")
         cmap_od = plt.cm.Greens; max_deliveries = max(1, dest_buildings_plot['deliveries_this_hour'].max())
         norm_od = Normalize(vmin=1, vmax=max_deliveries)
         valid_dest_polys = dest_buildings_plot[dest_buildings_plot.geometry.is_valid & ~dest_buildings_plot.geometry.is_empty]
         if not valid_dest_polys.empty:
             valid_dest_polys.plot(column='deliveries_this_hour', cmap=cmap_od, norm=norm_od, ax=ax_f, alpha=0.8, edgecolor='darkgreen', linewidth=0.5, zorder=2)
             sm_od = ScalarMappable(cmap=cmap_od, norm=norm_od); sm_od.set_array([])
    else: print(f"No destination buildings with deliveries this hour ({match_hour}).")
    num_to_plot = min(500, len(routes_plot_f)); sample_routes = routes_plot_f.sample(num_to_plot, random_state=seed) if num_to_plot < len(routes_plot_f) else routes_plot_f
    valid_lines = sample_routes[sample_routes.geometry.is_valid & ~sample_routes.geometry.is_empty]
    if not valid_lines.empty: print(f"Plotting {len(valid_lines)} O-D lines (sampled)..."); valid_lines.plot(ax=ax_f, color='black', linewidth=0.5, alpha=0.1, zorder=1)
    else: print("No valid O-D lines to plot for this hour.")
    try: cx.add_basemap(ax_f, crs='epsg:3857', source=cx.providers.CartoDB.Positron, alpha=0.5, zorder=0)
    except Exception as e: print(f"Basemap failed: {e}")
    style_map(ax_f, f'Origin-Destination Matches (Hour {match_hour}:00)')
    if sm_od:
        cbar_od = fig_f.colorbar(sm_od, ax=ax_f, shrink=0.6, location='left', pad=0.01); cbar_od.set_label("# Deliveries to Destination")
        if max_deliveries <= 10: cbar_od.set_ticks(np.arange(1, int(max_deliveries) + 1)) # Use arange for integer ticks
        else: cbar_od.set_ticks(np.linspace(1, max_deliveries, num=5, dtype=int))
    if store_handles: ax_f.legend(handles=store_handles, title="Origin Store Types", loc='upper right', fontsize=9)
    plt.savefig(os.path.join(viz_subdir, 'od_map.png'), dpi=150, bbox_inches='tight'); plt.close(fig_f)

    # --- Generate O-D Animation Frames ---
    print("\nGenerating O-D Animation Frames...")
    if matched_gdf_all is not None and not matched_gdf_all.empty:
        buildings_anim_od = buildings_gdf_f.to_crs(epsg=3857)
        stores_anim_od = all_buildings_classified_f[all_buildings_classified_f['is_store']].to_crs(epsg=3857)
        union_anim_od = union_gdf_f.to_crs(epsg=3857)
        # Use matched_gdf_all which has POINT geometry for origin/dest
        all_routes_for_anim = matched_gdf_all.to_crs(epsg=3857).copy()
        # Recreate LineString geometry for ALL routes for animation plotting
        if 'origin_geometry' not in all_routes_for_anim.columns:
             all_routes_for_anim = all_routes_for_anim.merge(stores_utm[['store_id', 'geometry']].rename(columns={'store_id':'origin_id', 'geometry':'origin_geometry'}), on='origin_id', how='left')
        all_routes_for_anim['line_geometry'] = all_routes_for_anim.apply(lambda row: LineString([row['origin_geometry'], row.geometry]) if row.get('origin_geometry') and row.geometry and row.get('origin_geometry').is_valid and row.geometry.is_valid else None, axis=1)
        all_routes_for_anim = gpd.GeoDataFrame(all_routes_for_anim, geometry='line_geometry', crs=utm_crs).to_crs(epsg=3857)
        all_routes_for_anim['timestamp_dt'] = pd.to_datetime(all_routes_for_anim['timestamp'])

        frame_count_od = 0
        sim_start_anim = datetime.strptime(config['demand_model']['simulation_start_date'], '%Y-%m-%d')
        sim_end_anim = sim_start_anim + timedelta(days=config['demand_model']['simulation_duration_days'])
        os.makedirs(frames_dir, exist_ok=True)
        cmap_od_anim = plt.cm.Greens
        hourly_dest_counts_all = all_routes_anim.groupby([all_routes_anim['timestamp_dt'].dt.date, all_routes_anim['timestamp_dt'].dt.hour, 'building_unique_id']).size()
        max_hourly_deliveries_overall = max(1, hourly_dest_counts_all.max() if not hourly_dest_counts_all.empty else 1)
        norm_od_anim = Normalize(vmin=1, vmax=max_hourly_deliveries_overall)
        sm_od_anim = ScalarMappable(cmap=cmap_od_anim, norm=norm_od_anim); sm_od_anim.set_array([])

        for date in pd.date_range(sim_start_anim, sim_end_anim - timedelta(days=1)):
             sample_day_str = date.strftime('%Y-%m-%d'); print(f"  Generating O-D frames for {sample_day_str}...")
             for hour in range(6, 22):
                hour_start_anim = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour); hour_end_anim = hour_start_anim + timedelta(hours=1)
                hour_routes_anim = all_routes_anim[(all_routes_anim['timestamp_dt'] >= hour_start_anim) & (all_routes_anim['timestamp_dt'] < hour_end_anim)].copy()
                fig_anim_od, ax_anim_od = plt.subplots(figsize=(12, 12))
                union_anim_od.boundary.plot(ax=ax_anim_od, color='black', linewidth=1.0, alpha=0.6, zorder=5)
                dest_counts_anim = hour_routes_anim['building_unique_id'].value_counts().to_dict()
                buildings_anim_od['deliveries_this_hour'] = buildings_anim_od['unique_id'].map(dest_counts_anim).fillna(0)
                origin_ids_this_hour = hour_routes_anim['origin_id'].unique()
                active_store_footprints = stores_anim_od[stores_anim_od['store_id'].isin(origin_ids_this_hour)]
                plotted_store_types_anim = set(); store_handles_anim = []
                if not active_store_footprints.empty:
                    for store_type, group in active_store_footprints.groupby('store_type'):
                         color = store_colors_map.get(store_type, 'gray'); label = store_type if store_type not in plotted_store_types_anim else None
                         group.plot(ax=ax_anim_od, color=color, alpha=0.6, edgecolor='black', linewidth=0.7, label=label, zorder=3)
                         if label: plotted_store_types_anim.add(store_type); store_handles_anim.append(Patch(facecolor=color, edgecolor='black', label=label, alpha=0.6))
                active_dest_footprints = buildings_anim_od[(buildings_anim_od['deliveries_this_hour'] > 0) & (buildings_anim_od['residential'] == 'yes')]
                if not active_dest_footprints.empty: active_dest_footprints.plot(column='deliveries_this_hour', cmap=cmap_od_anim, norm=norm_od_anim, ax=ax_anim_od, alpha=0.8, edgecolor='darkgreen', linewidth=0.5, zorder=2)
                valid_lines_anim = hour_routes_anim[hour_routes_anim.geometry.is_valid & ~hour_routes_anim.geometry.is_empty]
                if not valid_lines_anim.empty: valid_lines_anim.plot(ax=ax_anim_od, color='black', linewidth=0.5, alpha=0.2, zorder=1)
                try: cx.add_basemap(ax_anim_od, crs='epsg:3857', source=cx.providers.CartoDB.Positron, alpha=0.5, zorder=0)
                except Exception as e: print(f"Basemap failed: {e}")
                time_str = f"{hour:02d}:00 - {hour+1:02d}:00"; title_str_anim = f"O-D Routes - {date.strftime('%a, %b %d')} | {time_str}"
                style_map(ax_anim_od, "")
                ax_anim_od.set_title(title_str_anim, fontsize=16, fontweight='bold')
                cbar_od_anim = fig_anim_od.colorbar(sm_od_anim, ax=ax_anim_od, shrink=0.6, location='left', pad=0.01); cbar_od_anim.set_label("# Deliveries to Destination")
                tick_max_od = int(norm_od_anim.vmax)
                if tick_max_od <= 10: cbar_od_anim.set_ticks(np.arange(1, tick_max_od + 1))
                else: cbar_od_anim.set_ticks(np.linspace(1, tick_max_od, num=5, dtype=int))
                if store_handles_anim: ax_anim_od.legend(handles=store_handles_anim, title="Active Store Types", loc='upper right', fontsize=9)
                frame_path = os.path.join(frames_dir, f"od_day{(date-sim_start_anim).days}_hour_{hour:02d}.png"); plt.savefig(frame_path, dpi=100, bbox_inches='tight'); plt.close(fig_anim_od); frame_count_od += 1
        print(f"Generated {frame_count_od} O-D animation frames.")
    else: print("Skipping O-D animation frame generation as no deliveries were matched.")

    # --- Final Statistics for the matched hour ---
    stats_f = {"Deliveries Matched (Hour)": len(matched_gdf_hour), "Total Deliveries Matched (Week)": len(matched_gdf_all) if matched_gdf_all is not None else 0, "Matching Method": match_method,}
    if matched_gdf_all is not None and 'store_type' in matched_gdf_all.columns: stats_f.update(matched_gdf_all['store_type'].value_counts().rename(lambda x: f"Total Deliveries from {x}").to_dict())
    if matched_gdf_all is not None and 'straight_line_distance_m' in matched_gdf_all.columns: stats_f['Avg Distance (m)'] = matched_gdf_all['straight_line_distance_m'].mean(); stats_f['Median Distance (m)'] = matched_gdf_all['straight_line_distance_m'].median()
    stats_f_df = pd.DataFrame.from_dict(stats_f, orient='index', columns=['Value']); stats_f_df.to_csv(os.path.join(stats_subdir, 'od_stats.csv'))
    print("Module F Stats:"); print(stats_f_df)
else:
    print("Skipping Module F."); save_empty_df(os.path.join(data_subdir, "routing_dataset.csv")); save_empty_gdf(os.path.join(data_subdir, "routing_dataset_detailed_hour.geojson")); save_empty_df(os.path.join(stats_subdir, 'od_stats.csv'))
module_F_time = time.time() - module_F_start
print(f"Module F completed in {module_F_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module G: Reporting
# -----------------------------------------------------------------------------
# ... [Code as in v19.1, now creates both GIFs] ...
print("\n## Module G: Reporting...")
module_G_start = time.time()
report_path = os.path.join(output_dir, "summary_report.txt")
stats_subdir = os.path.join(output_dir, "stats"); viz_subdir = os.path.join(output_dir, "visualizations")
try:
    with open(report_path, 'w') as f:
        f.write("UAS Delivery Demand Simulation Report\n"); f.write("="*40 + "\n"); f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write(f"Seed: {config['random_seed']}\n\n")
        def write_stats(f, module_letter, module_name, csv_path):
            try:
                f.write(f"--- {module_name} ---\n"); df = pd.read_csv(csv_path)
                if df.shape[1] == 2 and df.columns[0] != 'Metric': df = df.rename(columns={df.columns[0]: 'Metric', df.columns[1]: 'Value'})
                if 'Metric' in df.columns: df = df.set_index('Metric')
                try: f.write(df.to_markdown(floatfmt=".2f") + "\n")
                except Exception as md_e: f.write(df.to_string() + "\n")
                module_time = globals().get(f'module_{module_letter}_time', 0)
                f.write(f"(Module completed in {module_time:.1f}s)\n\n")
            except FileNotFoundError: f.write(f"Stats file not found or module skipped: {csv_path}\n\n")
            except Exception as e: f.write(f"Error writing stats for {module_name}: {e}\n\n")
        write_stats(f, "A", "Setup", os.path.join(stats_subdir, 'setup_stats.csv'))
        write_stats(f, "B", "Classification", os.path.join(stats_subdir, 'classification_stats.csv'))
        write_stats(f, "C", "Height Estimation", os.path.join(stats_subdir, 'height_stats.csv'))
        write_stats(f, "D", "Population Allocation", os.path.join(stats_subdir, 'population_stats.csv'))
        write_stats(f, "E", "Demand Modeling", os.path.join(stats_subdir, 'demand_stats.csv'))
        write_stats(f, "F", "O-D Matching", os.path.join(stats_subdir, 'od_stats.csv'))
    print(f"Summary report saved to {report_path}")
except Exception as e: print(f"Error generating summary report: {e}")
# --- Create Demand GIF ---
try:
    print("\nAttempting to create Demand GIF...")
    frames_dir_demand = os.path.join(output_dir, 'animation_frames')
    frames = []; frame_files = sorted(glob.glob(os.path.join(frames_dir_demand, 'demand*.png')))
    if frame_files:
        print(f"Found {len(frame_files)} demand frames...");
        for frame_file in frame_files:
            try: frames.append(Image.open(frame_file))
            except Exception as img_e: print(f"Could not open frame {frame_file}: {img_e}")
        if frames:
            gif_path = os.path.join(viz_subdir, 'demand_animation.gif')
            frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0, optimize=True)
            print(f"Created demand GIF: {gif_path}")
        else: print("No valid demand frames opened.")
    else: print("No demand PNG frames found.")
except ImportError: print("Pillow library not found. Skipping GIF.")
except Exception as gif_e: print(f"Error creating demand GIF: {gif_e}")
# --- Create O-D GIF ---
try:
    print("\nAttempting to create O-D GIF...")
    frames_dir_od = os.path.join(output_dir, 'animation_frames')
    frames = []; frame_files = sorted(glob.glob(os.path.join(frames_dir_od, 'od_*.png'))) # Look for od_ frames
    if frame_files:
        print(f"Found {len(frame_files)} O-D frames...");
        for frame_file in frame_files:
            try: frames.append(Image.open(frame_file))
            except Exception as img_e: print(f"Could not open frame {frame_file}: {img_e}")
        if frames:
            gif_path = os.path.join(viz_subdir, 'od_animation.gif') # Corrected path
            frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0, optimize=True) # Use same duration
            print(f"Created O-D GIF: {gif_path}")
        else: print("No valid O-D frames opened.")
    else: print("No O-D PNG frames found.") # Updated message
except ImportError: print("Pillow library not found. Skipping GIF.")
except Exception as gif_e: print(f"Error creating O-D GIF: {gif_e}")

module_G_time = time.time() - module_G_start
print(f"Module G completed in {module_G_time:.2f} seconds.")
total_time = time.time() - module_A_start
print(f"\nTotal execution time: {total_time:.2f} seconds.")
print("\nProcessing Complete. Check output directory.")

# --- Display Final GIFs in Colab ---
demand_gif_path = os.path.join(output_dir, 'visualizations', 'demand_animation.gif')
od_gif_path = os.path.join(output_dir, 'visualizations', 'od_animation.gif')
if os.path.exists(demand_gif_path):
    from IPython.display import Image as IPImage, display
    print("\nDisplaying Demand Animation GIF:")
    display(IPImage(filename=demand_gif_path))
else: print("\nDemand animation GIF not found.")
if os.path.exists(od_gif_path):
    from IPython.display import Image as IPImage, display
    print("\nDisplaying O-D Animation GIF:")
    display(IPImage(filename=od_gif_path))
else: print("\nO-D animation GIF not found.")