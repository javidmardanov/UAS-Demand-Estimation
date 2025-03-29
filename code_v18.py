# -*- coding: utf-8 -*-
"""
UAS Last-Mile Delivery Demand Modeling and O-D Matching (Colab Implementation - v18 Corrected)

Implements the detailed plan.
Fixes KeyError: nan in Module F during Market Share Weighted matching by ensuring
stores used for probability assignment have a valid 'store_type'.
Also adds minor cleanup and robustness checks.
"""

# -----------------------------------------------------------------------------
# 0. Setup Environment
# -----------------------------------------------------------------------------
print("## 0. Setting up Environment...")
print("Installing required packages...")
# Ensure all necessary packages are included
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
from matplotlib.colors import Normalize # Import Normalize
from matplotlib.cm import ScalarMappable # Import ScalarMappable
from matplotlib.patches import Patch # For custom legends
from matplotlib.lines import Line2D # For custom legends
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

# Matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# --- Optional: Add your Census API Key ---
print("Checking for Census API Key...")
census_api_key = os.environ.get("CENSUS_API_KEY", None) # Standard env variable name
# census_api_key = "YOUR_API_KEY_HERE" # Uncomment and paste key here
if census_api_key and census_api_key != "YOUR_API_KEY_HERE":
    # Cenpy uses the environment variable directly if set
    print("Using Census API Key found in environment variable.")
else:
    census_api_key = None # Explicitly set to None if not found/set
    print("Census API Key not set. Using default access (may be rate-limited).")

print("Environment setup complete.")

# -----------------------------------------------------------------------------
# 1. Configuration (Simulating Streamlit Inputs)
# -----------------------------------------------------------------------------
print("\n## 1. Configuration...")

# Use YAML format for better readability and easier saving/loading
config_yaml = """
area_selection:
  method: 'coordinates'
  coordinates: [30.265, -97.745, 30.270, -97.740] # SW_lat, SW_lon, NE_lat, NE_lon (Smaller Downtown Austin Area)
  buffer_km: 1 # Reduced buffer for faster execution

data_acquisition:
  osm_tags: {'building': True, 'shop': True, 'amenity': True, 'landuse': True}
  census_variables:
    - B19013_001E  # Median household income
    - B01003_001E  # Total population
    - B25010_001E  # Average household size
    - B23025_004E  # Employed population (16+)
    - B23025_002E  # Population in labor force (16+)
    - B25001_001E  # Total housing units
  census_product: "ACSDT5Y2022" # Use 2022 data
  state_code: '48'
  county_code: '453'
  tract_year: 2022 # Use 2022 TIGER/Line

building_classification:
  method: 'rule_based'
  rule_based_parameters: {min_residential_area_m2: 30, max_residential_area_m2: 1000, likely_max_residential_area_m2: 500, min_nonresidential_area_m2: 1500, residential_building_tags: ['residential', 'house', 'apartments', 'detached', 'terrace', 'semidetached_house', 'bungalow', 'dormitory'], nonresidential_building_tags: ['commercial', 'retail', 'industrial', 'warehouse', 'office', 'supermarket', 'shop', 'mall', 'store', 'school', 'hospital', 'church', 'public', 'civic', 'government', 'hotel', 'motel']}
  store_tags:
    shop: ['supermarket', 'department_store', 'convenience', 'mall', 'wholesale', 'grocery']
    building: ['warehouse', 'retail', 'commercial']
    amenity: ['marketplace', 'fast_food']
    name_keywords: ["walmart", "target", "amazon", "kroger", "H-E-B", "heb", "costco", "distribution center", "fulfillment", "supercenter", "warehouse", "grocery", "market", "cvs", "walgreens"]

height_estimation:
  default_height_m: 3.5
  meters_per_level: 3.5
  knn_neighbors: 5
  use_area_feature: True
  max_height_cap_m: 150

population_allocation:
  population_scale_factor: 1.0
  avg_household_size_override: Null # Use None/Null in YAML, e.g., 2.5, or Null to use Census data

demand_model:
  base_deliveries_per_household_per_day: 0.18
  income_coef: 0.6
  pop_density_coef: 0.2
  household_size_coef: -0.15
  employment_coef: 0.1
  reference_values: {ref_median_income: 75000, ref_pop_density: 3000, ref_avg_household_size: 2.5, ref_employment_rate: 95}
  daily_variation: {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 0.8, 6: 0.5}
  # Ensure floats are used and sums to 1.0
  hourly_distribution: {0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.01, 7:0.02, 8:0.04, 9:0.07, 10:0.09, 11:0.10, 12:0.10, 13:0.11, 14:0.12, 15:0.12, 16:0.11, 17:0.09, 18:0.07, 19:0.05, 20:0.02, 21:0.01, 22:0.0, 23:0.0}
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
if 'population_allocation' in config and config['population_allocation'].get('avg_household_size_override') == 'Null':
    config['population_allocation']['avg_household_size_override'] = None
if 'demand_model' in config and 'hourly_distribution' in config['demand_model']:
    try:
        hourly_dist = config['demand_model']['hourly_distribution']
        hourly_dist = {int(k): float(v) if isinstance(v, (int, float)) else 0.0 for k, v in hourly_dist.items()}
        # Ensure all hours 0-23 are present
        for h in range(24):
            if h not in hourly_dist: hourly_dist[h] = 0.0
        hourly_sum = sum(hourly_dist.values())
        if not math.isclose(hourly_sum, 1.0, rel_tol=1e-5):
             print(f"Warning: Hourly distribution sum is {hourly_sum:.4f}. Renormalizing.")
             if hourly_sum > 1e-9: hourly_dist = {k: v / hourly_sum for k, v in hourly_dist.items()}
             else: print("Warning: Hourly sum zero. Using uniform."); hourly_dist = {h: 1/24 for h in range(24)}
        config['demand_model']['hourly_distribution'] = hourly_dist # Assign back
        print("Hourly distribution processed successfully.")
    except Exception as e:
        print(f"ERROR processing hourly distribution: {e}. Using uniform fallback.")
        config['demand_model']['hourly_distribution'] = {h: 1/24 for h in range(24)}

# Apply Random Seed
seed = config['random_seed']
np.random.seed(seed); random.seed(seed)
print(f"Using Random Seed: {seed}")

# Create output directories
output_dir = config['output_dir']
subdirs = ['data', 'visualizations', 'stats', 'config', 'animation_frames']
for subdir in subdirs: os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
print(f"Output directory structure created in: {os.path.abspath(output_dir)}")
try:
    with open(os.path.join(output_dir, 'config', 'used_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
except Exception as e: print(f"Warning: Could not save config file - {e}")
WGS84 = 'EPSG:4326'
print("Configuration loaded.")

# Define data subdirectories consistently
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats"); config_subdir = os.path.join(output_dir, "config"); frames_dir = os.path.join(output_dir, 'animation_frames')

# --- Helper function for CRS estimation ---
def estimate_utm_crs(gdf_wgs84):
    print("Estimating UTM CRS...")
    try:
        if gdf_wgs84.empty or gdf_wgs84.geometry.iloc[0] is None: raise ValueError("Input geometry is None or empty.")
        unified_geom = unary_union(gdf_wgs84.geometry) # Use unary_union method
        if unified_geom.is_empty: raise ValueError("Unified geometry is empty.")
        center_lon, center_lat = unified_geom.centroid.x, unified_geom.centroid.y
        utm_zone = math.floor((center_lon + 180) / 6) + 1
        crs_proj_str = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        if center_lat < 0: crs_proj_str += " +south"
        crs = pyproj.CRS(crs_proj_str); epsg_code = crs.to_epsg()
        if epsg_code: print(f"-> Estimated UTM CRS (EPSG:{epsg_code})"); return f"EPSG:{epsg_code}"
        else: print(f"-> Estimated UTM CRS (Proj String): {crs_proj_str}"); return crs_proj_str
    except Exception as e: print(f"Warning: Could not estimate UTM CRS ({e}). Using default EPSG:32614."); return 'EPSG:32614'

# --- Function to save empty GeoDataFrame ---
def save_empty_gdf(filepath, driver="GeoJSON", crs=WGS84):
     print(f"Saving empty GeoDataFrame to: {filepath}")
     gdf = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=crs)
     try: gdf.to_file(filepath, driver=driver)
     except Exception as e: print(f"Error saving empty GDF {filepath}: {e}")

# --- Function to save empty DataFrame ---
def save_empty_df(filepath, columns=None):
     print(f"Saving empty DataFrame to: {filepath}")
     pd.DataFrame(columns=columns if columns else []).to_csv(filepath, index=False)

# -----------------------------------------------------------------------------
# Module A: Area Selection & Data Acquisition
# -----------------------------------------------------------------------------
print("\n## Module A: Area Selection & Data Acquisition...")
module_A_start = time.time()

# --- Define Study Area ---
sel_config = config['area_selection']
selected_polygon = None
if sel_config['method'] == 'coordinates':
    lat_min, lon_min, lat_max, lon_max = sel_config['coordinates']
    selected_polygon = box(lon_min, lat_min, lon_max, lat_max)
else: raise ValueError(f"Area selection method '{sel_config['method']}' not implemented.")
selected_gdf = gpd.GeoDataFrame(geometry=[selected_polygon], crs=WGS84)
utm_crs = estimate_utm_crs(selected_gdf)
area_selected_km2 = selected_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
print(f"Selected Study Area Polygon defined. Area: {area_selected_km2:.2f} km²")

# --- Fetch Census Tracts FIRST to define Union ---
print("Fetching Census tracts intersecting selected area...")
tracts_gdf = gpd.GeoDataFrame(columns=['GEOID', 'geometry'], geometry='geometry', crs=WGS84)
temp_dir = None
try:
    state = config['data_acquisition']['state_code']; county = config['data_acquisition']['county_code']
    tract_year = config['data_acquisition']['tract_year']
    tract_url = f"https://www2.census.gov/geo/tiger/TIGER{tract_year}/TRACT/tl_{tract_year}_{state}_tract.zip"
    print(f"Downloading tracts from: {tract_url}"); response = requests.get(tract_url, stream=True, timeout=120); response.raise_for_status()
    temp_dir = tempfile.mkdtemp(); zip_path = os.path.join(temp_dir, f"tl_{tract_year}_{state}_tract.zip")
    extracted_dir = os.path.join(temp_dir, "extracted"); os.makedirs(extracted_dir, exist_ok=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192*4): f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extracted_dir)
    shp_files = glob.glob(os.path.join(extracted_dir, '*.shp'))
    if not shp_files: raise FileNotFoundError("No .shp file found in tract zip.")
    all_tracts_in_state = gpd.read_file(shp_files[0]); print(f"Downloaded and read tracts for state {state}.")
    if all_tracts_in_state.crs is None: all_tracts_in_state.crs = WGS84
    else: all_tracts_in_state = all_tracts_in_state.to_crs(WGS84)
    county_tracts = all_tracts_in_state[all_tracts_in_state['COUNTYFP'] == county]
    # Ensure geometries are valid before intersection test
    county_tracts = county_tracts[county_tracts.geometry.is_valid]
    intersecting_mask = county_tracts.geometry.intersects(selected_polygon)
    tracts_gdf = county_tracts[intersecting_mask].copy()
    geoid_col = next((col for col in ['GEOID', f'GEOID{str(tract_year)[-2:]}', 'GEOID20', 'GEOID10'] if col in tracts_gdf.columns), None)
    if geoid_col and geoid_col != 'GEOID': tracts_gdf.rename(columns={geoid_col: 'GEOID'}, inplace=True)
    elif 'GEOID' not in tracts_gdf.columns and all({'STATEFP', 'COUNTYFP', 'TRACTCE'}) <= set(tracts_gdf.columns): tracts_gdf['GEOID'] = tracts_gdf['STATEFP'] + tracts_gdf['COUNTYFP'] + tracts_gdf['TRACTCE']
    if 'GEOID' not in tracts_gdf.columns: raise ValueError("Could not find or create GEOID in tracts.")
    tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str)
    print(f"Filtered to {len(tracts_gdf)} Census tracts intersecting selected area.")
except Exception as e: print(f"ERROR fetching/processing Census tracts: {e}")
finally:
    if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir); print("Cleaned temp tract dir.")

# --- Define Union and Buffered Union based on Tracts ---
if not tracts_gdf.empty:
    # Ensure tract geometries are valid before union
    tracts_gdf = tracts_gdf[tracts_gdf.geometry.is_valid]
    if tracts_gdf.empty: print("Warning: No valid tract geometries after filtering."); raise ValueError("No valid tracts.")
    union_of_tracts_poly = unary_union(tracts_gdf.geometry)
    union_gdf = gpd.GeoDataFrame(geometry=[union_of_tracts_poly], crs=WGS84)
    union_utm = union_gdf.to_crs(utm_crs)
    area_union_km2 = union_utm.geometry.iloc[0].area / 1e6
    buffer_m = sel_config['buffer_km'] * 1000
    buffered_poly_utm = union_utm.geometry.iloc[0].buffer(buffer_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84)
    buffered_polygon = buffered_gdf.geometry.iloc[0]
    area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
    print(f"Union of {len(tracts_gdf)} tracts created. Area: {area_union_km2:.2f} km²")
    print(f"Buffered Union Area ({sel_config['buffer_km']} km buffer). Area: {area_buffered_km2:.2f} km²")
else:
    print("No intersecting tracts found. Using selected rectangle for buffer and fetch.")
    union_of_tracts_poly = selected_polygon; union_gdf = selected_gdf.copy(); area_union_km2 = area_selected_km2
    selected_utm = selected_gdf.to_crs(utm_crs); buffer_m = sel_config['buffer_km'] * 1000; buffered_poly_utm = selected_utm.geometry.iloc[0].buffer(buffer_m)
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84); buffered_polygon = buffered_gdf.geometry.iloc[0]
    area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
    print(f"Using selected rectangle as union. Buffered Area: {area_buffered_km2:.2f} km²")

# --- Fetch OSM Data ---
print("Fetching OSM data within buffered area...")
osm_config = config['data_acquisition']['osm_tags']
osm_gdf = gpd.GeoDataFrame(columns=['osmid', 'unique_id', 'geometry'], geometry='geometry', crs=WGS84)
try:
    osm_fetch_bbox = buffered_polygon.bounds; fetched_osm = ox.features_from_bbox(bbox=osm_fetch_bbox, tags=osm_config)
    osm_gdf = gpd.clip(fetched_osm, buffered_gdf); print(f"Fetched {len(osm_gdf)} OSM features within buffer.")
    osm_gdf = osm_gdf.reset_index()
    if isinstance(osm_gdf.columns, pd.MultiIndex): osm_gdf.columns = ['_'.join(map(str, col)).strip('_') for col in osm_gdf.columns.values]
    if 'osmid' not in osm_gdf.columns and 'element_type_osmid' in osm_gdf.columns: osm_gdf.rename(columns={'element_type_osmid': 'osmid'}, inplace=True)
    elif 'osmid' not in osm_gdf.columns and 'id' in osm_gdf.columns: osm_gdf.rename(columns={'id': 'osmid'}, inplace=True)
    if 'element_type' in osm_gdf.columns and 'osmid' in osm_gdf.columns: osm_gdf['unique_id'] = osm_gdf['element_type'] + osm_gdf['osmid'].astype(str)
    elif 'osmid' in osm_gdf.columns: osm_gdf['unique_id'] = 'way' + osm_gdf['osmid'].astype(str)
    else: osm_gdf = osm_gdf.reset_index().rename(columns={'index':'generated_id'}); osm_gdf['unique_id'] = 'feat_' + osm_gdf['generated_id'].astype(str)
    osm_gdf = osm_gdf[~osm_gdf.geometry.isna()]; osm_gdf['geometry'] = osm_gdf.geometry.buffer(0); osm_gdf = osm_gdf[osm_gdf.geometry.is_valid]
    print(f"Remaining valid OSM features after cleaning: {len(osm_gdf)}")
except Exception as e: print(f"ERROR fetching/processing OSM data: {e}")

# --- Fetch Census Data ---
print("Fetching Census demographic data...")
cen_config = config['data_acquisition']
census_columns = ['GEOID'] + cen_config['census_variables'] + ['total_population', 'median_income', 'avg_household_size', 'employment_rate', 'pop_density', 'total_housing_units']
census_data_df = pd.DataFrame(columns=census_columns)
census_data_fetched = False
if not tracts_gdf.empty:
    try:
        print(f"Attempting cenpy query for {cen_config['census_product']}...")
        conn = products.APIConnection(cen_config['census_product'])
        census_api_data = conn.query(cols=cen_config['census_variables'], geo_unit='tract:*', geo_filter={'state': state, 'county': county}, apikey=census_api_key)
        census_api_data['GEOID'] = census_api_data['state'] + census_api_data['county'] + census_api_data['tract']; census_api_data['GEOID'] = census_api_data['GEOID'].astype(str)
        census_api_data = census_api_data[census_api_data['GEOID'].isin(tracts_gdf['GEOID'])]
        print(f"Fetched demographic data for {len(census_api_data)} tracts via cenpy.")
        census_data_fetched = True
    except Exception as cenpy_e:
        print(f"WARNING: cenpy query failed: {cenpy_e}. Trying direct API call...")
        try:
            census_year_api = ''.join(filter(str.isdigit, cen_config['census_product']))[-4:]
            api_endpoint = f"https://api.census.gov/data/{census_year_api}/acs/acs5"; variables_str = ",".join(['NAME'] + cen_config['census_variables']); geo_str = f"for=tract:*&in=state:{state}&in=county:{county}"
            api_url = f"{api_endpoint}?get={variables_str}&{geo_str}";
            if census_api_key: api_url += f"&key={census_api_key}"
            print(f"Direct API URL: {api_endpoint}?get=..."); response = requests.get(api_url, timeout=60); response.raise_for_status(); data = response.json()
            if len(data) > 1:
                census_api_data = pd.DataFrame(data[1:], columns=data[0]); census_api_data['GEOID'] = census_api_data['state'] + census_api_data['county'] + census_api_data['tract']; census_api_data['GEOID'] = census_api_data['GEOID'].astype(str)
                census_api_data = census_api_data[census_api_data['GEOID'].isin(tracts_gdf['GEOID'])]; print(f"Fetched data for {len(census_api_data)} tracts via direct API."); census_data_fetched = True
            else: print("Direct API call returned no data rows.")
        except Exception as direct_api_e: print(f"ERROR fetching Census data via direct API: {direct_api_e}")
    if census_data_fetched and not census_api_data.empty:
        print("Processing fetched Census data..."); tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str); census_api_data['GEOID'] = census_api_data['GEOID'].astype(str)
        for col in cen_config['census_variables']:
            if col in census_api_data.columns: census_api_data[col] = pd.to_numeric(census_api_data[col], errors='coerce')
        census_api_data['total_population'] = census_api_data.get('B01003_001E', 0)
        census_api_data['median_income'] = census_api_data.get('B19013_001E', 0)
        census_api_data['avg_household_size'] = census_api_data.get('B25010_001E', 0)
        census_api_data['total_housing_units'] = census_api_data.get('B25001_001E', 0)
        pop_in_labor_force = census_api_data.get('B23025_002E', pd.Series(0, index=census_api_data.index))
        employed_pop = census_api_data.get('B23025_004E', pd.Series(0, index=census_api_data.index))
        census_api_data['employment_rate'] = np.where(pop_in_labor_force > 0, (employed_pop / pop_in_labor_force) * 100, 0)
        tracts_with_area = tracts_gdf.to_crs(utm_crs); tracts_with_area['area_sqkm'] = tracts_with_area.geometry.area / 1e6
        census_api_data = census_api_data.merge(tracts_with_area[['GEOID', 'area_sqkm']], on='GEOID', how='left')
        census_api_data['pop_density'] = np.where(census_api_data['area_sqkm'] > 0, census_api_data['total_population'] / census_api_data['area_sqkm'], 0)
        cols_to_keep = ['GEOID', 'total_population', 'median_income', 'avg_household_size', 'employment_rate', 'pop_density', 'total_housing_units']
        existing_cols_to_keep = [col for col in cols_to_keep if col in census_api_data.columns]
        census_data_df = census_api_data[existing_cols_to_keep].fillna(0); print("Census data processing complete.")
    else: print("Census data could not be fetched or processed.")
else: print("No tracts found, skipping Census data fetch.")

# --- Save Outputs ---
print("Saving Module A outputs...")
selected_gdf.to_file(os.path.join(data_subdir, "area_selection.geojson"), driver="GeoJSON")
union_gdf.to_file(os.path.join(data_subdir, "union_of_tracts.geojson"), driver="GeoJSON")
buffered_gdf.to_file(os.path.join(data_subdir, "buffered_area.geojson"), driver="GeoJSON")
if not osm_gdf.empty: osm_gdf.to_file(os.path.join(data_subdir, "initial_osm_features.geojson"), driver="GeoJSON")
else: save_empty_gdf(os.path.join(data_subdir, "initial_osm_features.geojson"))
tracts_gdf.to_file(os.path.join(data_subdir, "census_tracts.geojson"), driver="GeoJSON")
census_data_df.to_csv(os.path.join(data_subdir, "census_data.csv"), index=False)

# --- Visualize Setup ---
print("Generating setup visualizations...")
fig1, ax1 = plt.subplots(figsize=(10, 10)); selected_gdf.to_crs(epsg=3857).plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3, label='Selected Area')
try: cx.add_basemap(ax1, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
except Exception as e: print(f"Basemap Fig 1 failed: {e}")
ax1.set_title('Figure 1: Selected Study Area'); ax1.set_axis_off(); ax1.legend(); plt.savefig(os.path.join(viz_subdir, 'setup_map_1.png'), dpi=150, bbox_inches='tight'); plt.close(fig1)
fig2, ax2 = plt.subplots(figsize=(10, 10)); union_plot_gdf = union_gdf.to_crs(epsg=3857); tracts_plot_gdf = tracts_gdf.to_crs(epsg=3857); selected_plot_gdf = selected_gdf.to_crs(epsg=3857)
osm_buildings_in_union = gpd.GeoDataFrame()
if not osm_gdf.empty and not union_gdf.empty: osm_buildings_in_union = gpd.clip(osm_gdf[osm_gdf['building'].notna()], union_gdf).to_crs(epsg=3857)
tracts_plot_gdf.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.5, label='Tracts')
if not osm_buildings_in_union.empty: osm_buildings_in_union.plot(ax=ax2, color='darkgray', alpha=0.6, linewidth=0, label='Buildings in Union')
union_plot_gdf.boundary.plot(ax=ax2, edgecolor='green', linewidth=2.5, label='Union of Tracts'); selected_plot_gdf.boundary.plot(ax=ax2, edgecolor='red', linewidth=1.5, linestyle='--', label='Selected Area')
try: cx.add_basemap(ax2, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
except Exception as e: print(f"Basemap Fig 2 failed: {e}")
ax2.set_title('Figure 2: Tracts, Union, Buildings & Selection'); ax2.set_axis_off(); ax2.legend(); plt.savefig(os.path.join(viz_subdir, 'setup_map_2.png'), dpi=150, bbox_inches='tight'); plt.close(fig2)
fig3, ax3 = plt.subplots(figsize=(10, 10)); buffered_plot_gdf = buffered_gdf.to_crs(epsg=3857); union_plot_gdf3 = union_gdf.to_crs(epsg=3857); all_osm_plot_gdf = osm_gdf.to_crs(epsg=3857)
store_footprints_gdf_placeholder = gpd.GeoDataFrame() # Placeholder for Fig 3
buffered_plot_gdf.plot(ax=ax3, facecolor='lightblue', edgecolor='blue', alpha=0.15, label=f"{sel_config['buffer_km']}km Buffer")
all_osm_plot_gdf[all_osm_plot_gdf['building'].notna()].plot(ax=ax3, color='gray', alpha=0.4, linewidth=0, label='Buildings in Buffer')
if not store_footprints_gdf_placeholder.empty: store_footprints_gdf_placeholder.to_crs(epsg=3857).plot(ax=ax3, facecolor='none', edgecolor='magenta', linewidth=1.5, label='Store Footprints')
union_plot_gdf3.boundary.plot(ax=ax3, edgecolor='green', linewidth=2.5, label='Union of Tracts')
try: cx.add_basemap(ax3, crs='epsg=3857', source=cx.providers.CartoDB.Positron)
except Exception as e: print(f"Basemap Fig 3 failed: {e}")
ax3.set_title('Figure 3: Buffer, All Buildings, & Stores (Stores to be identified)'); ax3.set_axis_off(); ax3.legend(); plt.savefig(os.path.join(viz_subdir, 'setup_map_3.png'), dpi=150, bbox_inches='tight'); plt.close(fig3)

# --- Statistics ---
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
try:
    osm_file = os.path.join(data_subdir, "initial_osm_features.geojson"); union_file = os.path.join(data_subdir, "union_of_tracts.geojson")
    if not os.path.exists(osm_file): raise FileNotFoundError("initial_osm_features.geojson not found.")
    if not os.path.exists(union_file): raise FileNotFoundError("union_of_tracts.geojson not found.")
    buildings_all = gpd.read_file(osm_file); union_gdf = gpd.read_file(union_file)
    if buildings_all.empty or union_gdf.empty: raise ValueError("Input data empty.")
    if 'unique_id' not in buildings_all.columns:
        if 'element_type' in buildings_all.columns and 'osmid' in buildings_all.columns: buildings_all['unique_id'] = buildings_all['element_type'] + buildings_all['osmid'].astype(str)
        elif 'osmid' in buildings_all.columns: buildings_all['unique_id'] = 'way' + buildings_all['osmid'].astype(str)
        else: buildings_all = buildings_all.reset_index().rename(columns={'index':'unique_id'}); buildings_all['unique_id'] = 'feat_' + buildings_all['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading initial OSM features: {e}. Skipping Module B.")

if not buildings_all.empty and not union_gdf.empty:
    cls_config = config['building_classification']
    # --- Classification Logic ---
    def classify_building_rules(row, config):
        rules = config['rule_based_parameters']
        tags = row.to_dict(); area = row.get('building_area_m2', None); name = str(tags.get('name', '')).lower()
        non_res_buildings = rules.get('nonresidential_building_tags', [])
        res_buildings = rules.get('residential_building_tags', [])
        non_res_keywords = ['school', 'church', 'store', 'office', 'center', 'bank', 'hospital', 'clinic', 'inc', 'corp', 'ltd', 'university', 'college', 'hotel', 'motel', 'station', 'county', 'city of', 'state of']
        res_keywords = ['apartments', 'residences', 'condos', 'living', 'manor', 'place', 'tower', 'lofts', 'village', 'gardens']
        if tags.get('shop') or tags.get('amenity') or tags.get('office') or tags.get('public_transport') or tags.get('industrial') or tags.get('leisure') or tags.get('tourism'): return 'no'
        if tags.get('building') in non_res_buildings: return 'no'
        if any(keyword in name for keyword in non_res_keywords): return 'no'
        if tags.get('building') in res_buildings: return 'yes'
        if any(keyword in name for keyword in res_keywords): return 'yes'
        if area is not None:
            if area < rules.get('min_residential_area_m2', 30) and tags.get('building') != 'house': return 'no'
            if area < rules.get('likely_max_residential_area_m2', 500): return 'likely_yes'
            if area > rules.get('min_nonresidential_area_m2', 1500): return 'likely_no'
        return 'unknown'
    # --- Store Identification Logic ---
    store_cfg = cls_config.get('store_tags', {}); store_keywords_pattern = "|".join(store_cfg.get('name_keywords', []))
    def identify_store_type(row):
        tags = row.to_dict(); name = str(tags.get('name', '')).lower(); shop = str(tags.get('shop', '')).lower(); bldg = str(tags.get('building', '')).lower(); amenity = str(tags.get('amenity', '')).lower()
        if 'walmart' in name: return 'Walmart'
        if 'target' in name: return 'Target'
        if 'kroger' in name: return 'Kroger'
        if 'h-e-b' in name or 'heb' in name: return 'H-E-B'
        if 'costco' in name: return 'Costco'
        if 'amazon' in name: return 'Warehouse/Fulfillment'
        if 'distribution center' in name or 'fulfillment' in name: return 'Warehouse/Fulfillment'
        if 'warehouse' in name or bldg == 'warehouse': return 'Warehouse/Fulfillment'
        if 'supercenter' in name or 'supermarket' in shop or 'grocery' in name or 'market' in name: return 'Retail Hub'
        if 'department_store' in shop or 'mall' in shop or 'wholesale' in shop: return 'Retail Hub'
        # Check specific tags from config
        if shop in store_cfg.get('shop', []): return 'Retail Hub'
        if amenity in store_cfg.get('amenity', []): return 'Retail Hub'
        # Check keywords pattern only if it's not empty
        if store_keywords_pattern and re.search(store_keywords_pattern, name, re.IGNORECASE): return 'Retail Hub / Warehouse'
        return None # Return None if not identified

    if 'building_area_m2' not in buildings_all.columns: buildings_all['building_area_m2'] = buildings_all.to_crs(utm_crs).geometry.area
    print("Classifying buildings (Rule-Based)...")
    buildings_all['residential_rule'] = buildings_all.apply(lambda row: classify_building_rules(row, cls_config), axis=1)
    buildings_all['residential'] = buildings_all['residential_rule'].replace({'likely_yes': 'yes', 'likely_no': 'no', 'unknown': 'no'})
    print("Identifying stores...")
    buildings_all['store_type'] = buildings_all.apply(identify_store_type, axis=1)
    buildings_all['is_store'] = buildings_all['store_type'].notna()
    buildings_all.loc[buildings_all['is_store'], 'residential'] = 'no'
    stores_gdf = buildings_all[buildings_all['is_store']].copy()
    if not stores_gdf.empty:
        stores_gdf['geometry'] = stores_gdf.geometry.centroid
        stores_gdf = stores_gdf.reset_index(drop=True).reset_index().rename(columns={'index':'store_seq_id'})
        stores_gdf['store_id'] = 'Store_' + stores_gdf['store_seq_id'].astype(str)
        stores_gdf = stores_gdf[['unique_id', 'osmid', 'store_id', 'name', 'shop', 'amenity', 'building', 'store_type', 'geometry']].copy()
    print(f"Identified {len(stores_gdf)} potential stores.")
    if not stores_gdf.empty: print("Store types found:\n", stores_gdf['store_type'].value_counts())
    else: print("No stores identified.")
    print("Saving Module B outputs...")
    cols_to_save_b = [col for col in buildings_all.columns if col not in ['knn_details']]
    buildings_save_b = buildings_all[cols_to_save_b].copy()
    for col in buildings_save_b.select_dtypes(include=['object']).columns:
        mask = buildings_save_b[col].apply(lambda x: isinstance(x, (list, dict)))
        if mask.any(): buildings_save_b.loc[mask, col] = buildings_save_b.loc[mask, col].astype(str)
    buildings_save_b.to_file(os.path.join(data_subdir, "classified_buildings.geojson"), driver="GeoJSON")
    stores_gdf.to_file(os.path.join(data_subdir, "stores.geojson"), driver="GeoJSON")
    print("Clipping buildings to union of tracts area...")
    buildings_in_union_area = gpd.clip(buildings_all, union_gdf)
    print(f"Buildings within union: {len(buildings_in_union_area)}")
    print("Generating classification visualization (Fig 4)...")
    fig_b4, ax_b4 = plt.subplots(figsize=(12, 12)); plot_data_b4 = buildings_in_union_area.to_crs(epsg=3857); union_plot_gdf_b4 = union_gdf.to_crs(epsg=3857); stores_plot_b4 = stores_gdf.to_crs(epsg=3857)
    plot_data_b4[plot_data_b4['residential'] == 'no'].plot(ax=ax_b4, color='gray', alpha=0.6, label='Non-Residential', linewidth=0.5)
    plot_data_b4[plot_data_b4['residential'] == 'yes'].plot(ax=ax_b4, color='green', alpha=0.7, label='Residential', linewidth=0.5)
    union_plot_gdf_b4.boundary.plot(ax=ax_b4, edgecolor='black', linewidth=2.0, label='Union of Tracts')
    if not stores_plot_b4.empty: stores_plot_b4.plot(ax=ax_b4, color='red', marker='*', markersize=100, label='Stores (in Buffer)', zorder=5)
    try: cx.add_basemap(ax_b4, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    ax_b4.set_title('Figure 4: Classification within Union & Stores in Buffer'); ax_b4.set_axis_off(); ax_b4.legend(loc='upper left'); ax_b4.set_aspect('equal', adjustable='box') # Set aspect ratio
    plt.savefig(os.path.join(viz_subdir, 'classification_map.png'), dpi=150, bbox_inches='tight'); plt.close(fig_b4)
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
# ... [Code as in v16, operates on buildings_in_union] ...
print("\n## Module C: Building Height Estimation...")
module_C_start = time.time()
buildings_with_heights = gpd.GeoDataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    buildings_classified = gpd.read_file(os.path.join(data_subdir, "classified_buildings.geojson"))
    union_gdf = gpd.read_file(os.path.join(data_subdir, "union_of_tracts.geojson")) # Load union
    if buildings_classified.empty or union_gdf.empty: raise ValueError("Input data missing.")
    if 'unique_id' not in buildings_classified.columns:
        if 'element_type' in buildings_classified.columns and 'osmid' in buildings_classified.columns: buildings_classified['unique_id'] = buildings_classified['element_type'] + buildings_classified['osmid'].astype(str)
        elif 'osmid' in buildings_classified.columns: buildings_classified['unique_id'] = 'way' + buildings_classified['osmid'].astype(str)
        else: buildings_classified = buildings_classified.reset_index().rename(columns={'index':'unique_id'}); buildings_classified['unique_id'] = 'bldg_' + buildings_classified['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for height estimation: {e}. Skipping Module C.")

if not buildings_classified.empty and not union_gdf.empty:
    print(f"Clipping {len(buildings_classified)} buildings to union area for height processing...")
    buildings_in_union = gpd.clip(buildings_classified, union_gdf)
    print(f"Processing heights for {len(buildings_in_union)} buildings within union.")

    h_config = config['height_estimation']
    # --- Height Estimation Logic ---
    def extract_known_height(row, meters_per_level): # Same as v16
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
    def estimate_missing_heights(buildings_to_estimate, known_buildings_df, config): # Same as v16
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
            buildings_with_heights['knn_details'] = knn_details_series.apply(lambda x: json.dumps(x) if x and isinstance(x, dict) else None)
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
        plot_data_c[plot_data_c['height_source'] != 'tag'].plot(column='height_m', cmap=cmap_height, norm=norm_height, ax=ax_c[0], alpha=0.7, edgecolor='gray', linewidth=0.2, linestyle='--')
        plot_data_c[plot_data_c['height_source'] == 'tag'].plot(column='height_m', cmap=cmap_height, norm=norm_height, legend=False, ax=ax_c[0], alpha=0.9, edgecolor='black', linewidth=0.3)
        sm = ScalarMappable(cmap=cmap_height, norm=norm_height); sm.set_array([]); cbar = fig_c.colorbar(sm, ax=ax_c[0], shrink=0.7); cbar.set_label("Height (m)")
        try: cx.add_basemap(ax_c[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
        except Exception as e: print(f"Basemap failed: {e}")
        ax_c[0].set_title('Building Heights (Within Union Area)'); ax_c[0].set_axis_off(); ax_c[0].set_aspect('equal', adjustable='box') # Ensure aspect ratio
        sns.histplot(plot_data_c['height_m'], bins=30, kde=True, ax=ax_c[1]); ax_c[1].set_title('Distribution of Building Heights'); ax_c[1].set_xlabel('Height (m)'); ax_c[1].set_ylabel('Number of Buildings')
    else: ax_c[0].text(0.5, 0.5, "No buildings", ha='center'); ax_c[1].text(0.5, 0.5, "No data", ha='center')
    plt.tight_layout(); plt.savefig(os.path.join(viz_subdir, 'height_visualization.png'), dpi=150, bbox_inches='tight'); plt.close(fig_c)
    stats_c = buildings_with_heights['height_m'].describe().to_dict(); stats_c.update(buildings_with_heights['height_source'].value_counts().rename(lambda x: f"count_{x}").to_dict())
    stats_c_df = pd.DataFrame.from_dict(stats_c, orient='index', columns=['Value']); stats_c_df.to_csv(os.path.join(stats_subdir, 'height_stats.csv'))
    print("Module C Stats:"); print(stats_c_df)
else:
    print("Skipping Module C: Input data empty.")
    save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson")); save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'), columns=['Value'])
module_C_time = time.time() - module_C_start
print(f"Module C completed in {module_C_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module D: Population Allocation
# -----------------------------------------------------------------------------
# ... [Code as in v16, operates on buildings_with_heights (which are already clipped)] ...
print("\n## Module D: Population Allocation...")
module_D_start = time.time()
buildings_with_pop = gpd.GeoDataFrame()
tracts_with_data_d = gpd.GeoDataFrame()
census_data_df_d = pd.DataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    buildings_with_heights = gpd.read_file(os.path.join(data_subdir, "buildings_with_heights.geojson")) # This now contains ONLY buildings within the union
    tracts_gdf_d = gpd.read_file(os.path.join(data_subdir, "census_tracts.geojson"))
    census_data_df_d = pd.read_csv(os.path.join(data_subdir, "census_data.csv"))
    if buildings_with_heights.empty or tracts_gdf_d.empty or census_data_df_d.empty or 'GEOID' not in census_data_df_d.columns: raise ValueError("Input data missing/invalid.")
    if 'unique_id' not in buildings_with_heights.columns:
        if 'element_type' in buildings_with_heights.columns and 'osmid' in buildings_with_heights.columns: buildings_with_heights['unique_id'] = buildings_with_heights['element_type'] + buildings_with_heights['osmid'].astype(str)
        elif 'osmid' in buildings_with_heights.columns: buildings_with_heights['unique_id'] = 'way' + buildings_with_heights['osmid'].astype(str)
        else: buildings_with_heights = buildings_with_heights.reset_index().rename(columns={'index':'unique_id'}); buildings_with_heights['unique_id'] = 'bldg_' + buildings_with_heights['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for population allocation: {e}. Skipping Module D.")

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
        residential_buildings = residential_buildings.to_crs(tracts_with_data_d.crs) # Match CRS
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
                      print(f"sjoin_nearest completed, found {len(nearest_join)} potential assignments.")
                      nearest_join = nearest_join.loc[~nearest_join.index.duplicated(keep='first')]
                      geo_col_right = 'GEOID_right' if 'GEOID_right' in nearest_join.columns else 'GEOID'
                      pop_col_right = 'total_population_right' if 'total_population_right' in nearest_join.columns else 'total_population'
                      hh_col_right = 'avg_household_size_right' if 'avg_household_size_right' in nearest_join.columns else 'avg_household_size'
                      if geo_col_right in nearest_join.columns:
                          update_data = nearest_join[[geo_col_right, pop_col_right, hh_col_right]].rename(columns={geo_col_right: 'GEOID', pop_col_right: 'total_population', hh_col_right: 'avg_household_size'})
                          update_data.index = missing_indices # Use original index for update
                          bldg_tract_join.update(update_data[['GEOID', 'total_population', 'avg_household_size']], overwrite=True)
                          print(f"Successfully updated {len(update_data.dropna())} buildings.")
                      else: print("ERROR: Necessary columns not found in nearest_join result.")
                  else: print("No missing buildings found to assign nearest tract.")

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
        print("No residential buildings found within the union. Skipping population allocation.")
        buildings_with_pop = buildings_with_heights.copy()
        for col in ['GEOID', 'total_population', 'avg_household_size', 'total_tract_res_volume_m3', 'volume_share', 'allocated_population', 'estimated_households']: buildings_with_pop[col] = 0 if col not in ['GEOID'] else None

    print("Saving Module D outputs...")
    cols_to_save_d = [col for col in buildings_with_pop.columns if col not in ['knn_details', 'simulated_daily_deliveries', 'index_right', 'scale_factor']]; buildings_save_d = buildings_with_pop[cols_to_save_d].copy()
    for col in buildings_save_d.select_dtypes(include=['object']).columns:
        mask = buildings_save_d[col].apply(lambda x: isinstance(x, (list, dict)))
        if mask.any(): buildings_save_d.loc[mask, col] = buildings_save_d.loc[mask, col].astype(str)
    buildings_save_d.to_file(os.path.join(data_subdir, "buildings_with_population.geojson"), driver="GeoJSON")
    print("Generating population visualization...")
    fig_d, ax_d = plt.subplots(1, 2, figsize=(18, 8))
    # ... [Viz code] ...
    plot_data_d = buildings_with_pop.to_crs(epsg=3857)
    selected_gdf_proj = selected_gdf.to_crs(epsg=3857) # Use selected area for context
    # Get union GDF for clipping visualization
    try:
         union_gdf_d = gpd.read_file(os.path.join(data_subdir, "union_of_tracts.geojson"))
         union_gdf_proj = union_gdf_d.to_crs(epsg=3857)
         plot_data_d_clipped = gpd.clip(plot_data_d, union_gdf_proj) # Clip to union
    except Exception as clip_e:
         print(f"Warning: Clipping failed, plotting all data in union - {clip_e}")
         plot_data_d_clipped = plot_data_d[plot_data_d.geometry.within(unary_union(union_gdf.to_crs(epsg=3857).geometry))] # Fallback clip

    res_plot_d = plot_data_d_clipped[plot_data_d_clipped['residential'] == 'yes']
    nonres_plot_d = plot_data_d_clipped[plot_data_d_clipped['residential'] == 'no']

    nonres_plot_d.plot(ax=ax_d[0], color='lightgray', alpha=0.5, label='Non-Residential')
    if not res_plot_d.empty and res_plot_d['allocated_population'].sum() > 0:
         cmap_pop = 'Reds'; pop_max_quantile = res_plot_d['allocated_population'].quantile(0.98)
         norm_pop = Normalize(vmin=0, vmax=pop_max_quantile if pop_max_quantile > 0 else 1)
         res_plot_d.plot(column='allocated_population', cmap=cmap_pop, norm=norm_pop, legend=False, ax=ax_d[0], alpha=0.8)
         sm_pop = ScalarMappable(cmap=cmap_pop, norm=norm_pop); sm_pop.set_array([])
         cbar_pop = fig_d.colorbar(sm_pop, ax=ax_d[0], shrink=0.7); cbar_pop.set_label("Allocated Population")
    else: ax_d[0].text(0.5, 0.5, "No residential population", transform=ax_d[0].transAxes, ha='center')
    try: cx.add_basemap(ax_d[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    ax_d[0].set_title('Population Allocation (Union Area)'); ax_d[0].set_axis_off(); ax_d[0].set_aspect('equal', adjustable='box') # Aspect Ratio

    pop_for_hist = res_plot_d[res_plot_d['allocated_population'] > 0]['allocated_population']
    if not pop_for_hist.empty:
        sns.histplot(pop_for_hist.clip(upper=pop_for_hist.quantile(0.98)), bins=30, kde=False, ax=ax_d[1])
        ax_d[1].set_title('Distribution of Allocated Population (>0)')
        ax_d[1].set_xlabel('Est. Population per Building'); ax_d[1].set_ylabel('Number of Buildings')
    else: ax_d[1].text(0.5, 0.5, "No data", ha='center'); ax_d[1].set_title('Distribution of Allocated Population')

    plt.tight_layout(); plt.savefig(os.path.join(viz_subdir, 'population_visualization.png'), dpi=150, bbox_inches='tight'); plt.close(fig_d)

    # --- Statistics ---
    stats_d = {}; stats_d_df = pd.DataFrame(columns=['Value'])
    if not res_plot_d.empty:
        stats_d = res_plot_d['allocated_population'].describe().to_dict()
        stats_d['total_allocated_population'] = res_plot_d['allocated_population'].sum()
        stats_d.update(res_plot_d['estimated_households'].describe().rename(lambda x: f"households_{x}").to_dict())
        stats_d_df = pd.DataFrame.from_dict(stats_d, orient='index', columns=['Value'])
    stats_d_df.to_csv(os.path.join(stats_subdir, 'population_stats.csv'))
    print("Module D Stats:"); print(stats_d_df)

else:
    print("Skipping Module D.")
    save_empty_gdf(os.path.join(data_subdir, "buildings_with_population.geojson"))
    save_empty_df(os.path.join(stats_subdir, 'population_stats.csv'), columns=['Value'])

module_D_time = time.time() - module_D_start
print(f"Module D completed in {module_D_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module E: Demand Modeling
# -----------------------------------------------------------------------------
# ... [Code largely as in v16] ...
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
        print(f"Merging missing Census columns for demand: {missing_census_cols}")
        buildings_with_pop['GEOID'] = buildings_with_pop['GEOID'].astype(str); census_data_df_e['GEOID'] = census_data_df_e['GEOID'].astype(str)
        buildings_with_pop = buildings_with_pop.merge(census_data_df_e[[col for col in required_census_cols if col in census_data_df_e.columns]], on='GEOID', how='left', suffixes=('', '_census'))
    for col, ref_val in zip(['pop_density', 'employment_rate', 'median_income', 'avg_household_size'], [ref['ref_pop_density'], ref['ref_employment_rate'], ref['ref_median_income'], ref['ref_avg_household_size']]):
         if col not in buildings_with_pop.columns: buildings_with_pop[col] = ref_val
         else: buildings_with_pop[col] = buildings_with_pop[col].fillna(ref_val)
    buildings_with_pop['estimated_households'] = pd.to_numeric(buildings_with_pop['estimated_households'], errors='coerce').fillna(0).astype(int)

    def calculate_demand_rate(row, config): # Formula as in v16
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
    print("Calculating base demand rates...")
    demand_results = buildings_with_pop.apply(lambda row: calculate_demand_rate(row, config), axis=1)
    buildings_with_pop[['demand_rate', 'income_adj', 'density_adj', 'hh_size_adj', 'emp_adj']] = pd.DataFrame(demand_results.tolist(), index=buildings_with_pop.index)

    print("Simulating delivery events...")
    sim_start = datetime.strptime(dem_config['simulation_start_date'], '%Y-%m-%d'); sim_end = sim_start + timedelta(days=dem_config['simulation_duration_days'])
    delivery_events = []; building_daily_counts = {idx: [] for idx in buildings_with_pop.index}
    union_geom_unary = unary_union(union_gdf_e.geometry)
    # Ensure mask uses same index type if possible
    buildings_with_pop = buildings_with_pop.set_index('unique_id', drop=False)
    buildings_in_union_mask = buildings_with_pop.geometry.within(union_geom_unary)
    sim_buildings = buildings_with_pop[buildings_in_union_mask & (buildings_with_pop['demand_rate'] > 0) & (buildings_with_pop['residential'] == 'yes')]
    if not sim_buildings.empty:
        hourly_dist_config = dem_config['hourly_distribution']; hourly_dist_values = list(hourly_dist_config.values()); hourly_dist_keys = list(hourly_dist_config.keys()); hourly_probs = np.array(hourly_dist_values, dtype=float); probs_sum = hourly_probs.sum()
        for date in pd.date_range(sim_start, sim_end - timedelta(days=1)):
            day_of_week = date.weekday(); sim_month = date.month; day_factor = dem_config['daily_variation'].get(day_of_week, 1.0); month_factor = dem_config['monthly_factors'].get(sim_month, 1.0)
            daily_adjusted_rates = sim_buildings['demand_rate'] * day_factor * month_factor
            num_deliveries_today = np.random.poisson(daily_adjusted_rates.clip(lower=0))
            for idx, count in zip(daily_adjusted_rates.index, num_deliveries_today): building_daily_counts.setdefault(idx, []).append(count)
            total_deliveries_today = num_deliveries_today.sum()
            if total_deliveries_today > 0:
                building_indices = np.repeat(daily_adjusted_rates.index, num_deliveries_today)
                if not hourly_dist_values or probs_sum <= 1e-9: assigned_hours = np.random.choice(24, size=total_deliveries_today)
                else: normalized_probs = hourly_probs / probs_sum; assigned_hours = np.random.choice(hourly_dist_keys, size=total_deliveries_today, p=normalized_probs)
                skipped_events = 0
                for i in range(total_deliveries_today):
                    bldg_idx = building_indices[i]
                    if bldg_idx not in sim_buildings.index: skipped_events += 1; continue
                    building_row = sim_buildings.loc[bldg_idx]
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
    buildings_with_pop['simulated_daily_deliveries'] = buildings_with_pop.index.map(lambda idx: str(building_daily_counts.get(idx, [0]*dem_config['simulation_duration_days'])))
    buildings_with_pop['simulated_total_deliveries'] = buildings_with_pop['simulated_daily_deliveries'].apply(lambda x: sum(eval(x)) if isinstance(x, str) and x.startswith('[') else 0)
    buildings_with_demand = buildings_with_pop.reset_index(drop=True) # Reset index after simulation

    # --- Generate Animation Frames ---
    print("\nGenerating Hourly Delivery Animation Frames...")
    if not delivery_events_df.empty:
        os.makedirs(frames_dir, exist_ok=True); animation_norm = Normalize(vmin=0, vmax=3); animation_cmap = plt.cm.YlOrRd
        buildings_for_anim = buildings_with_demand.to_crs(epsg=3857); tracts_for_anim = union_gdf_e.to_crs(epsg=3857)
        frame_count = 0
        for date in pd.date_range(sim_start, sim_end - timedelta(days=1)):
             sample_day_str = date.strftime('%Y-%m-%d'); print(f"  Generating frames for {sample_day_str}...")
             for hour in range(6, 22):
                hour_start = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=hour); hour_end = hour_start + timedelta(hours=1)
                hour_deliveries = delivery_events_df[ (delivery_events_df['timestamp'] >= hour_start) & (delivery_events_df['timestamp'] < hour_end) ]
                hour_counts = hour_deliveries.groupby('building_unique_id').size().to_dict()
                buildings_for_anim['deliveries_this_hour'] = buildings_for_anim['unique_id'].map(lambda uid: hour_counts.get(uid, 0))
                buildings_for_anim['active'] = buildings_for_anim['deliveries_this_hour'] > 0
                fig_anim, ax_anim = plt.subplots(figsize=(12, 12))
                tracts_for_anim.boundary.plot(ax=ax_anim, color='black', linewidth=1.0, alpha=0.6)
                inactive = buildings_for_anim[(~buildings_for_anim['active']) & buildings_for_anim.geometry.within(tracts_for_anim.iloc[0].geometry)]
                active = buildings_for_anim[buildings_for_anim['active'] & buildings_for_anim.geometry.within(tracts_for_anim.iloc[0].geometry)]
                if not inactive.empty: inactive.plot(ax=ax_anim, color='lightgray', edgecolor='darkgray', linewidth=0.1, alpha=0.5)
                if not active.empty: active.plot(column='deliveries_this_hour', ax=ax_anim, cmap=animation_cmap, norm=animation_norm, edgecolor='black', linewidth=0.3, alpha=0.9)
                try: cx.add_basemap(ax_anim, crs=buildings_for_anim.crs, source=cx.providers.CartoDB.Positron, alpha=0.7)
                except Exception as e: print(f"Basemap failed: {e}")
                time_str = f"{hour:02d}:00 - {hour+1:02d}:00"; ax_anim.set_title(f"UAS Deliveries - {date.strftime('%a, %b %d')} | {time_str}", fontsize=16)
                ax_anim.text(0.02, 0.02, f"Deliveries: {len(hour_deliveries)}", transform=ax_anim.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
                ax_anim.set_axis_off(); ax_anim.set_aspect('equal', adjustable='box')
                frame_path = os.path.join(frames_dir, f"day{(date-sim_start).days}_hour_{hour:02d}.png"); plt.savefig(frame_path, dpi=100, bbox_inches='tight'); plt.close(fig_anim); frame_count += 1
        print(f"Generated {frame_count} animation frames.")
    else: print("Skipping animation frame generation.")

    print("Saving Module E outputs...")
    cols_to_save_e = [col for col in buildings_with_demand.columns if col not in ['knn_details', 'index_right', 'scale_factor']] # Clean up more columns
    buildings_save_e = buildings_with_demand[cols_to_save_e].copy()
    for col in buildings_save_e.select_dtypes(include=['object']).columns:
        mask = buildings_save_e[col].apply(lambda x: isinstance(x, (list, dict)))
        if mask.any(): buildings_save_e.loc[mask, col] = buildings_save_e.loc[mask, col].astype(str)
    buildings_save_e.to_file(os.path.join(data_subdir, "buildings_with_demand.geojson"), driver="GeoJSON")
    if not delivery_events_df.empty: delivery_events_df.to_csv(os.path.join(data_subdir, "delivery_events.csv"), index=False)
    else: save_empty_df(os.path.join(data_subdir, "delivery_events.csv"), columns=['building_unique_id', 'timestamp', 'latitude', 'longitude'])
    with open(os.path.join(config_subdir, "demand_parameters.yaml"), 'w') as f: yaml.dump(config['demand_model'], f)

    print("Generating demand visualization...")
    fig_e, ax_e = plt.subplots(1, 2, figsize=(18, 8)); plot_data_e = buildings_with_demand.to_crs(epsg=3857); union_gdf_proj_e = union_gdf_e.to_crs(epsg=3857)
    plot_data_e = gpd.clip(plot_data_e, union_gdf_proj_e)
    res_plot_e = plot_data_e[plot_data_e['residential'] == 'yes']; nonres_plot_e = plot_data_e[plot_data_e['residential'] == 'no']
    nonres_plot_e.plot(ax=ax_e[0], color='lightgray', alpha=0.5, label='Non-Residential')
    if not res_plot_e.empty and res_plot_e['simulated_total_deliveries'].sum() > 0:
         cmap_dem = 'YlOrRd'; vmax_dem = res_plot_e['simulated_total_deliveries'].quantile(0.98); norm_dem = Normalize(vmin=0, vmax=vmax_dem if vmax_dem > 0 else 1)
         res_plot_e.plot(column='simulated_total_deliveries', cmap=cmap_dem, norm=norm_dem, legend=False, ax=ax_e[0], alpha=0.8)
         sm_dem = ScalarMappable(cmap=cmap_dem, norm=norm_dem); sm_dem.set_array([]); cbar_dem = fig_e.colorbar(sm_dem, ax=ax_e[0], shrink=0.7); cbar_dem.set_label(f"Simulated Deliveries ({dem_config['simulation_duration_days']} days)")
    else: ax_e[0].text(0.5, 0.5, "No residential demand", transform=ax_e[0].transAxes, ha='center')
    try: cx.add_basemap(ax_e[0], crs='epsg:3857', source=cx.providers.CartoDB.Positron)
    except Exception as e: print(f"Basemap failed: {e}")
    ax_e[0].set_title('Simulated Delivery Demand (Union Area)'); ax_e[0].set_axis_off(); ax_e[0].set_aspect('equal', adjustable='box')
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
    print("Skipping Module E."); save_empty_gdf(os.path.join(data_subdir, "buildings_with_demand.geojson"))
    save_empty_df(os.path.join(data_subdir, "delivery_events.csv"), columns=['building_unique_id', 'timestamp', 'latitude', 'longitude'])
    save_empty_df(os.path.join(stats_subdir, 'demand_stats.csv'), columns=['Value'])
    if not os.path.exists(config_subdir): os.makedirs(config_subdir);
    with open(os.path.join(config_subdir, "demand_parameters.yaml"), 'w') as f: yaml.dump({}, f)

module_E_time = time.time() - module_E_start
print(f"Module E completed in {module_E_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module F: Origin-Destination Matching & Dataset Generation
# -----------------------------------------------------------------------------
print("\n## Module F: Origin-Destination Matching & Dataset Generation...")
module_F_start = time.time()
matched_gdf = gpd.GeoDataFrame()
final_csv = pd.DataFrame()
data_subdir = os.path.join(output_dir, "data"); viz_subdir = os.path.join(output_dir, "visualizations"); stats_subdir = os.path.join(output_dir, "stats")
try:
    stores_gdf_f = gpd.read_file(os.path.join(data_subdir, "stores.geojson"))
    delivery_events_f = pd.read_csv(os.path.join(data_subdir, "delivery_events.csv"))
    # Load buildings_with_pop as it has unique_id needed later, demand rate not strictly needed here
    buildings_gdf_f = gpd.read_file(os.path.join(data_subdir, "buildings_with_population.geojson"))
    if stores_gdf_f.empty or delivery_events_f.empty or buildings_gdf_f.empty: raise ValueError("Input data empty.")
    # Ensure unique ID
    if 'unique_id' not in buildings_gdf_f.columns:
        if 'element_type' in buildings_gdf_f.columns and 'osmid' in buildings_gdf_f.columns: buildings_gdf_f['unique_id'] = buildings_gdf_f['element_type'] + buildings_gdf_f['osmid'].astype(str)
        elif 'osmid' in buildings_gdf_f.columns: buildings_gdf_f['unique_id'] = 'way' + buildings_gdf_f['osmid'].astype(str)
        else: buildings_gdf_f = buildings_gdf_f.reset_index().rename(columns={'index':'unique_id'}); buildings_gdf_f['unique_id'] = 'bldg_' + buildings_gdf_f['unique_id'].astype(str)
except (FileNotFoundError, ValueError, Exception) as e: print(f"ERROR loading data for O-D matching: {e}. Skipping Module F.")

if not stores_gdf_f.empty and not delivery_events_f.empty and not buildings_gdf_f.empty:
    delivery_events_f['timestamp'] = pd.to_datetime(delivery_events_f['timestamp'])
    match_config = config['origin_destination_matching']; match_hour = match_config['simulation_hour_for_matching']
    sim_start_date_f = datetime.strptime(config['demand_model']['simulation_start_date'], '%Y-%m-%d').date()
    hour_start = datetime.combine(sim_start_date_f, datetime.min.time()) + timedelta(hours=match_hour); hour_end = hour_start + timedelta(hours=1)
    deliveries_for_matching = delivery_events_f[(delivery_events_f['timestamp'] >= hour_start) & (delivery_events_f['timestamp'] < hour_end)].copy()
    print(f"Matching {len(deliveries_for_matching)} deliveries for hour {match_hour}:00...")
    od_config = config['origin_destination_matching']; match_method = od_config['method']
    stores_utm = stores_gdf_f.to_crs(utm_crs)
    if 'store_id' not in stores_utm.columns: stores_utm = stores_utm.reset_index().rename(columns={'index':'store_id'}); stores_utm['store_id'] = 'Store_' + stores_utm['store_id'].astype(str)
    deliveries_gdf = gpd.GeoDataFrame(deliveries_for_matching, geometry=gpd.points_from_xy(deliveries_for_matching['longitude'], deliveries_for_matching['latitude']), crs=WGS84).to_crs(utm_crs)
    matched_gdf = None
    if deliveries_gdf.empty: print("No deliveries to match.")
    elif match_method == 'Proximity-Based':
        print("Using Proximity-Based matching...")
        deliveries_gdf = deliveries_gdf.reset_index(drop=True); stores_utm = stores_utm.reset_index(drop=True)
        # Ensure stores_utm has valid geometries before join
        stores_utm_valid = stores_utm[stores_utm.geometry.is_valid & ~stores_utm.geometry.is_empty]
        matched_gdf = gpd.sjoin_nearest(deliveries_gdf, stores_utm_valid, how='left', distance_col="dist_to_store")
        matched_gdf = matched_gdf.rename(columns={'store_id_right':'origin_id', 'name_right':'origin_name', 'store_type_right':'store_type', 'geometry_right': 'origin_geometry'})
        if 'index_right' in matched_gdf.columns: matched_gdf = matched_gdf.drop(columns=['index_right'])
# [Previous code in Module F remains the same]
elif match_method == 'Market Share Weighted':
    print("Using Market Share Weighted matching...")
    shares = od_config.get('market_shares', {})
    stores_utm_typed = stores_utm[stores_utm['store_type'].notna()] # Ensure no NaN types
    available_types = stores_utm_typed['store_type'].unique()
    valid_shares = {st: sh for st, sh in shares.items() if st in available_types and pd.notna(st) and isinstance(sh, (int,float)) and sh > 0}

    if not valid_shares:
        print("ERROR: No market shares match available store types or shares are invalid. Falling back to random.")
        match_method = 'Random' # Fallback will be handled later
    else:
        total_share = sum(valid_shares.values())
        if not math.isclose(total_share, 1.0, rel_tol=1e-5):
            print(f"Warning: Market shares sum to {total_share:.4f}. Normalizing.")
            norm_shares = {st: sh / total_share for st, sh in valid_shares.items()}
        else:
            norm_shares = valid_shares

        store_types_for_choice = list(norm_shares.keys()); probabilities = list(norm_shares.values())
        deliveries_gdf['assigned_store_type'] = np.random.choice(store_types_for_choice, size=len(deliveries_gdf), p=probabilities)

        matched_list = []
        stores_by_type = {stype: sdf[sdf.geometry.is_valid & ~sdf.geometry.is_empty] for stype, sdf in stores_utm_typed.groupby('store_type')} # Group valid geometries

        for idx, delivery in deliveries_gdf.iterrows():
            assigned_type = delivery['assigned_store_type']
            chosen_origin = None

            # Check 1: Is delivery geometry valid?
            if not (delivery.geometry and delivery.geometry.is_valid and not delivery.geometry.is_empty):
                print(f"Warning: Invalid delivery geometry for index {idx}. Falling back.")
                # Fallback logic (nearest overall valid store)
                valid_stores = stores_utm_typed[stores_utm_typed.geometry.is_valid & ~stores_utm_typed.geometry.is_empty]
                if not valid_stores.empty:
                    # Need a valid point to calculate distance *from*
                    # If delivery geom is invalid, use a default point (e.g., centroid of all stores)
                    fallback_point = valid_stores.unary_union.centroid if not valid_stores.empty else delivery.geometry # Last resort
                    if fallback_point and fallback_point.is_valid:
                         distances = valid_stores.geometry.distance(fallback_point)
                         if not distances.empty and not distances.isna().all():
                              min_dist_idx_label = distances.idxmin()
                              if min_dist_idx_label in valid_stores.index:
                                   chosen_origin = valid_stores.loc[min_dist_idx_label]
                # If still no chosen_origin, assign the first valid store
                if chosen_origin is None and not valid_stores.empty:
                     chosen_origin = valid_stores.iloc[0]

            # Check 2: Is assigned type valid and are there valid stores of that type?
            elif pd.isna(assigned_type) or assigned_type not in stores_by_type or stores_by_type[assigned_type].empty:
                print(f"Warning: Invalid assigned type '{assigned_type}' or no valid stores of this type for delivery index {idx}. Falling back.")
                # Fallback logic (nearest overall valid store)
                valid_stores = stores_utm_typed[stores_utm_typed.geometry.is_valid & ~stores_utm_typed.geometry.is_empty]
                if not valid_stores.empty:
                    distances = valid_stores.geometry.distance(delivery.geometry)
                    if not distances.empty and not distances.isna().all():
                        min_dist_idx_label = distances.idxmin()
                        if min_dist_idx_label in valid_stores.index:
                            chosen_origin = valid_stores.loc[min_dist_idx_label]
                if chosen_origin is None and not valid_stores.empty:
                    chosen_origin = valid_stores.iloc[0]

            # Check 3: Calculate distances to stores of the assigned type
            else:
                possible_origins = stores_by_type[assigned_type] # Already filtered for valid geometry
                distances = possible_origins.geometry.distance(delivery.geometry)

                # Check 4: Are calculated distances valid?
                if distances.empty or distances.isna().all():
                    print(f"Warning: Could not calculate valid distances to stores of type '{assigned_type}' for delivery {idx}. Falling back.")
                    # Fallback logic (nearest overall valid store)
                    valid_stores = stores_utm_typed[stores_utm_typed.geometry.is_valid & ~stores_utm_typed.geometry.is_empty]
                    if not valid_stores.empty:
                        fallback_distances = valid_stores.geometry.distance(delivery.geometry)
                        if not fallback_distances.empty and not fallback_distances.isna().all():
                             min_dist_idx_label = fallback_distances.idxmin()
                             if min_dist_idx_label in valid_stores.index:
                                 chosen_origin = valid_stores.loc[min_dist_idx_label]
                    if chosen_origin is None and not valid_stores.empty:
                         chosen_origin = valid_stores.iloc[0]
                else:
                    # Proceed with finding the nearest among possible_origins
                    min_dist_idx_label = distances.idxmin()
                    if min_dist_idx_label in possible_origins.index:
                        chosen_origin = possible_origins.loc[min_dist_idx_label]
                    else:
                        print(f"Warning: idxmin label {min_dist_idx_label} not found in possible_origins index for delivery {idx}. Falling back.")
                        # Fallback logic (nearest overall valid store)
                        valid_stores = stores_utm_typed[stores_utm_typed.geometry.is_valid & ~stores_utm_typed.geometry.is_empty]
                        if not valid_stores.empty:
                             fallback_distances = valid_stores.geometry.distance(delivery.geometry)
                             if not fallback_distances.empty and not fallback_distances.isna().all():
                                  min_dist_idx_label_fb = fallback_distances.idxmin()
                                  if min_dist_idx_label_fb in valid_stores.index:
                                       chosen_origin = valid_stores.loc[min_dist_idx_label_fb]
                        if chosen_origin is None and not valid_stores.empty:
                             chosen_origin = valid_stores.iloc[0]

            # Final check if we still failed to assign an origin
            if chosen_origin is None:
                 print(f"ERROR: Could not assign any valid origin for delivery {idx}. Skipping.")
                 continue # Skip this delivery

            # Append matched data
            delivery_dict = delivery.to_dict();
            delivery_dict['origin_id'] = chosen_origin['store_id']
            delivery_dict['origin_name'] = chosen_origin['name']
            delivery_dict['store_type'] = chosen_origin['store_type']
            delivery_dict['origin_geometry'] = chosen_origin['geometry']
            matched_list.append(delivery_dict)

        if matched_list: # Check if any deliveries were successfully matched
            matched_gdf = gpd.GeoDataFrame(matched_list, crs=utm_crs)
        else:
            print("No deliveries could be matched using Market Share Weighted method.")
            matched_gdf = gpd.GeoDataFrame(columns=deliveries_gdf.columns.tolist() + ['origin_id', 'origin_name', 'store_type', 'origin_geometry'], crs=utm_crs) # Ensure empty gdf has columns

# ... [Rest of Module F code remains the same] ...


    if match_method == 'Random' or matched_gdf is None:
        if not deliveries_gdf.empty:
            print("Using Random matching..."); num_deliveries = len(deliveries_gdf); random_store_indices = np.random.choice(stores_utm.index, size=num_deliveries)
            chosen_origins = stores_utm.loc[random_store_indices].reset_index(drop=True)
            matched_gdf = deliveries_gdf.reset_index(drop=True).join(chosen_origins.add_suffix('_origin').rename(columns={'store_id_origin':'origin_id', 'name_origin':'origin_name', 'geometry_origin':'origin_geometry', 'store_type_origin':'store_type'}))
        else: matched_gdf = gpd.GeoDataFrame(columns=['building_unique_id', 'timestamp', 'latitude', 'longitude', 'geometry', 'origin_id', 'origin_name', 'store_type', 'origin_geometry'], crs=utm_crs)

    # --- Calculate Distance & Format Output ---
    if matched_gdf is not None and not matched_gdf.empty:
        print("Calculating distances and formatting output...")
        if 'origin_geometry' not in matched_gdf.columns: matched_gdf = matched_gdf.merge(stores_utm[['store_id', 'geometry']].rename(columns={'store_id':'origin_id', 'geometry':'origin_geometry'}), on='origin_id', how='left')
        # Recalculate distance with checks
        matched_gdf['straight_line_distance_m'] = matched_gdf.apply(
            lambda row: row.geometry.distance(row['origin_geometry'])
            if isinstance(row.geometry, Point) and isinstance(row.get('origin_geometry'), Point) and not row.geometry.is_empty and not row.get('origin_geometry').is_empty and row.get('origin_geometry').is_valid and row.geometry.is_valid
            else np.nan, axis=1 )

        final_output = matched_gdf.copy(); final_output['timestamp_unix'] = final_output['timestamp'].apply(lambda dt: int(dt.timestamp()))
        final_output_wgs84 = final_output.to_crs(WGS84)
        def format_coords(point_geom):
            if pd.notna(point_geom) and isinstance(point_geom, Point) and not point_geom.is_empty: return f"[{point_geom.y:.6f}, {point_geom.x:.6f}]"
            return None
        if 'origin_geometry' in final_output_wgs84.columns:
             origin_geom_series = gpd.GeoSeries(final_output_wgs84['origin_geometry'], crs=final_output_wgs84.crs)
             final_output['origin_coordinates'] = origin_geom_series.apply(format_coords)
        else: final_output['origin_coordinates'] = None
        destination_geom_series = gpd.GeoSeries(final_output_wgs84['geometry'], crs=final_output_wgs84.crs); final_output['destination_coordinates'] = destination_geom_series.apply(format_coords)
        final_output.insert(0, 'order_id', range(1, len(final_output) + 1))
        dest_id_col = 'building_unique_id' if 'building_unique_id' in final_output.columns else 'unique_id'
        final_csv = final_output[['order_id', 'timestamp_unix', 'origin_id', 'origin_coordinates', dest_id_col, 'destination_coordinates']].rename(columns={'timestamp_unix': 'timestamp', dest_id_col: 'destination_id'})
        print("Saving Module F outputs...")
        final_csv.to_csv(os.path.join(data_subdir, "routing_dataset.csv"), index=False)
        cols_to_save_f = [col for col in matched_gdf.columns if col != 'timestamp' and col != 'origin_geometry' and col not in ['assigned_store_type']] # Clean temp columns
        matched_gdf_save = matched_gdf[cols_to_save_f].copy()
        # Add line geometry for visualization
        matched_gdf_save['geometry'] = matched_gdf.apply(lambda row: LineString([row['origin_geometry'], row.geometry]) if row.get('origin_geometry') and row.geometry and row.get('origin_geometry').is_valid and row.geometry.is_valid else None, axis=1)
        matched_gdf_save = gpd.GeoDataFrame(matched_gdf_save, geometry='geometry', crs=utm_crs)
        for col in matched_gdf_save.select_dtypes(include=['object']).columns:
            mask = matched_gdf_save[col].apply(lambda x: isinstance(x, (list, dict)))
            if mask.any(): matched_gdf_save.loc[mask, col] = matched_gdf_save.loc[mask, col].astype(str)
        matched_gdf_save.to_file(os.path.join(data_subdir, "routing_dataset_detailed.geojson"), driver="GeoJSON")
        print("Generating O-D visualization...")
        fig_f, ax_f = plt.subplots(figsize=(12, 12)); buildings_plot_f = buildings_gdf_f.to_crs(epsg=3857); selected_gdf_proj = selected_gdf.to_crs(epsg=3857)
        buildings_plot_f = gpd.clip(buildings_plot_f, selected_gdf_proj); stores_plot_f = stores_utm.to_crs(epsg=3857); routes_plot_f = matched_gdf_save.to_crs(epsg=3857)
        buildings_plot_f.plot(ax=ax_f, color='lightgray', alpha=0.3)
        if not stores_plot_f.empty: stores_plot_f.plot(ax=ax_f, color='red', marker='*', markersize=150, label='Stores', zorder=3)
        num_to_plot = min(500, len(routes_plot_f)); sample_routes = routes_plot_f.sample(num_to_plot, random_state=seed) if num_to_plot < len(routes_plot_f) else routes_plot_f
        if not sample_routes.empty: sample_routes.plot(ax=ax_f, color='black', linewidth=0.5, alpha=0.1, zorder=2)
        try: cx.add_basemap(ax_f, crs='epsg:3857', source=cx.providers.CartoDB.Positron)
        except Exception as e: print(f"Basemap failed: {e}")
        ax_f.set_title(f'Origin-Destination Matches (Hour {match_hour}:00, Sampled)'); ax_f.set_axis_off(); ax_f.legend(); ax_f.set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(viz_subdir, 'od_map.png'), dpi=150, bbox_inches='tight'); plt.close(fig_f)
        stats_f = {"Deliveries Matched": len(matched_gdf), "Matching Method": match_method,}
        if 'store_type' in matched_gdf.columns: stats_f.update(matched_gdf['store_type'].value_counts().rename(lambda x: f"Deliveries from {x}").to_dict())
        if 'straight_line_distance_m' in matched_gdf.columns: stats_f['Avg Distance (m)'] = matched_gdf['straight_line_distance_m'].mean(); stats_f['Median Distance (m)'] = matched_gdf['straight_line_distance_m'].median()
        stats_f_df = pd.DataFrame.from_dict(stats_f, orient='index', columns=['Value']); stats_f_df.to_csv(os.path.join(stats_subdir, 'od_stats.csv'))
        print("Module F Stats:"); print(stats_f_df)
    else: print("No deliveries matched, skipping outputs."); save_empty_df(os.path.join(data_subdir, "routing_dataset.csv")); save_empty_gdf(os.path.join(data_subdir, "routing_dataset_detailed.geojson")); save_empty_df(os.path.join(stats_subdir, 'od_stats.csv'))
else:
    print("Skipping Module F because input data is empty.")
    save_empty_df(os.path.join(data_subdir, "routing_dataset.csv"), columns=['order_id', 'timestamp', 'origin_id', 'origin_coordinates', 'destination_id', 'destination_coordinates'])
    save_empty_gdf(os.path.join(data_subdir, "routing_dataset_detailed.geojson"))
    save_empty_df(os.path.join(stats_subdir, 'od_stats.csv'), columns=['Value'])
module_F_time = time.time() - module_F_start
print(f"Module F completed in {module_F_time:.2f} seconds.")

# -----------------------------------------------------------------------------
# Module G: Reporting
# -----------------------------------------------------------------------------
# ... [Code as in v16] ...
print("\n## Module G: Reporting...")
module_G_start = time.time()
report_path = os.path.join(output_dir, "summary_report.txt")
stats_subdir = os.path.join(output_dir, "stats"); viz_subdir = os.path.join(output_dir, "visualizations")
try:
    with open(report_path, 'w') as f:
        f.write("UAS Delivery Demand Simulation Report\n"); f.write("="*40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write(f"Seed: {config['random_seed']}\n\n")
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

# --- Create GIF ---
try:
    print("\nAttempting to create GIF...")
    frames_dir = os.path.join(output_dir, 'animation_frames')
    frames = []; frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if frame_files:
        print(f"Found {len(frame_files)} frames for GIF...")
        for frame_file in frame_files:
            try: frames.append(Image.open(frame_file))
            except Exception as img_e: print(f"Could not open frame {frame_file}: {img_e}")
        if frames:
            gif_path = os.path.join(viz_subdir, 'delivery_animation.gif')
            frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0, optimize=True)
            print(f"Created animated GIF: {gif_path}")
        else: print("No valid frames could be opened.")
    else: print("No PNG frames found in animation_frames directory.")
except ImportError: print("Pillow library not found. Skipping GIF creation.")
except Exception as gif_e: print(f"Error creating GIF: {gif_e}")

module_G_time = time.time() - module_G_start
print(f"Module G completed in {module_G_time:.2f} seconds.")
total_time = time.time() - module_A_start
print(f"\nTotal execution time: {total_time:.2f} seconds.")
print("\nProcessing Complete. Check the output directory structure for results.")

# --- Display Final GIF in Colab ---
gif_path_final = os.path.join(output_dir, 'visualizations', 'delivery_animation.gif')
if os.path.exists(gif_path_final):
    from IPython.display import Image as IPImage, display
    print("\nDisplaying final animation GIF:")
    display(IPImage(filename=gif_path_final))
else:
    print("\nAnimation GIF not found or not created.")