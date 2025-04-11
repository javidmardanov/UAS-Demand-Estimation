# modules/area_selection.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import tempfile
import shutil
import glob
import time
import math
import matplotlib.pyplot as plt
import contextily as cx
import osmnx as ox
from shapely.geometry import box, Point, MultiPolygon, Polygon
from shapely.ops import unary_union
import streamlit as st # Use streamlit for status updates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import utilities - Use try-except for robustness during development
try:
    from utils.crs import estimate_utm_crs
    from utils.helpers import save_empty_gdf, save_empty_df, style_map
except ImportError:
    st.error("Failed to import utility functions. Ensure utils/crs.py and utils/helpers.py exist.")
    # Define dummy functions to prevent crashing if imports fail
    def estimate_utm_crs(gdf): return 'EPSG:3857'
    def save_empty_gdf(f, **kwargs): pass
    def save_empty_df(f, **kwargs): pass
    def style_map(ax, title): pass


WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857' # Standard CRS for web maps and contextily

# --- Helper function for safe OSMnx fetch ---
def safe_ox_features(polygon, tags):
    """Safely fetches features from OSMnx within a given polygon.

    Args:
        polygon (shapely.geometry.Polygon): The polygon boundary.
        tags (dict): Dictionary of OSM tags to filter by.

    Returns:
        gpd.GeoDataFrame: Fetched features or an empty GeoDataFrame on error.
    """
    try:
        st.write(f"  Fetching OSM tags: {list(tags.keys())}...")
        features = ox.features_from_polygon(polygon, tags)
        st.write(f"  -> Fetched {len(features)} raw OSM features.")
        return features
    except Exception as e:
        st.warning(f"  OSMnx fetch failed. Tags: {list(tags.keys())}. Error: {e}")
        return gpd.GeoDataFrame() 

# --- Main Module Function ---
# @st.cache_data(ttl=3600) # Cache results for an hour - Disable for now during dev
def run_module_a(config):
    """Executes Module A: Area Selection & Data Acquisition.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing results:
              'selected_gdf', 'tracts_gdf', 'union_gdf', 'buffered_gdf',
              'osm_gdf', 'census_data_df', 'utm_crs', 'stats_a_df',
              'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module A: Area Selection & Data Acquisition")
    st.markdown("--- *Initializing* ---") # Progress indicator
    module_A_start = time.time()
    sel_config = config['area_selection']
    acq_config = config['data_acquisition']
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")

    # Create subdirectories if they don't exist
    for subdir in [data_subdir, viz_subdir, stats_subdir]:
        os.makedirs(subdir, exist_ok=True)

    # Initialize return values
    selected_gdf = gpd.GeoDataFrame()
    tracts_gdf = gpd.GeoDataFrame(columns=['GEOID', 'geometry'], geometry='geometry', crs=WGS84)
    union_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=WGS84)
    buffered_gdf = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=WGS84)
    osm_gdf = gpd.GeoDataFrame()
    census_data_df = pd.DataFrame()
    stats_a_df = pd.DataFrame()
    utm_crs = None
    area_selected_km2 = 0
    area_union_km2 = 0
    area_buffered_km2 = 0
    status_messages = []
    module_error = False

    # --- 1. Area Selection --- 
    st.markdown("--- *Processing Area Selection* ---")
    selected_polygon = None
    try:
        if sel_config['method'] == 'coordinates':
            coords = sel_config['coordinates']
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError(f"Invalid coordinates format: {coords}. Expected [lat_min, lon_min, lat_max, lon_max]")
            lat_min, lon_min, lat_max, lon_max = coords
            # Validate coordinates
            if not all(isinstance(c, (int, float)) for c in coords):
                 raise ValueError(f"Coordinates must be numeric: {coords}")
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90 and -180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                 raise ValueError(f"Coordinates out of bounds: {coords}")
            if lat_min >= lat_max or lon_min >= lon_max:
                raise ValueError(f"Invalid bounds: Min must be less than Max. Lat: {lat_min} >= {lat_max} or Lon: {lon_min} >= {lon_max}")

            selected_polygon = box(lon_min, lat_min, lon_max, lat_max)
            status_messages.append(f"Created selection box: [{lat_min:.5f}, {lon_min:.5f}, {lat_max:.5f}, {lon_max:.5f}]")
        # TODO: Add other selection methods later (e.g., place name)
        else:
            raise ValueError(f"Area selection method '{sel_config['method']}' not supported.")

        selected_gdf = gpd.GeoDataFrame(geometry=[selected_polygon], crs=WGS84)
        status_messages.append("Created selected_gdf.")

        # Estimate UTM CRS
        status_messages.append("Estimating UTM CRS...")
        utm_crs = estimate_utm_crs(selected_gdf)
        status_messages.append(f"-> Estimated UTM CRS: {utm_crs}")
        if utm_crs:
            selected_gdf_proj = selected_gdf.to_crs(utm_crs)
            area_selected_km2 = selected_gdf_proj.geometry.iloc[0].area / 1e6
            status_messages.append(f"Selected Area Size: {area_selected_km2:.2f} km²")
        else:
            st.warning("Could not estimate UTM CRS, area calculation skipped.")
            area_selected_km2 = 0

        # Save initial selection
        try:
            sel_file = os.path.join(data_subdir, "area_selection.geojson")
            selected_gdf.to_file(sel_file, driver="GeoJSON")
            status_messages.append(f"Saved: {os.path.basename(sel_file)}")
        except Exception as e_save:
             st.warning(f"Could not save area_selection.geojson: {e_save}")

    except ValueError as e:
        st.error(f"Configuration Error (Area Selection): {e}")
        module_error = True
    except Exception as e:
        st.error(f"Runtime Error (Area Selection): {e}")
        module_error = True

    # Return early if critical error in selection
    if module_error:
        st.error("Module A failed during Area Selection.")
        return {
            "selected_gdf": selected_gdf, "tracts_gdf": tracts_gdf, "union_gdf": union_gdf,
            "buffered_gdf": buffered_gdf, "osm_gdf": osm_gdf, "census_data_df": census_data_df,
            "utm_crs": utm_crs, "stats_a_df": stats_a_df, "status_messages": status_messages,
            "error": True
        }

    # --- 2. Fetch Census Tracts --- 
    if not module_error:
        st.markdown("--- *Fetching Census Tracts* ---")
        temp_dir = None
        try:
            state = acq_config['state_code']
            county = acq_config['county_code']
            tract_year = acq_config['tract_year']
            tract_url = f"https://www2.census.gov/geo/tiger/TIGER{tract_year}/TRACT/tl_{tract_year}_{state}_tract.zip"
            status_messages.append(f"Downloading Census tracts ({tract_year}) for state {state}...")
            st.write(f"Downloading: {tract_url}")

            response = requests.get(tract_url, stream=True, timeout=120)
            response.raise_for_status()

            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, f"tracts.zip")
            extracted_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extracted_dir, exist_ok=True)

            with open(zip_path, 'wb') as f: f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref: # Renamed 'z' to 'zip_ref'
                zip_ref.extractall(extracted_dir)
            status_messages.append("Downloaded and extracted tract archive.")

            shp_files = glob.glob(os.path.join(extracted_dir, '*.shp'))
            if not shp_files:
                raise FileNotFoundError("No .shp file found in downloaded Census tract archive.")

            all_tracts_in_state = gpd.read_file(shp_files[0])
            status_messages.append(f"Read {len(all_tracts_in_state)} tracts from shapefile.")

            # Ensure WGS84 CRS
            if all_tracts_in_state.crs is None:
                st.warning("Tract shapefile missing CRS, assuming WGS84.")
                all_tracts_in_state.crs = WGS84
            elif all_tracts_in_state.crs != WGS84:
                all_tracts_in_state = all_tracts_in_state.to_crs(WGS84)

            # Filter by county and validity
            county_tracts = all_tracts_in_state[all_tracts_in_state['COUNTYFP'] == county].copy()
            county_tracts = county_tracts[county_tracts.geometry.is_valid]
            status_messages.append(f"Filtered to {len(county_tracts)} valid tracts in county {county}.")

            # Find intersecting tracts
            if not county_tracts.empty and selected_polygon is not None:
                 # Perform intersection check with valid geometry
                 intersecting_mask = county_tracts.geometry.intersects(selected_polygon)
                 tracts_gdf = county_tracts[intersecting_mask].copy()
            else:
                 tracts_gdf = gpd.GeoDataFrame(columns=['GEOID', 'geometry'], geometry='geometry', crs=WGS84)

            # Standardize GEOID column
            if not tracts_gdf.empty:
                geoid_col = next((col for col in ['GEOID', f'GEOID{str(tract_year)[-2:]}', 'GEOID20', 'GEOID10'] if col in tracts_gdf.columns), None)
                if geoid_col and geoid_col != 'GEOID':
                    tracts_gdf.rename(columns={geoid_col: 'GEOID'}, inplace=True)
                elif 'GEOID' not in tracts_gdf.columns and all({'STATEFP', 'COUNTYFP', 'TRACTCE'}) <= set(tracts_gdf.columns):
                    tracts_gdf['GEOID'] = tracts_gdf['STATEFP'] + tracts_gdf['COUNTYFP'] + tracts_gdf['TRACTCE']

                if 'GEOID' not in tracts_gdf.columns:
                    st.warning("Could not find or construct GEOID column for tracts. Some features might break.")
                else:
                     tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str)

            status_messages.append(f"Found {len(tracts_gdf)} tracts intersecting the selected area.")

            # Save tracts
            try:
                tracts_file = os.path.join(data_subdir, "census_tracts.geojson")
                if not tracts_gdf.empty:
                     tracts_gdf.to_file(tracts_file, driver="GeoJSON")
                else:
                     save_empty_gdf(tracts_file)
                status_messages.append(f"Saved: {os.path.basename(tracts_file)}")
            except Exception as e_save:
                st.warning(f"Could not save census_tracts.geojson: {e_save}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading Census tracts: {e}")
            module_error = True
        except FileNotFoundError as e:
            st.error(f"Error processing tract files: {e}")
            module_error = True
        except Exception as e:
            st.error(f"An unexpected error occurred fetching tracts: {e}")
            module_error = True
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    status_messages.append("Cleaned temporary tract directory.")
                except Exception as e_clean:
                    st.warning(f"Could not remove temp directory {temp_dir}: {e_clean}")

    # --- 3. Calculate Union and Buffer --- 
    if not module_error:
        st.markdown("--- *Calculating Union and Buffer Areas* ---")
        buffered_polygon = None
        if not tracts_gdf.empty and utm_crs:
            tracts_gdf = tracts_gdf[tracts_gdf.geometry.is_valid]
            if not tracts_gdf.empty:
                try:
                    union_of_tracts_poly = unary_union(tracts_gdf.geometry)
                    union_gdf = gpd.GeoDataFrame(geometry=[union_of_tracts_poly], crs=WGS84)
                    union_utm = union_gdf.to_crs(utm_crs)
                    area_union_km2 = union_utm.geometry.iloc[0].area / 1e6
                    status_messages.append(f"Created union of {len(tracts_gdf)} tracts. Area: {area_union_km2:.2f} km²")

                    buffer_m = sel_config['buffer_km'] * 1000
                    buffered_poly_utm = union_utm.geometry.iloc[0].buffer(buffer_m)
                    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84)
                    buffered_polygon = buffered_gdf.geometry.iloc[0]
                    area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
                    status_messages.append(f"Created buffered area ({sel_config['buffer_km']} km). Area: {area_buffered_km2:.2f} km²")

                except Exception as e:
                    st.warning(f"Error creating union/buffer from tracts: {e}. Using selection rectangle as fallback.")
                    # Fallback to using selection rectangle if union fails
                    union_gdf = selected_gdf.copy()
                    area_union_km2 = area_selected_km2
                    buffer_m = sel_config['buffer_km'] * 1000
                    selected_utm = selected_gdf.to_crs(utm_crs)
                    buffered_poly_utm = selected_utm.geometry.iloc[0].buffer(buffer_m)
                    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84)
                    buffered_polygon = buffered_gdf.geometry.iloc[0]
                    area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
                    status_messages.append(f"WARN: Used selection rectangle for buffer due to union error. Buffered Area: {area_buffered_km2:.2f} km²")
            else:
                st.warning("No valid tracts intersected the area. Using selection rectangle for buffer.")
                union_gdf = selected_gdf.copy() # Use selection as union
                area_union_km2 = area_selected_km2
                selected_utm = selected_gdf.to_crs(utm_crs)
                buffer_m = sel_config['buffer_km'] * 1000
                buffered_poly_utm = selected_utm.geometry.iloc[0].buffer(buffer_m)
                buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84)
                buffered_polygon = buffered_gdf.geometry.iloc[0]
                area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
                status_messages.append(f"Used selection rectangle for buffer. Buffered Area: {area_buffered_km2:.2f} km²")
        elif utm_crs:
            st.warning("No tracts found. Using selection rectangle for buffer.")
            union_gdf = selected_gdf.copy() # Use selection as union
            area_union_km2 = area_selected_km2
            selected_utm = selected_gdf.to_crs(utm_crs)
            buffer_m = sel_config['buffer_km'] * 1000
            buffered_poly_utm = selected_utm.geometry.iloc[0].buffer(buffer_m)
            buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_poly_utm], crs=utm_crs).to_crs(WGS84)
            buffered_polygon = buffered_gdf.geometry.iloc[0]
            area_buffered_km2 = buffered_gdf.to_crs(utm_crs).geometry.iloc[0].area / 1e6
            status_messages.append(f"Used selection rectangle for buffer. Buffered Area: {area_buffered_km2:.2f} km²")
        else:
             st.error("Cannot calculate buffer without a valid UTM CRS.")
             buffered_polygon = None
             module_error = True

        # Save Union and Buffer GDFs
        try:
            union_file = os.path.join(data_subdir, "union_of_tracts.geojson")
            if not union_gdf.empty:
                 union_gdf.to_file(union_file, driver="GeoJSON")
            else:
                 save_empty_gdf(union_file)
            status_messages.append(f"Saved: {os.path.basename(union_file)}")

            buffer_file = os.path.join(data_subdir, "buffered_area.geojson")
            if not buffered_gdf.empty:
                buffered_gdf.to_file(buffer_file, driver="GeoJSON")
            else:
                save_empty_gdf(buffer_file)
            status_messages.append(f"Saved: {os.path.basename(buffer_file)}")
        except Exception as e_save:
            st.warning(f"Could not save union/buffer geojson: {e_save}")

    # --- 4. Fetch OSM Data --- 
    if not module_error:
        st.markdown("--- *Fetching OSM Data (this may take a while...)* ---")
        if buffered_polygon is not None and not buffered_gdf.empty:
            try:
                osm_tags_fetch = {k: v for k, v in acq_config['osm_tags'].items() if v}
                if not osm_tags_fetch:
                    st.warning("No OSM tags selected in config. Skipping OSM fetch.")
                    osm_gdf = gpd.GeoDataFrame()
                else:
                    # Fetch using precise polygon
                    status_messages.append(f"Fetching OSM features within buffered area...")
                    fetched_osm = safe_ox_features(buffered_polygon, osm_tags_fetch)

                    if not fetched_osm.empty:
                        # Ensure CRS match before clipping (belt-and-braces)
                        if fetched_osm.crs != buffered_gdf.crs:
                            fetched_osm = fetched_osm.to_crs(buffered_gdf.crs)

                        # Clip precisely to the buffered polygon
                        osm_gdf = gpd.clip(fetched_osm, buffered_gdf)
                        status_messages.append(f"Clipped to {len(osm_gdf)} OSM features within buffer polygon.")

                        osm_gdf = osm_gdf.reset_index()

                        # Handle MultiIndex columns if they exist
                        if isinstance(osm_gdf.columns, pd.MultiIndex):
                            osm_gdf.columns = ['_'.join(map(str, col)).strip('_') for col in osm_gdf.columns.values]

                        # Standardize unique ID
                        if 'osmid' not in osm_gdf.columns and 'element_type_osmid' in osm_gdf.columns:
                             osm_gdf.rename(columns={'element_type_osmid': 'osmid'}, inplace=True)
                        elif 'osmid' not in osm_gdf.columns and 'id' in osm_gdf.columns:
                             osm_gdf.rename(columns={'id': 'osmid'}, inplace=True)

                        if 'element_type' in osm_gdf.columns and 'osmid' in osm_gdf.columns:
                             osm_gdf['unique_id'] = osm_gdf['element_type'].astype(str) + osm_gdf['osmid'].astype(str)
                        elif 'osmid' in osm_gdf.columns:
                             osm_gdf['unique_id'] = 'way' + osm_gdf['osmid'].astype(str)
                        else:
                             osm_gdf = osm_gdf.reset_index().rename(columns={'index':'generated_id'})
                             osm_gdf['unique_id'] = 'feat_' + osm_gdf['generated_id'].astype(str)
                             status_messages.append("WARN: Could not find osmid, generated unique IDs.")

                        # Clean geometry
                        osm_gdf = osm_gdf[~osm_gdf.geometry.isna()]
                        if not osm_gdf.empty:
                            osm_gdf['geometry'] = osm_gdf.geometry.buffer(0)
                            osm_gdf = osm_gdf[osm_gdf.geometry.is_valid]
                        status_messages.append(f"Retained {len(osm_gdf)} valid OSM features after cleaning.")
                    else:
                        status_messages.append("OSM fetch returned no features for the area.")
                        osm_gdf = gpd.GeoDataFrame()

                # Save OSM data
                try:
                    osm_file = os.path.join(data_subdir, "initial_osm_features.geojson")
                    if not osm_gdf.empty:
                         osm_gdf.to_file(osm_file, driver="GeoJSON")
                    else:
                         save_empty_gdf(osm_file)
                    status_messages.append(f"Saved: {os.path.basename(osm_file)}")
                except Exception as e_save:
                    st.warning(f"Could not save initial_osm_features.geojson: {e_save}")

            except Exception as e:
                st.error(f"Runtime Error (OSM Fetch): {e}")
                osm_gdf = gpd.GeoDataFrame() # Ensure empty on error
                # Don't set module_error=True, maybe user wants to proceed without OSM?
                status_messages.append("WARN: OSM data fetching failed.")
        else:
            status_messages.append("Skipping OSM fetch: Buffered area is not defined.")
            osm_gdf = gpd.GeoDataFrame()
            save_empty_gdf(os.path.join(data_subdir, "initial_osm_features.geojson"))

    # --- 5. Load and Process Local Census Demographic Data --- 
    census_data_df = pd.DataFrame() # Initialize empty
    if not module_error and not tracts_gdf.empty:
        st.markdown("--- *Loading and Processing Local Census Data* ---")
        local_csv_path = "us_census_tracts_2023.csv" # Assume file is in root
        tract_geoids_to_load = tracts_gdf['GEOID'].unique().tolist()

        if not os.path.exists(local_csv_path):
            st.error(f"Local Census CSV file not found: {local_csv_path}. Please provide the file.")
            status_messages.append(f"ERROR: Missing {local_csv_path}")
            module_error = True
            # Save empty df to avoid downstream errors
            save_empty_df(os.path.join(data_subdir, "census_data.csv"), columns=['GEOID'])
        else:
            st.write(f"Reading data from: {local_csv_path} (This might take time for large files)")
            try:
                # --- Read the local CSV --- 
                # Potential Optimization: Use chunking or dtype specification if memory becomes an issue
                # For now, read directly, assuming GEOID is string and others can be inferred
                # We need to identify the GEOID column correctly.
                # Let's try common names first.
                possible_geoid_cols = ['GEOID', 'geoid', 'Id', 'id', 'geo_id']
                
                # Read just the header first to find the GEOID column
                # This handles variations in user-provided CSV column naming conventions.
                try:
                    header_df = pd.read_csv(local_csv_path, nrows=0) # Read only header
                    actual_geoid_col = next((col for col in possible_geoid_cols if col in header_df.columns), None)
                    if not actual_geoid_col:
                         raise ValueError(f"Could not automatically find a GEOID column (tried: {possible_geoid_cols}) in {local_csv_path}. Please ensure the column exists and matches one of these names.")
                    status_messages.append(f"Identified GEOID column as: '{actual_geoid_col}'")
                except Exception as e_header:
                     st.error(f"Could not read header to find GEOID column from {local_csv_path}: {e_header}")
                     raise # Re-raise to stop processing

                # Read the full CSV, specifying GEOID column dtype as string
                census_data_full = pd.read_csv(local_csv_path, dtype={actual_geoid_col: str})
                status_messages.append(f"Read {len(census_data_full)} rows from {local_csv_path}.")

                # Rename the identified GEOID column to standard 'GEOID' if necessary
                if actual_geoid_col != 'GEOID':
                    census_data_full.rename(columns={actual_geoid_col: 'GEOID'}, inplace=True)
                    status_messages.append(f"Renamed column '{actual_geoid_col}' to 'GEOID'.")

                # --- Filter to required tracts --- 
                status_messages.append(f"Filtering Census data for {len(tract_geoids_to_load)} intersecting tracts...")
                census_data_filtered = census_data_full[census_data_full['GEOID'].isin(tract_geoids_to_load)].copy()
                status_messages.append(f"Filtered down to {len(census_data_filtered)} relevant Census rows.")
                del census_data_full # Free up memory

                if census_data_filtered.empty:
                    st.warning("No matching GEOIDs found in the local Census CSV for the intersecting tracts.")
                    census_data_df = pd.DataFrame(columns=['GEOID'])
                    save_empty_df(os.path.join(data_subdir, "census_data.csv"), columns=['GEOID'])
                else:
                    # --- Process the Filtered Data --- 
                    variables = acq_config['census_variables']
                    status_messages.append("Processing filtered Census data...")
                    st.write("Processing filtered Census data...")

                    # --- Check & Rename Required Variable Columns ---
                    renamed_columns = {}
                    missing_vars = []
                    current_columns = census_data_filtered.columns.tolist()

                    for req_var in variables:
                        found_col = None
                        # Check for exact match first
                        if req_var in current_columns:
                            found_col = req_var
                        else:
                            # Check if req_var is a substring in any column
                            for col in current_columns:
                                if req_var in col:
                                    found_col = col
                                    break # Found the first match
                        
                        if found_col:
                            if found_col != req_var:
                                # Only rename if the found column isn't the exact variable name
                                if req_var in current_columns:
                                    # Avoid renaming if target name already exists (e.g., from previous loop)
                                    st.warning(f"Column matching '{req_var}' found ('{found_col}'), but target name '{req_var}' already exists. Check CSV for duplicates or naming conflicts.")
                                else:
                                    renamed_columns[found_col] = req_var
                        else:
                            missing_vars.append(req_var)

                    if missing_vars:
                        st.error(f"Missing required Census variable codes in CSV column headers (or partial match not found): {missing_vars}. Cannot proceed.")
                        raise ValueError(f"Missing required Census variables: {missing_vars}")

                    # Perform the renaming
                    if renamed_columns:
                        census_data_filtered.rename(columns=renamed_columns, inplace=True)
                        status_messages.append(f"Renamed CSV columns based on required codes: {renamed_columns}")

                    # Now all required columns should exist with the standard names (req_var)
                    # Check again (optional, but safe)
                    final_missing = [v for v in variables if v not in census_data_filtered.columns]
                    if final_missing:
                         # This shouldn't happen if logic above is correct
                         st.error(f"Internal Error: Columns still missing after renaming attempt: {final_missing}")
                         raise ValueError(f"Internal Error: Columns still missing after rename: {final_missing}")

                    # Convert columns to numeric, coercing errors
                    for col in variables:
                        # Check if column exists before converting (should always exist now)
                        if col in census_data_filtered.columns:
                            census_data_filtered[col] = pd.to_numeric(census_data_filtered[col], errors='coerce')
                        else:
                            # Add warning if somehow missing despite checks
                            st.warning(f"Attempted to convert non-existent column '{col}' to numeric.")

                    # Calculate derived metrics
                    # Use .get() on the DataFrame columns object to check existence before calculation
                    if 'B01003_001E' in census_data_filtered.columns: census_data_filtered['total_population'] = census_data_filtered['B01003_001E']
                    else: census_data_filtered['total_population'] = 0; st.warning("Missing B01003_001E for total_population.")
                    
                    if 'B19013_001E' in census_data_filtered.columns: census_data_filtered['median_income'] = census_data_filtered['B19013_001E']
                    else: census_data_filtered['median_income'] = 0; st.warning("Missing B19013_001E for median_income.")
                    
                    if 'B25010_001E' in census_data_filtered.columns: census_data_filtered['avg_household_size'] = census_data_filtered['B25010_001E']
                    else: census_data_filtered['avg_household_size'] = 0; st.warning("Missing B25010_001E for avg_household_size.")
                    
                    if 'B25001_001E' in census_data_filtered.columns: census_data_filtered['total_housing_units'] = census_data_filtered['B25001_001E']
                    else: census_data_filtered['total_housing_units'] = 0; st.warning("Missing B25001_001E for total_housing_units.")

                    # Employment Rate
                    if 'B23025_002E' in census_data_filtered.columns and 'B23025_004E' in census_data_filtered.columns:
                        pop_in_labor_force = pd.to_numeric(census_data_filtered['B23025_002E'], errors='coerce').fillna(0).astype(float)
                        employed_pop = pd.to_numeric(census_data_filtered['B23025_004E'], errors='coerce').fillna(0).astype(float)
                        census_data_filtered['employment_rate'] = np.where(pop_in_labor_force > 0, (employed_pop / pop_in_labor_force) * 100.0, 0.0)
                    else: 
                        census_data_filtered['employment_rate'] = 0.0
                        st.warning("Missing B23025_002E or B23025_004E for employment_rate.")

                    # Population Density
                    if not tracts_gdf.empty and utm_crs:
                        tracts_with_area = tracts_gdf.to_crs(utm_crs)
                        tracts_with_area['area_sqkm'] = tracts_with_area.geometry.area / 1e6
                        # Merge area onto the filtered census data
                        census_data_processed = census_data_filtered.merge(tracts_with_area[['GEOID', 'area_sqkm']], on='GEOID', how='left')
                        area_sqkm_filled = census_data_processed['area_sqkm'].fillna(0.0).astype(float)
                        total_pop_filled = census_data_processed['total_population'].fillna(0.0).astype(float)
                        census_data_processed['pop_density'] = np.where(area_sqkm_filled > 0, total_pop_filled / area_sqkm_filled, 0.0)
                    else:
                        census_data_processed = census_data_filtered.copy()
                        census_data_processed['area_sqkm'] = 0.0
                        census_data_processed['pop_density'] = 0.0
                        status_messages.append("WARN: Could not calculate population density (missing tracts or UTM CRS).")

                    # Select and clean final columns
                    cols_to_keep = ['GEOID'] + variables + [
                        'total_population', 'median_income', 'avg_household_size',
                        'employment_rate', 'pop_density', 'total_housing_units', 'area_sqkm'
                    ]
                    existing_cols = [col for col in cols_to_keep if col in census_data_processed.columns]
                    census_data_df = census_data_processed[existing_cols].copy()
                    
                    # Final fillna for numeric columns
                    for col in census_data_df.columns:
                        if col != 'GEOID' and pd.api.types.is_numeric_dtype(census_data_df[col]):
                            census_data_df[col] = census_data_df[col].fillna(0)
                    
                    status_messages.append("Local Census data processing complete.")
                    st.write(f"Processed {len(census_data_df)} tracts with demographic data from local file.")
                    
                    # Save the processed (filtered) Census data
                    try:
                        census_file = os.path.join(data_subdir, "census_data.csv")
                        census_data_df.to_csv(census_file, index=False)
                        status_messages.append(f"Saved: {os.path.basename(census_file)}")
                    except Exception as e_save:
                        st.warning(f"Could not save processed census_data.csv: {e_save}")

            except ValueError as e:
                st.error(f"Error processing local Census CSV: {e}")
                module_error = True
                save_empty_df(os.path.join(data_subdir, "census_data.csv"), columns=['GEOID'])
            except Exception as e:
                st.error(f"An unexpected error occurred processing {local_csv_path}: {e}")
                module_error = True
                save_empty_df(os.path.join(data_subdir, "census_data.csv"), columns=['GEOID'])

    elif not module_error:
        status_messages.append("No intersecting tracts found, skipping Census data loading.")
        census_data_df = pd.DataFrame(columns=['GEOID'])
        save_empty_df(os.path.join(data_subdir, "census_data.csv"), columns=['GEOID'])

    # --- 6. Generate Visualizations --- 
    if not module_error:
        st.markdown("--- *Generating Setup Visualizations* ---")
        # Fig 1: Selection Rectangle
        try:
            st.write("Generating Fig 1: Selected Study Area...")
            fig1, ax1 = plt.subplots(figsize=(10, 10))
            if not selected_gdf.empty:
                selected_plot_gdf = selected_gdf.to_crs(WEB_MERCATOR)
                selected_plot_gdf.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=3, label='Selected Area')
            try:
                cx.add_basemap(ax1, crs=WEB_MERCATOR, source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap for Fig 1 failed: {e_base}")
            style_map(ax1, 'Figure 1: Selected Study Area')
            ax1.legend()
            fig1_path = os.path.join(viz_subdir, 'setup_map_1.png')
            plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
            plt.close(fig1)
            status_messages.append(f"Saved: {os.path.basename(fig1_path)}")
        except Exception as e_fig:
            st.warning(f"Could not generate/save Figure 1: {e_fig}")

        # Fig 2: Tracts/Union/BuildingsInUnion/Selection
        try:
            st.write("Generating Fig 2: Tracts, Union, Buildings & Selection...")
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            handles, labels = [], [] # For legend

            # Plot Tracts
            if not tracts_gdf.empty:
                tracts_plot_gdf = tracts_gdf.to_crs(WEB_MERCATOR)
                tracts_plot_gdf.plot(ax=ax2, facecolor='whitesmoke', edgecolor='gray', linewidth=0.5, label='Tracts', alpha=0.6)
                handles.append(Patch(facecolor='whitesmoke', edgecolor='gray', label='Intersecting Tracts'))

            # Plot Buildings in Union (if available)
            if not osm_gdf.empty and not union_gdf.empty:
                try:
                    # Use spatial index for faster clipping if many buildings
                    # osm_gdf.sindex
                    osm_buildings_in_union = gpd.clip(osm_gdf[osm_gdf['building'].notna()], union_gdf).to_crs(WEB_MERCATOR)
                    if not osm_buildings_in_union.empty:
                        osm_buildings_in_union.plot(ax=ax2, color='darkgray', alpha=0.7, linewidth=0.1)
                        handles.append(Patch(color='darkgray', label='Buildings in Union (from OSM)', alpha=0.7))
                except Exception as e_clip:
                    st.warning(f"Could not clip buildings for Fig 2: {e_clip}")

            # Plot Union Boundary
            if not union_gdf.empty:
                union_plot_gdf = union_gdf.to_crs(WEB_MERCATOR)
                union_plot_gdf.boundary.plot(ax=ax2, edgecolor='green', linewidth=2.5, label='Union of Tracts', zorder=4)
                handles.append(Line2D([0], [0], color='green', lw=2.5, label='Union of Tracts Boundary'))

            # Plot Selection Boundary
            if not selected_gdf.empty:
                selected_plot_gdf = selected_gdf.to_crs(WEB_MERCATOR)
                selected_plot_gdf.boundary.plot(ax=ax2, edgecolor='red', linewidth=1.5, linestyle='--', label='Selected Area', zorder=5)
                handles.append(Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='Selected Area Boundary'))

            try:
                cx.add_basemap(ax2, crs=WEB_MERCATOR, source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap for Fig 2 failed: {e_base}")
            style_map(ax2, 'Figure 2: Tracts, Union, Buildings & Selection')
            ax2.legend(handles=handles, loc='upper left', fontsize=9)
            fig2_path = os.path.join(viz_subdir, 'setup_map_2.png')
            plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            status_messages.append(f"Saved: {os.path.basename(fig2_path)}")
        except Exception as e_fig:
            st.warning(f"Could not generate/save Figure 2: {e_fig}")

        # Fig 3 (Initial) is typically done after classification to show stores
        # We can create a basic version here showing just buffer and buildings
        try:
            st.write("Generating Fig 3 (Initial): Buffer & All Buildings...")
            fig3, ax3 = plt.subplots(figsize=(10, 10))
            handles, labels = [], []

            # Plot Buffer Area
            if not buffered_gdf.empty:
                buffered_plot_gdf = buffered_gdf.to_crs(WEB_MERCATOR)
                buffered_plot_gdf.plot(ax=ax3, facecolor='lightblue', edgecolor='blue', alpha=0.15)
                handles.append(Patch(facecolor='lightblue', edgecolor='blue', label=f"{sel_config['buffer_km']}km Buffer Area", alpha=0.4))

            # Plot All Buildings in Buffer (if available)
            if not osm_gdf.empty:
                all_osm_plot_gdf = osm_gdf.to_crs(WEB_MERCATOR)
                buildings_in_buffer_plot = all_osm_plot_gdf[all_osm_plot_gdf['building'].notna()]
                if not buildings_in_buffer_plot.empty:
                    buildings_in_buffer_plot.plot(ax=ax3, color='gray', alpha=0.4, linewidth=0)
                    handles.append(Patch(color='gray', label='Buildings in Buffer (from OSM)', alpha=0.4))

             # Plot Union Boundary
            if not union_gdf.empty:
                union_plot_gdf3 = union_gdf.to_crs(WEB_MERCATOR)
                union_plot_gdf3.boundary.plot(ax=ax3, edgecolor='green', linewidth=2.5, label='Union of Tracts')
                handles.append(Line2D([0], [0], color='green', lw=2.5, label='Union of Tracts Boundary'))

            try:
                cx.add_basemap(ax3, crs=WEB_MERCATOR, source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap for Fig 3 failed: {e_base}")
            style_map(ax3, 'Figure 3 (Initial): Buffer & All Buildings')
            ax3.legend(handles=handles, loc='upper left', fontsize=9)
            fig3_path = os.path.join(viz_subdir, 'setup_map_3_initial.png')
            plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            status_messages.append(f"Saved: {os.path.basename(fig3_path)}")
        except Exception as e_fig:
            st.warning(f"Could not generate/save Figure 3 (Initial): {e_fig}")

    # --- 7. Calculate and Save Statistics --- 
    if not module_error:
        st.markdown("--- *Calculating Setup Statistics* ---")
        try:
            stats_a = {
                "Selected Area (km²)": area_selected_km2,
                "Union Area (km²)": area_union_km2,
                "Buffered Area (km²)": area_buffered_km2,
                "OSM Features Fetched (Buffer)": len(osm_gdf),
                "Tracts Fetched (Intersecting)": len(tracts_gdf),
                "Tracts with Census Data": len(census_data_df[census_data_df['GEOID'].isin(tracts_gdf['GEOID'])]) if 'GEOID' in census_data_df and not census_data_df.empty else 0,
                "Total Pop (Fetched Tracts)": census_data_df['total_population'].sum() if 'total_population' in census_data_df else 0,
                "Avg Income (Fetched Tracts)": census_data_df['median_income'].replace(0, np.nan).mean() if 'median_income' in census_data_df else 0, # Avoid zeros in mean
            }
            stats_a_df = pd.DataFrame.from_dict(stats_a, orient='index', columns=['Value'])
            stats_a_path = os.path.join(stats_subdir, 'setup_stats.csv')
            stats_a_df.to_csv(stats_a_path)
            status_messages.append(f"Saved: {os.path.basename(stats_a_path)}")
            st.write("Setup Statistics:")
            st.dataframe(stats_a_df)
        except Exception as e_stat:
            st.warning(f"Could not calculate/save setup stats: {e_stat}")
            stats_a_df = pd.DataFrame() # Ensure empty on error

    # --- Completion --- 
    st.markdown("--- *Module A Complete* ---")
    module_A_time = time.time() - module_A_start
    status_messages.append(f"Module A completed in {module_A_time:.2f} seconds.")
    st.success(f"Module A finished in {module_A_time:.2f}s.")

    # Return all results
    return {
        "selected_gdf": selected_gdf,
        "tracts_gdf": tracts_gdf,
        "union_gdf": union_gdf,
        "buffered_gdf": buffered_gdf,
        "osm_gdf": osm_gdf,
        "census_data_df": census_data_df,
        "utm_crs": utm_crs,
        "stats_a_df": stats_a_df,
        "status_messages": status_messages,
        "error": module_error
    }

# --- Removed old placeholder code --- 