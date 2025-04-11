# modules/classification.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import json
import streamlit as st
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
# Add imports needed for detailed classification logic
import re
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Import utilities - Use try-except for robustness during development
try:
    from utils.helpers import save_empty_gdf, save_empty_df, style_map, estimate_utm_crs # Add estimate_utm_crs
except ImportError:
    st.error("Failed to import utility functions from utils.helpers. Ensure it exists.")
    def save_empty_gdf(f, **kwargs): pass
    def save_empty_df(f, **kwargs): pass
    def style_map(ax, title): pass
    def estimate_utm_crs(gdf): return 'EPSG:32614' # Basic fallback

WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857'

# --- Detailed Rule-Based Classification Function (from Colab v20.3) ---
def classify_building_rules_detailed(buildings_gdf, all_osm_features_gdf, buffer_poly, utm_crs, config):
    """Classifies buildings based on detailed rules similar to Colab v20.3.

    Uses OSM tags, area, road proximity, and landuse context.

    Args:
        buildings_gdf (gpd.GeoDataFrame): DataFrame of building features.
        all_osm_features_gdf (gpd.GeoDataFrame): DataFrame containing all OSM features 
                                                (including roads and landuse) in the buffer.
        buffer_poly (Polygon/MultiPolygon): The buffer geometry for context.
        utm_crs (str): Estimated UTM CRS for accurate area/buffer calculations.
        config (dict): The main configuration dictionary.

    Returns:
        gpd.GeoDataFrame: Input buildings_gdf with added 'residential' column ('yes'/'no').
    """
    print("--- Starting Detailed Rule-Based Classification ---")
    buildings = buildings_gdf.copy()
    # Ensure rule_based_parameters exist
    if 'building_classification' not in config or 'rule_based_parameters' not in config['building_classification']:
        st.error("Config missing 'building_classification.rule_based_parameters'. Cannot run detailed classification.")
        buildings['residential'] = 'no' # Default fallback
        return buildings
    
    rules = config['building_classification']['rule_based_parameters']
    buildings['residential'] = 'unknown' # Start with unknown

    # Calculate area if missing
    if 'building_area_m2' not in buildings.columns or buildings['building_area_m2'].isna().any():
        print("  Calculating building areas...")
        buildings_proj_area = buildings.to_crs(utm_crs)
        buildings['building_area_m2'] = buildings_proj_area.geometry.area
    
    print("  Applying OSM tag rules...")
    res_tags = rules.get('residential_building_tags', [])
    nonres_tags = rules.get('nonresidential_building_tags', [])
    res_keywords = rules.get('residential_name_keywords', [])
    nonres_keywords = rules.get('nonresidential_name_keywords', [])
    
    # Ensure relevant tag columns exist and are string type
    for tag in ['building', 'shop', 'amenity', 'office', 'tourism', 'leisure', 'name']:
        if tag not in buildings.columns:
            buildings[tag] = None # Add column if missing
        buildings[tag] = buildings[tag].astype(str).fillna('')

    # Apply rules based on 'building' tag
    buildings.loc[buildings['building'].str.lower().isin(res_tags), 'residential'] = 'yes'
    buildings.loc[buildings['building'].str.lower().isin(nonres_tags), 'residential'] = 'no'
    
    # Apply rules for other explicit non-residential tags
    # --- COMMENTED OUT - This rule was likely too aggressive --- 
    # for tag in ['shop', 'amenity', 'office', 'tourism', 'leisure']:
    #      buildings.loc[buildings[tag] != '', 'residential'] = 'no' # If tag has any value, assume non-res
    # --- End Commented Out ---
    
    # Apply name keyword rules
    if nonres_keywords:
        # Corrected regex format
        nonres_name_pattern = r'(?:^|\b)(?:{})(?:$|\b)'.format('|'.join(map(re.escape, nonres_keywords)))
        nonres_name_mask = buildings['name'].str.contains(nonres_name_pattern, case=False, na=False, regex=True)
        buildings.loc[nonres_name_mask, 'residential'] = 'no'
    if res_keywords:
        # Corrected regex format
        res_name_pattern = r'(?:^|\b)(?:{})(?:$|\b)'.format('|'.join(map(re.escape, res_keywords)))
        res_name_mask = buildings['name'].str.contains(res_name_pattern, case=False, na=False, regex=True)
        # Only apply residential name if current status is unknown or likely_yes (from area)
        buildings.loc[res_name_mask & buildings['residential'].isin(['unknown', 'likely_yes']), 'residential'] = 'yes'

    print(f"    After Tag/Name Rules: {buildings['residential'].value_counts().to_dict()}") # DEBUG

    print("  Applying area rules...")
    small_thresh = rules.get('likely_max_residential_area_m2', 500)
    large_thresh = rules.get('min_nonresidential_area_m2', 1500)
    tiny_thresh = rules.get('min_residential_area_m2', 30)
    
    likely_res_mask = (buildings['building_area_m2'] < small_thresh) & (buildings['residential'] == 'unknown')
    likely_nonres_mask = (buildings['building_area_m2'] > large_thresh) & (buildings['residential'] == 'unknown')
    buildings.loc[likely_res_mask, 'residential'] = 'likely_yes'
    buildings.loc[likely_nonres_mask, 'residential'] = 'likely_no'
    
    # Tiny buildings marked likely_yes might be sheds/garages unless tagged explicitly
    tiny_mask = (buildings['building_area_m2'] < tiny_thresh) & \
                (buildings['residential'] == 'likely_yes') & \
                (~buildings['building'].str.lower().isin(['house', 'bungalow', 'shed', 'garage', 'detached']))
    buildings.loc[tiny_mask, 'residential'] = 'no'

    print(f"    After Area Rules: {buildings['residential'].value_counts().to_dict()}") # DEBUG

    # --- Temporarily Comment Out Context Rules --- 
    # print("  Applying road proximity rules...")
    # try:
    #     # Use all OSM features for road/landuse context
    #     context_features = all_osm_features_gdf.copy()
    #     if not isinstance(buffer_poly, (Polygon, MultiPolygon)):
    #         buffer_poly = unary_union(buffer_poly)
    #     
    #     if 'highway' not in context_features.columns:
    #         print("  Skipping road proximity: 'highway' tag not found in OSM features.")
    #     else:
    #         roads = context_features[context_features['highway'].notna()].copy()
    #         if not roads.empty:
    #             # ... (rest of road logic) ...
    #         else:
    #             print("  No highway features found for proximity checks.")
    # except Exception as e:
    #     print(f"  Road proximity check failed: {e}")
    #     st.warning(f"Road proximity check failed: {e}")
    # print(f"    After Road Rules: {buildings['residential'].value_counts().to_dict()}") # DEBUG

    # print("  Applying landuse context rules...")
    # try:
    #     context_features = all_osm_features_gdf # Reuse from above or reload if needed
    #     if 'landuse' not in context_features.columns:
    #         print("  Skipping landuse context: 'landuse' tag not found.")
    #     else:
    #         landuse = context_features[context_features['landuse'].notna()].copy()
    #         if not landuse.empty:
    #             # ... (rest of landuse logic) ...
    #         else:
    #             print("  No landuse features found for context checks.")
    # except Exception as e:
    #     print(f"  Landuse context check failed: {e}")
    #     st.warning(f"Landuse context check failed: {e}")
    # print(f"    After Landuse Rules: {buildings['residential'].value_counts().to_dict()}") # DEBUG
    # --- End Temporarily Commented Out --- 

    # Final assignment: resolve 'likely' states
    buildings['residential'] = buildings['residential'].replace({'likely_yes': 'yes', 'likely_no': 'no', 'unknown': 'no'})
    
    print("--- Detailed Rule-Based Classification Finished ---")
    return buildings

# --- Store Identification Logic (can be kept similar or adapted from Colab) ---
def identify_store_type(row, store_config):
    """Identifies store type based on tags and keywords.
    Adapted from Colab v20.3 identify_store_type.
    """
    tags = row.to_dict()
    # Ensure keys exist, default to empty string
    name = str(tags.get('name', '')).lower()
    shop = str(tags.get('shop', '')).lower()
    bldg = str(tags.get('building', '')).lower()
    amenity = str(tags.get('amenity', '')).lower()
    
    store_keywords_config = store_config.get('name_keywords', [])
    store_keywords_pattern = "|".join(map(re.escape, store_keywords_config)) if store_keywords_config else None

    # Specific checks (similar to Colab)
    if 'walmart' in name: return 'Walmart'
    if 'target' in name: return 'Target'
    if 'kroger' in name: return 'Kroger'
    if 'h-e-b' in name or 'heb' in name: return 'H-E-B'
    if 'costco' in name: return 'Costco'
    if 'amazon' in name: return 'Warehouse/Fulfillment'
    if 'distribution center' in name or 'fulfillment' in name: return 'Warehouse/Fulfillment'
    if 'warehouse' in name or bldg == 'warehouse': return 'Warehouse/Fulfillment'
    if 'supercenter' in name or shop == 'supermarket' or 'grocery' in name or 'market' in name:
         # Check if it's also a pharmacy to prioritize that?
         if shop == 'pharmacy' or amenity == 'pharmacy' or any(p in name for p in ['cvs', 'walgreens', 'pharmacy']):
              return 'Pharmacy' 
         return 'Retail Hub' # Grocery/Supermarket
    if shop == 'department_store' or shop == 'mall' or shop == 'wholesale': return 'Retail Hub'
    if shop == 'pharmacy' or amenity == 'pharmacy' or any(p in name for p in ['cvs', 'walgreens', 'pharmacy']):
         return 'Pharmacy'
    if shop == 'convenience': return 'Convenience'
    if amenity == 'fast_food': return 'Fast Food'

    # Check general keywords from config for broader categories
    # Check shop tags from config
    if shop in store_config.get('shop', []): return 'Retail Hub' # Default for matched shop tags
    # Check amenity tags from config
    if amenity in store_config.get('amenity', []): return 'Retail Hub' # Default for matched amenity tags
    # Check building tags from config
    if bldg in store_config.get('building', []): return 'Retail Hub' # Default for matched building tags
    
    # Check name keywords last as a fallback broad category
    if store_keywords_pattern and re.search(store_keywords_pattern, name, re.IGNORECASE):
        return 'Retail Hub / Warehouse' # Broad category if name matches general list
        
    return None # No match found


# --- Main Module Function ---
def run_module_b(config, osm_gdf, buffer_gdf, union_gdf):
    """Executes Module B: Building Classification & Store Identification.

    Args:
        config (dict): The configuration dictionary.
        osm_gdf (gpd.GeoDataFrame): GeoDataFrame of OSM features from Module A.
        buffer_gdf (gpd.GeoDataFrame): GeoDataFrame of the buffered area from Module A.
        union_gdf (gpd.GeoDataFrame): GeoDataFrame of the union of tracts from Module A.

    Returns:
        dict: A dictionary containing results:
              'classified_buildings_gdf', 'stores_gdf',
              'stats_b_df', 'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module B: Building Classification & Store ID")
    st.markdown("--- *Initializing* ---")
    module_B_start = time.time()
    class_config = config['building_classification']
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")

    # Initialize return values
    classified_buildings_gdf = gpd.GeoDataFrame()
    stores_gdf = gpd.GeoDataFrame()
    stats_b_df = pd.DataFrame()
    status_messages = []
    module_error = False

    # --- Check Inputs --- 
    if osm_gdf is None or osm_gdf.empty:
        st.warning("Module B: Input OSM GeoDataFrame is empty. Skipping classification.")
        status_messages.append("WARN: Skipping Module B - No OSM features provided.")
        # Save empty files to prevent downstream errors
        save_empty_gdf(os.path.join(data_subdir, "classified_buildings.geojson"))
        save_empty_gdf(os.path.join(data_subdir, "stores.geojson"))
        save_empty_df(os.path.join(stats_subdir, 'classification_stats.csv'))
        return {
            "classified_buildings_gdf": classified_buildings_gdf, "stores_gdf": stores_gdf,
            "stats_b_df": stats_b_df, "status_messages": status_messages, "error": False # Not an error, just skipped
        }

    # --- 1. Filter to Buildings & Prepare Columns --- 
    st.markdown("--- *Filtering & Preparing OSM Features* ---")
    buildings_gdf = gpd.GeoDataFrame() # Initialize
    try:
        # Ensure we have a copy to avoid modifying Module A's output
        # Crucially, include GEOID and unique_id if they exist in osm_gdf
        cols_to_select = [col for col in osm_gdf.columns if col != 'geometry'] # Select all non-geometry cols
        if 'unique_id' not in cols_to_select: cols_to_select.append('unique_id') # Ensure unique_id is there
        if 'GEOID' not in cols_to_select and 'GEOID' in osm_gdf.columns: 
            cols_to_select.append('GEOID') # Ensure GEOID is kept if present
        
        buildings_gdf = osm_gdf[osm_gdf['building'].notna()][cols_to_select + ['geometry']].copy()
        status_messages.append(f"Filtered to {len(buildings_gdf)} building features from OSM data.")

        if buildings_gdf.empty:
            st.warning("Module B: No building features found in OSM data. Skipping classification.")
            status_messages.append("WARN: Skipping Module B - No buildings found in OSM data.")
            save_empty_gdf(os.path.join(data_subdir, "classified_buildings.geojson"))
            save_empty_gdf(os.path.join(data_subdir, "stores.geojson"))
            save_empty_df(os.path.join(stats_subdir, 'classification_stats.csv'))
            return {
                "classified_buildings_gdf": classified_buildings_gdf, "stores_gdf": stores_gdf,
                "stats_b_df": stats_b_df, "status_messages": status_messages, "error": False
            }
        
        # Initialize classification columns
        buildings_gdf['residential'] = False # Default to non-residential
        buildings_gdf['is_store'] = False    # Default to not a store
        buildings_gdf['store_type'] = None   # Default store type

        # Convert relevant columns to string to handle potential mixed types or NaNs safely
        for col in ['building', 'amenity', 'shop', 'landuse', 'name']:
             if col in buildings_gdf.columns:
                 buildings_gdf[col] = buildings_gdf[col].astype(str).fillna('') # Fill NaN with empty string

    except Exception as e:
        st.error(f"Error preparing building data: {e}")
        status_messages.append(f"ERROR: Preparing building data failed: {e}")
        module_error = True
        # Fallback to empty GDFs on error
        buildings_gdf = gpd.GeoDataFrame()
        classified_buildings_gdf = gpd.GeoDataFrame()
        stores_gdf = gpd.GeoDataFrame()

    # --- 2. Classify Residential Buildings (Using Detailed Logic) --- 
    if not module_error:
        st.markdown("--- *Classifying Residential Buildings (Detailed)* ---")
        try:
            # Estimate UTM CRS if not provided directly (needed for area calcs)
            # Note: Module A should return this, but add fallback.
            utm_crs = config.get('utm_crs') # Check if passed in config first
            if not utm_crs:
                 st.warning("UTM CRS not found in config, estimating for Module B.")
                 utm_crs = estimate_utm_crs(buildings_gdf)
            
            # Get buffer polygon geometry
            if buffer_gdf is None or buffer_gdf.empty:
                 st.error("Buffer geometry missing, cannot perform detailed classification.")
                 raise ValueError("Missing buffer geometry for classification context.")
            buffer_polygon = buffer_gdf.geometry.iloc[0]
            
            # Call the detailed classification function
            # It needs the original osm_gdf for context (roads, landuse)
            buildings_gdf = classify_building_rules_detailed(
                buildings_gdf=buildings_gdf, 
                all_osm_features_gdf=osm_gdf, # Pass the full OSM data 
                buffer_poly=buffer_polygon, 
                utm_crs=utm_crs, 
                config=config
            )
            # Correctly count residential buildings by summing the boolean column
            final_residential_count = (buildings_gdf['residential'] == 'yes').sum()
            status_messages.append(f"Final residential building count (detailed): {final_residential_count}")
            
            # --- DEBUG PRINT AFTER CLASSIFICATION FUNCTION ---
            # print(f"DEBUG (Module B): AFTER classify_building_rules_detailed - Residential counts: {buildings_gdf['residential'].value_counts().to_dict()}")
            # --- END DEBUG PRINT ---
            
        except Exception as e_classify:
            st.error(f"Error during detailed residential classification: {e_classify}")
            status_messages.append(f"ERROR: Detailed residential classification failed: {e_classify}")
            module_error = True

    # --- 3. Identify Stores (Using Enhanced Logic) --- 
    if not module_error:
        st.markdown("--- *Identifying Stores (Origins)* ---")
        try:
            store_cfg = class_config.get('store_tags_keywords', {})
            status_messages.append(f"Identifying stores using config: {list(store_cfg.keys())}")
            
            # Apply the enhanced identify_store_type function
            buildings_gdf['store_type'] = buildings_gdf.apply(lambda row: identify_store_type(row, store_cfg), axis=1)
            buildings_gdf['is_store'] = buildings_gdf['store_type'].notna()
            
            store_count = buildings_gdf['is_store'].sum()
            status_messages.append(f"Identified {store_count} potential stores.")
            if store_count > 0:
                 status_messages.append(f"Store types found:\n{buildings_gdf[buildings_gdf['is_store']]['store_type'].value_counts().to_string()}")

            # Force stores to be non-residential
            stores_marked_residential = buildings_gdf[buildings_gdf['is_store'] & (buildings_gdf['residential'] == 'yes')]
            if not stores_marked_residential.empty:
                st.warning(f"{len(stores_marked_residential)} buildings were classified as both residential and store. Forcing them to non-residential for population allocation.")
                status_messages.append(f"WARN: Forcing {len(stores_marked_residential)} store buildings to non-residential.")
                buildings_gdf.loc[stores_marked_residential.index, 'residential'] = 'no'

            # --- DEBUG PRINT AFTER OVERLAP LOGIC ---
            # print(f"DEBUG (Module B): AFTER overlap logic - Residential counts: {buildings_gdf['residential'].value_counts().to_dict()}")
            # --- END DEBUG PRINT ---
            
        except Exception as e:
            st.error(f"Error during store identification: {e}")
            status_messages.append(f"ERROR: Store identification failed: {e}")
            module_error = True

    # --- 4. Create Stores GeoDataFrame (Points) --- 
    if not module_error:
        st.markdown("--- *Generating Stores GeoDataFrame* ---")
        try:
            potential_stores = buildings_gdf[buildings_gdf['is_store']].copy()
            if not potential_stores.empty:
                potential_stores = potential_stores[potential_stores.geometry.is_valid]
                if not potential_stores.empty:
                    # Select relevant columns for stores output, including unique_id
                    store_cols = ['unique_id', 'store_type', 'geometry']
                    # Add other relevant tags if needed, e.g., 'name'
                    if 'name' in potential_stores.columns: store_cols.insert(1, 'name') 
                    
                    stores_gdf = potential_stores[store_cols].copy()
                    stores_gdf['geometry'] = stores_gdf.geometry.centroid
                    stores_gdf['store_id'] = range(len(stores_gdf)) 
                    stores_gdf.set_crs(potential_stores.crs, inplace=True) # Ensure CRS is set
                    status_messages.append(f"Created stores_gdf with {len(stores_gdf)} store points.")
                else:
                    st.warning("No valid store geometries found after filtering.")
                    stores_gdf = gpd.GeoDataFrame() # Ensure empty
            else:
                status_messages.append("No stores identified, creating empty stores_gdf.")
                stores_gdf = gpd.GeoDataFrame()
        except Exception as e_store:
            st.error(f"Error creating stores GeoDataFrame: {e_store}")
            status_messages.append(f"ERROR: Stores GDF creation failed: {e_store}")
            module_error = True
            stores_gdf = gpd.GeoDataFrame() # Ensure empty on error

    # --- 5. Save Classified Buildings --- 
    if not module_error: 
        st.markdown("--- *Saving Classified Buildings* ---")
        try:
            classified_file = os.path.join(data_subdir, "classified_buildings.geojson")
            # Select columns to save - INCLUDE unique_id and GEOID if present
            cols_to_save = ['unique_id', 'residential', 'is_store', 'store_type', 'geometry']
            if 'GEOID' in buildings_gdf.columns: cols_to_save.insert(1, 'GEOID')
            # Add other potentially useful columns? e.g., building tag, name?
            if 'building' in buildings_gdf.columns: cols_to_save.append('building')
            if 'name' in buildings_gdf.columns: cols_to_save.append('name')
            
            buildings_to_save = buildings_gdf[[col for col in cols_to_save if col in buildings_gdf.columns]].copy()
            
            # Ensure WGS84 before saving GeoJSON
            if buildings_to_save.crs != WGS84:
                 buildings_to_save = buildings_to_save.to_crs(WGS84)
                 
            if not buildings_to_save.empty:
                 buildings_to_save.to_file(classified_file, driver="GeoJSON")
            else:
                 save_empty_gdf(classified_file) # Save empty if needed
            status_messages.append(f"Saved: {os.path.basename(classified_file)}")
        except Exception as e_save:
            st.warning(f"Could not save classified_buildings.geojson: {e_save}")
            status_messages.append(f"WARN: Failed to save {os.path.basename(classified_file)}: {e_save}")
            # Don't set module_error=True, maybe user can proceed partially?
            classified_buildings_gdf = buildings_gdf # Return original if save failed but processing okay?
            # OR set module_error = True if saving is critical?

    # Assign the processed buildings_gdf to the return variable
    classified_buildings_gdf = buildings_gdf 

    # --- 6. Calculate Stats --- 
    if not module_error:
        st.markdown("--- *Calculating Classification Statistics* ---")
        try:
            total_buildings_classified = len(buildings_gdf)
            # Use boolean comparison and sum for counts
            res_count = (buildings_gdf['residential'] == 'yes').sum()
            non_res_count = (buildings_gdf['residential'] == 'no').sum() # Use explicit 'no'
            stores_count = len(stores_gdf)
            
            stats_b = {
                "Total Buildings Classified": total_buildings_classified,
                "Residential Buildings": res_count,
                "Non-Residential Buildings": non_res_count,
                "Stores Identified": stores_count,
            }
            stats_b_df = pd.DataFrame.from_dict(stats_b, orient='index', columns=['Value'])
            stats_b_df['Value'] = stats_b_df['Value'].astype(str) # Convert Value column to string
            stats_b_path = os.path.join(stats_subdir, 'classification_stats.csv')
            stats_b_df.to_csv(stats_b_path)
            status_messages.append(f"Saved: {os.path.basename(stats_b_path)}")
            st.write("Classification Statistics:")
            st.dataframe(stats_b_df)
        except Exception as e_stats:
            st.warning(f"WARN: Stats calculation failed: {e_stats}")
            status_messages.append(f"WARN: Stats calculation failed: {e_stats}")
            stats_b_df = pd.DataFrame() # Ensure empty on error

    # --- 7. Generate Visualization (Fig 4) --- 
    if not module_error:
        st.markdown("--- *Generating Classification Map (Fig 4)* ---")
        try:
            st.write("Generating Fig 4: Building Classification & Stores...")
            fig_b4, ax_b4 = plt.subplots(figsize=(12, 12))
            
            # Project data for plotting
            plot_data_b4 = buildings_gdf.to_crs(WEB_MERCATOR)
            union_plot_gdf_b4 = union_gdf.to_crs(WEB_MERCATOR)
            stores_plot_b4 = stores_gdf.to_crs(WEB_MERCATOR) # Stores are points

            # Clip buildings to union for display (optional, but matches Colab figure)
            plot_data_b4_clipped = gpd.clip(plot_data_b4, union_plot_gdf_b4)
            
            # Correctly filter based on the 'residential' column value
            non_res_plot = plot_data_b4_clipped[plot_data_b4_clipped['residential'] == 'no']
            res_plot = plot_data_b4_clipped[plot_data_b4_clipped['residential'] == 'yes']

            # Plot non-residential first (background)
            if not non_res_plot.empty:
                non_res_plot.plot(ax=ax_b4, color='orange', alpha=0.6, label='Non-Residential Buildings (in Union)', linewidth=0.5)
            
            # Plot residential
            if not res_plot.empty:
                res_plot.plot(ax=ax_b4, color='blue', alpha=0.7, label='Residential Buildings (in Union)', linewidth=0.5)

            # Plot union boundary
            union_plot_gdf_b4.boundary.plot(ax=ax_b4, color='green', linewidth=2.5, linestyle=':', label='Union of Tracts Boundary', zorder=4)
            
            # Plot store points
            if not stores_plot_b4.empty:
                stores_plot_b4.plot(ax=ax_b4, color='red', marker='*', markersize=100, label='Store Locations (in Buffer)', zorder=5)

            try:
                cx.add_basemap(ax_b4, crs=WEB_MERCATOR, source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap failed for Fig 4: {e_base}")
            
            style_map(ax_b4, 'Figure 4: Building Classification & Stores')
            
            # Create custom legend
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='Residential Buildings (in Union)'),
                Patch(facecolor='orange', alpha=0.6, label='Non-Residential Buildings (in Union)'),
                Line2D([0], [0], linestyle=':', color='green', lw=2.5, label='Union of Tracts Boundary'),
                Line2D([0], [0], marker='*', color='w', label='Store Locations (in Buffer)', 
                       markerfacecolor='red', markersize=15)
            ]
            ax_b4.legend(handles=legend_elements, loc='upper left', title="Legend")
            
            fig4_path = os.path.join(viz_subdir, 'classification_map_4.png')
            plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
            plt.close(fig_b4)
            status_messages.append(f"Saved: {os.path.basename(fig4_path)}")
            st.image(fig4_path)
        except Exception as e_fig:
            st.warning(f"WARN: Figure 4 generation failed: {e_fig}")
            status_messages.append(f"WARN: Fig 4 generation failed: {e_fig}")

    # --- Completion --- 
    module_B_time = time.time() - module_B_start
    status_messages.append(f"Module B completed in {module_B_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module B finished in {module_B_time:.2f}s.")
    else:
        st.error(f"Module B finished with errors in {module_B_time:.2f}s.")

    # --- DEBUG PRINT BEFORE RETURN --- 
    # if 'classified_buildings_gdf' in locals() and not classified_buildings_gdf.empty:
    #     print(f"DEBUG (Module B): BEFORE RETURN - Residential counts: {classified_buildings_gdf['residential'].value_counts().to_dict()}")
    # else:
    #     print("DEBUG (Module B): BEFORE RETURN - classified_buildings_gdf is empty or not defined.")
    # --- END DEBUG PRINT ---

    # --- Return Results --- 
    result_dict = {
        "classified_buildings_gdf": classified_buildings_gdf, # Return the dataframe with GEOID/unique_id
        "stores_gdf": stores_gdf,
        "stats_b_df": stats_b_df,
        "status_messages": status_messages,
        "error": module_error
    }
    return result_dict 