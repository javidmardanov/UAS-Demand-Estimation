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

# Import utilities - Use try-except for robustness during development
try:
    from utils.helpers import save_empty_gdf, save_empty_df, style_map
except ImportError:
    st.error("Failed to import utility functions from utils.helpers. Ensure it exists.")
    def save_empty_gdf(f, **kwargs): pass
    def save_empty_df(f, **kwargs): pass
    def style_map(ax, title): pass

WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857'

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

    # --- 2. Classify Residential Buildings --- 
    if not module_error:
        st.markdown("--- *Classifying Residential Buildings* ---")
        try:
            res_keywords = class_config.get('residential_keywords', [])
            non_res_keywords = class_config.get('non_residential_keywords', []) # Keywords that override residential
            status_messages.append(f"Residential Keywords: {res_keywords}")
            status_messages.append(f"Non-Residential Keywords: {non_res_keywords}")
            
            # Initial residential classification based on 'building' tag
            buildings_gdf['residential'] = buildings_gdf['building'].str.lower().isin(res_keywords)
            initial_residential_count = buildings_gdf['residential'].sum()
            status_messages.append(f"Initially classified {initial_residential_count} buildings as residential based on `building` tag.")
            
            # Override: Mark as non-residential if 'building' tag matches non-residential keywords
            override_non_res_mask = buildings_gdf['building'].str.lower().isin(non_res_keywords)
            buildings_gdf.loc[override_non_res_mask, 'residential'] = False
            override_count = override_non_res_mask.sum()
            if override_count > 0:
                status_messages.append(f"Overrode {override_count} buildings to non-residential based on specific `building` tags.")

            final_residential_count = buildings_gdf['residential'].sum()
            status_messages.append(f"Final residential building count: {final_residential_count}")
        except Exception as e:
            st.error(f"Error during residential classification: {e}")
            status_messages.append(f"ERROR: Residential classification failed: {e}")
            module_error = True

    # --- 3. Identify Stores --- 
    if not module_error:
        st.markdown("--- *Identifying Stores (Origins)* ---")
        try:
            store_tags = class_config.get('store_tags_keywords', {})
            store_keywords_flat = {kw.lower() for kws in store_tags.values() for kw in kws} # Flattened set for faster checking
            status_messages.append(f"Looking for store keywords across tags: {list(store_tags.keys())}")

            # Iterate through OSM tags specified in config (amenity, shop, etc.)
            store_mask = pd.Series(False, index=buildings_gdf.index)
            assigned_types = pd.Series(None, index=buildings_gdf.index, dtype=object)

            for tag, keywords in store_tags.items():
                if tag in buildings_gdf.columns:
                    lowercase_keywords = {kw.lower() for kw in keywords}
                    # Check if the tag value exactly matches any keyword for this tag type
                    current_tag_mask = buildings_gdf[tag].str.lower().isin(lowercase_keywords)
                    
                    # Update the overall store mask
                    store_mask |= current_tag_mask
                    
                    # Assign store type based on the *first* matching tag/keyword found (simplistic)
                    # TODO: Refine logic if multiple store types match a single building?
                    assigned_types.loc[current_tag_mask & assigned_types.isna()] = tag # Assign parent tag as type for now
            
            buildings_gdf['is_store'] = store_mask
            buildings_gdf['store_type'] = assigned_types
            
            # Also check 'name' field if specified in config (common for some places)
            if 'name' in store_tags and 'name' in buildings_gdf.columns:
                 name_keywords = {kw.lower() for kw in store_tags['name']}
                 name_mask = buildings_gdf['name'].str.lower().apply(lambda x: any(kw in x for kw in name_keywords if kw)) # Partial match in name
                 buildings_gdf.loc[name_mask, 'is_store'] = True
                 # Assign type 'name_match' or similar? For now, let previous type stand or be None.
                 buildings_gdf.loc[name_mask & buildings_gdf['store_type'].isna(), 'store_type'] = 'name_match'
            
            store_count = buildings_gdf['is_store'].sum()
            status_messages.append(f"Identified {store_count} potential stores based on configured tags/keywords.")

            # Ensure residential buildings are not marked as stores (config preference?)
            # For now, assume stores are non-residential for demand allocation
            stores_marked_residential = buildings_gdf[buildings_gdf['is_store'] & buildings_gdf['residential']].copy()
            if not stores_marked_residential.empty:
                st.warning(f"{len(stores_marked_residential)} buildings were classified as both residential and store. Forcing them to non-residential for population allocation.")
                status_messages.append(f"WARN: Forcing {len(stores_marked_residential)} store buildings to non-residential.")
                buildings_gdf.loc[stores_marked_residential.index, 'residential'] = False

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

    # --- 6. Calculate and Save Statistics --- 
    if not module_error:
        st.markdown("--- *Calculating Classification Statistics* ---")
        try:
            total_buildings = len(classified_buildings_gdf)
            residential_count = classified_buildings_gdf['residential'].sum() if 'residential' in classified_buildings_gdf.columns else 0
            non_residential_count = total_buildings - residential_count
            store_count = classified_buildings_gdf['is_store'].sum() if 'is_store' in classified_buildings_gdf.columns else 0
            
            # Calculate store counts by type
            store_type_counts = pd.Series(dtype=int)
            if not stores_gdf.empty and 'store_type' in stores_gdf.columns:
                store_type_counts = stores_gdf['store_type'].fillna('Unknown').value_counts()
            
            stats_b = {
                "Total Buildings Processed": total_buildings,
                "Residential Buildings": residential_count,
                "Non-Residential Buildings": non_residential_count,
                "Identified Stores": store_count,
            }
            # Add store type counts to the stats dictionary
            for store_type, count in store_type_counts.items():
                stats_b[f"Stores - {store_type}"] = count

            stats_b_df = pd.DataFrame.from_dict(stats_b, orient='index', columns=['Value'])
            stats_b_path = os.path.join(stats_subdir, 'classification_stats.csv')
            stats_b_df.to_csv(stats_b_path)
            status_messages.append(f"Saved: {os.path.basename(stats_b_path)}")
            st.write("Classification Statistics:")
            st.dataframe(stats_b_df)
        except Exception as e_stat:
            st.warning(f"Could not calculate/save classification stats: {e_stat}")
            status_messages.append(f"WARN: Stats calculation failed: {e_stat}")
            stats_b_df = pd.DataFrame() # Ensure empty on error

    # --- 7. Generate Visualization (Fig 4) --- 
    if not module_error:
        st.markdown("--- *Generating Classification Map (Fig 4)* ---")
        try:
            st.write("Generating Fig 4: Building Classification & Stores...")
            fig4, ax4 = plt.subplots(figsize=(10, 10))
            handles, labels = [], []

            # Plot Union Boundary first
            if not union_gdf.empty:
                 union_plot_gdf = union_gdf.to_crs(WEB_MERCATOR)
                 union_plot_gdf.boundary.plot(ax=ax4, edgecolor='green', linewidth=2.0, linestyle=':', label='Union Boundary', zorder=3)
                 handles.append(Line2D([0], [0], color='green', lw=2.0, linestyle=':', label='Union of Tracts Boundary'))

            # Plot Buildings within Union, colored by type
            if not classified_buildings_gdf.empty and not union_gdf.empty:
                buildings_in_union = gpd.clip(classified_buildings_gdf, union_gdf).to_crs(WEB_MERCATOR)
                if not buildings_in_union.empty:
                    # Residential
                    res_buildings = buildings_in_union[buildings_in_union['residential']]
                    if not res_buildings.empty:
                        res_buildings.plot(ax=ax4, color='blue', alpha=0.5, label='Residential')
                        handles.append(Patch(color='blue', label='Residential Buildings (in Union)', alpha=0.5))
                    # Non-Residential
                    non_res_buildings = buildings_in_union[~buildings_in_union['residential']]
                    if not non_res_buildings.empty:
                        non_res_buildings.plot(ax=ax4, color='orange', alpha=0.5, label='Non-Residential')
                        handles.append(Patch(color='orange', label='Non-Residential Buildings (in Union)', alpha=0.5))
            
            # Plot Stores (Points) in Buffer
            if not stores_gdf.empty:
                stores_plot_gdf = stores_gdf.to_crs(WEB_MERCATOR)
                stores_plot_gdf.plot(ax=ax4, color='red', marker='*', markersize=50, label='Stores', zorder=5)
                handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Store Locations (in Buffer)')) # Use Line2D for marker legend

            try:
                cx.add_basemap(ax4, crs=WEB_MERCATOR, source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap for Fig 4 failed: {e_base}")
            style_map(ax4, 'Figure 4: Building Classification & Stores')
            ax4.legend(handles=handles, loc='upper left', fontsize=9)
            fig4_path = os.path.join(viz_subdir, 'classification_map_4.png')
            plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
            plt.close(fig4)
            status_messages.append(f"Saved: {os.path.basename(fig4_path)}")
        except Exception as e_fig:
            st.warning(f"Could not generate/save Figure 4: {e_fig}")
            status_messages.append(f"WARN: Figure 4 generation failed: {e_fig}")

    # --- Completion --- 
    module_B_time = time.time() - module_B_start
    status_messages.append(f"Module B completed in {module_B_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module B finished in {module_B_time:.2f}s.")
    else:
        st.error(f"Module B finished with errors in {module_B_time:.2f}s.")

    # --- Return Results --- 
    return {
        "classified_buildings_gdf": classified_buildings_gdf, # Return the dataframe with GEOID/unique_id
        "stores_gdf": stores_gdf,
        "stats_b_df": stats_b_df,
        "status_messages": status_messages,
        "error": module_error
    } 