# modules/od_matching.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import json
import streamlit as st
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point, LineString
from datetime import datetime

# Import utilities
try:
    from utils.helpers import save_empty_gdf, save_empty_df, style_map
except ImportError:
    st.error("Failed to import utility functions from utils.helpers. Ensure it exists.")
    def save_empty_gdf(f, **kwargs): pass
    def save_empty_df(f, **kwargs): pass
    def style_map(ax, title): pass

WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857' # Added constant

# --- Helper Function for Distance Calculation (if needed, geopandas handles this mostly) ---
# Consider adding helper if complex distance logic needed beyond GeoPandas defaults

# --- Main Module Function ---
def run_module_f(config, delivery_events_df, stores_gdf, utm_crs):
    """Executes Module F: O-D Matching & Dataset Generation.

    Args:
        config (dict): The configuration dictionary.
        delivery_events_df (pd.DataFrame): Simulated delivery events from Module E.
        stores_gdf (gpd.GeoDataFrame): Identified store locations from Module B.
        utm_crs (str): Estimated UTM CRS from Module A.

    Returns:
        dict: A dictionary containing results:
              'routing_dataset_df', 'routing_dataset_full_gdf' (optional),
              'stats_f_df', 'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module F: O-D Matching & Dataset Generation")
    st.markdown("--- *Initializing* ---")
    module_F_start = time.time()
    od_config = config['origin_destination_matching']
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")
    anim_subdir = os.path.join(output_dir, "animation_frames", "od") # Subdir for frames
    os.makedirs(anim_subdir, exist_ok=True) # Create anim frame dir

    # Initialize return values
    routing_dataset_df = pd.DataFrame()
    routing_dataset_full_gdf = gpd.GeoDataFrame() # Optional detailed output
    stats_f_df = pd.DataFrame()
    status_messages = []
    module_error = False

    # --- Check Inputs --- 
    if delivery_events_df is None or delivery_events_df.empty:
        st.warning("Module F: Input delivery events DataFrame is empty. Skipping O-D matching.")
        status_messages.append("WARN: Skipping Module F - No delivery events provided.")
        module_error = True # Treat as error for skipping?
    elif not all(col in delivery_events_df.columns for col in ['order_id', 'building_unique_id', 'timestamp', 'latitude', 'longitude']):
        st.error("Module F: Input delivery events DataFrame missing required columns. Skipping.")
        status_messages.append("ERROR: Skipping Module F - Delivery events missing required columns.")
        module_error = True
        
    if stores_gdf is None or stores_gdf.empty:
        st.warning("Module F: Input stores GeoDataFrame is empty. Skipping O-D matching.")
        status_messages.append("WARN: Skipping Module F - No stores provided.")
        module_error = True
    elif not all(col in stores_gdf.columns for col in ['store_id', 'store_type', 'geometry']):
        st.error("Module F: Input stores GeoDataFrame missing required columns (store_id, store_type, geometry). Skipping.")
        status_messages.append("ERROR: Skipping Module F - Stores GDF missing required columns.")
        module_error = True
        
    if not utm_crs:
         st.warning("Module F: UTM CRS not provided. Needed for distance calculations. Skipping.")
         status_messages.append("WARN: Skipping Module F - UTM CRS missing.")
         module_error = True

    # Save empty files and return if critical error found
    if module_error:
        save_empty_df(os.path.join(data_subdir, 'routing_dataset.csv'))
        save_empty_gdf(os.path.join(data_subdir, 'routing_dataset_detailed_weekly.geojson'))
        save_empty_df(os.path.join(stats_subdir, 'od_matching_stats.csv'))
        return {
            "routing_dataset_df": routing_dataset_df, 
            "routing_dataset_full_gdf": routing_dataset_full_gdf,
            "stats_f_df": stats_f_df, 
            "status_messages": status_messages, 
            "error": True
        }

    # --- 1. Prepare Data for Matching --- 
    st.markdown("--- *Preparing Data for O-D Matching* ---")
    try:
        # --- Add Check for Valid Event Coordinates ---
        initial_event_count = len(delivery_events_df)
        coord_cols = ['latitude', 'longitude']
        # Drop rows with NaN/None in coordinates
        delivery_events_df.dropna(subset=coord_cols, inplace=True)
        valid_coord_event_count = len(delivery_events_df)
        if valid_coord_event_count < initial_event_count:
            removed_count = initial_event_count - valid_coord_event_count
            st.warning(f"Removed {removed_count} delivery events with missing lat/lon coordinates.")
            status_messages.append(f"WARN: Removed {removed_count} events with missing coordinates.")
        if valid_coord_event_count == 0:
            st.error("No delivery events with valid coordinates found. Cannot perform O-D matching.")
            raise ValueError("No valid delivery event coordinates available for matching.")
        # --- End Check ---
            
        # Convert delivery events to GeoDataFrame
        events_gdf = gpd.GeoDataFrame(
            delivery_events_df,
            geometry=gpd.points_from_xy(delivery_events_df.longitude, delivery_events_df.latitude),
            crs=WGS84
        )
        
        # --- Add Check for Valid Event Geometries (Post-Creation) ---
        initial_geom_count = len(events_gdf)
        valid_geom_mask_events = events_gdf.geometry.notna() & (~events_gdf.geometry.is_empty)
        events_gdf = events_gdf[valid_geom_mask_events].copy()
        valid_geom_event_count = len(events_gdf)
        if valid_geom_event_count < initial_geom_count:
            removed_geom_count = initial_geom_count - valid_geom_event_count
            st.warning(f"Removed {removed_geom_count} delivery events with invalid/empty geometries after points_from_xy.")
            status_messages.append(f"WARN: Removed {removed_geom_count} events with invalid geometries.")
        if valid_geom_event_count == 0:
            st.error("No delivery events with valid geometries found after points_from_xy. Cannot perform O-D matching.")
            raise ValueError("No valid delivery event geometries available for matching.")
        # --- End Check ---
            
        # Project valid events to UTM
        events_gdf = events_gdf.to_crs(utm_crs)
        status_messages.append(f"Validated and projected {len(events_gdf)} delivery events.")
        
        # Project stores to UTM
        stores_proj = stores_gdf.to_crs(utm_crs)
        # Ensure store_type is usable
        stores_proj['store_type'] = stores_proj['store_type'].fillna('unknown')
        
        # --- Add Check for Valid Store Geometries ---
        initial_store_count = len(stores_proj)
        valid_geom_mask = stores_proj.geometry.notna() & (~stores_proj.geometry.is_empty)
        stores_proj = stores_proj[valid_geom_mask].copy()
        valid_store_count = len(stores_proj)
        if valid_store_count < initial_store_count:
            removed_count = initial_store_count - valid_store_count
            st.warning(f"Removed {removed_count} stores with invalid/empty geometries before matching.")
            status_messages.append(f"WARN: Removed {removed_count} stores with invalid geometries.")
        if valid_store_count == 0:
             st.error("No stores with valid geometries found. Cannot perform O-D matching.")
             raise ValueError("No valid store geometries available for matching.")
        # --- End Check ---
             
        available_store_types = stores_proj['store_type'].unique().tolist()
        status_messages.append(f"Projected {valid_store_count} stores with valid geometry. Available types: {available_store_types}")

    except Exception as e:
        st.error(f"Error preparing data for matching: {e}")
        status_messages.append(f"ERROR: Data preparation for matching failed: {e}")
        module_error = True
        events_gdf = gpd.GeoDataFrame() # Ensure empty on error
        stores_proj = gpd.GeoDataFrame()

    # --- 2. Perform O-D Matching --- 
    matched_events_list = []
    if not module_error:
        st.markdown("--- *Performing O-D Matching* ---")
        matching_method = od_config.get('method', 'proximity').lower()
        status_messages.append(f"Using matching method: {matching_method}")
        st.write(f"Matching {len(events_gdf)} events using '{matching_method}' method...")
        
        match_prog = st.progress(0.0)
        total_events = len(events_gdf)
        processed_count = 0
        batch_size = 1000 # Process in batches for progress update

        try:
            if matching_method == 'proximity':
                # Find nearest store for each event using sjoin_nearest
                # Note: sjoin_nearest finds the single nearest feature
                matched_gdf = gpd.sjoin_nearest(events_gdf, stores_proj, how='left')
                matched_gdf.rename(columns={'index_right': 'store_index'}, inplace=True)
                
                # --- Add Check for Join Success ---
                if matched_gdf['store_index'].isna().all():
                    st.error("Spatial join (sjoin_nearest) failed to find any nearby stores for any delivery events.")
                    st.warning("Check if stores and residential areas overlap or if stores were correctly identified in Module B.")
                    status_messages.append("ERROR: sjoin_nearest found no matching origins for any events.")
                    raise ValueError("Proximity matching failed: No stores found near events.")
                # --- End Check ---
                
                status_messages.append(f"Matched events to nearest store using proximity.")
                matched_events_list.append(matched_gdf)
                match_prog.progress(1.0)

            elif matching_method == 'market_share_weighted':
                # Normalize market shares based on available store types
                market_shares = od_config.get('market_shares', {})
                available_shares = {stype: market_shares.get(stype, 0) 
                                    for stype in available_store_types if stype != 'unknown'}
                total_share = sum(available_shares.values())
                if total_share > 1e-6:
                    normalized_shares = {stype: share / total_share for stype, share in available_shares.items()}
                else:
                    st.warning("Total market share is zero. Falling back to proximity matching.")
                    normalized_shares = {} # Fallback handled below
                status_messages.append(f"Normalized Market Shares: {json.dumps(normalized_shares, indent=2)}")
                
                if not normalized_shares: # Fallback to proximity if shares are empty/zero
                    matched_gdf = gpd.sjoin_nearest(events_gdf, stores_proj, how='left')
                    matched_gdf.rename(columns={'index_right': 'store_index'}, inplace=True)
                    status_messages.append("WARN: Market shares invalid, fell back to proximity matching.")
                    matched_events_list.append(matched_gdf)
                    match_prog.progress(1.0)
                else:
                    # Assign store type probabilistically, then find nearest of that type
                    share_types = list(normalized_shares.keys())
                    share_probs = list(normalized_shares.values())
                    
                    # Assign target store type randomly based on shares
                    events_gdf['assigned_store_type'] = np.random.choice(share_types, size=total_events, p=share_probs)
                    status_messages.append("Assigned target store types based on market shares.")
                    
                    # Group events by assigned type and find nearest store *of that type*
                    for store_type, group in events_gdf.groupby('assigned_store_type'):
                        st.write(f"  Matching events assigned to type: '{store_type}'...")
                        possible_origins = stores_proj[stores_proj['store_type'] == store_type]
                        if possible_origins.empty:
                            st.warning(f"No stores found for assigned type '{store_type}'. Falling back to nearest overall store for these events.")
                            status_messages.append(f"WARN: No stores of type '{store_type}', using fallback.")
                            # Fallback: find nearest of *any* type
                            matched_group = gpd.sjoin_nearest(group, stores_proj, how='left')
                        else:
                            matched_group = gpd.sjoin_nearest(group, possible_origins, how='left')
                        
                        matched_group.rename(columns={'index_right': 'store_index'}, inplace=True)
                        matched_events_list.append(matched_group)
                        processed_count += len(group)
                        match_prog.progress(processed_count / total_events)
                    
                    status_messages.append("Matched events to nearest store of assigned type.")
                    match_prog.progress(1.0)

            elif matching_method == 'random':
                 # Assign a random store to each event
                 store_indices = stores_proj.index.tolist()
                 random_store_indices = np.random.choice(store_indices, size=total_events)
                 events_gdf['store_index'] = random_store_indices
                 matched_events_list.append(events_gdf) # Events now have store_index
                 status_messages.append("Assigned random store to each event.")
                 match_prog.progress(1.0)

            else:
                st.error(f"Unsupported O-D matching method: {matching_method}. Defaulting to proximity.")
                status_messages.append(f"ERROR: Unsupported matching method '{matching_method}'. Used proximity.")
                matched_gdf = gpd.sjoin_nearest(events_gdf, stores_proj, how='left')
                matched_gdf.rename(columns={'index_right': 'store_index'}, inplace=True)
                matched_events_list.append(matched_gdf)
                match_prog.progress(1.0)
                
            # Consolidate results if needed (market share does this iteratively)
            if matched_events_list:
                 consolidated_matches = pd.concat(matched_events_list).sort_index()
                 # Merge store details back based on store_index
                 routing_dataset_full_gdf = consolidated_matches.merge(
                     stores_proj.add_prefix('origin_'), 
                     left_on='store_index', right_index=True, how='left'
                 )
                 status_messages.append("Merged matched store details.")
            else:
                 st.error("O-D Matching failed to produce results.")
                 raise ValueError("O-D Matching produced no results.")
                 
        except Exception as e:
            st.error(f"Error during O-D matching ({matching_method}): {e}")
            status_messages.append(f"ERROR: O-D matching failed: {e}")
            module_error = True
            routing_dataset_full_gdf = gpd.GeoDataFrame() # Ensure empty on error

    # --- 3. Finalize Datasets & Calculate Distances --- 
    if not module_error and not routing_dataset_full_gdf.empty:
        st.markdown("--- *Finalizing Datasets & Calculating Distances* ---")
        try:
            # Calculate straight-line distance in meters (using projected coords)
            # Ensure both origin and destination geometries exist and are not empty
            valid_geoms = (
                routing_dataset_full_gdf['origin_geometry'].notna() & 
                routing_dataset_full_gdf['geometry'].notna() & 
                (~routing_dataset_full_gdf['origin_geometry'].is_empty) & 
                (~routing_dataset_full_gdf.geometry.is_empty)
            )
            routing_dataset_full_gdf['distance_m'] = np.nan # Initialize with NaN
            if valid_geoms.any(): # Check if there are any valid pairs before trying distance
                routing_dataset_full_gdf.loc[valid_geoms, 'distance_m'] = (
                    routing_dataset_full_gdf.loc[valid_geoms].geometry.distance(
                        routing_dataset_full_gdf.loc[valid_geoms, 'origin_geometry']
                    ).round(2)
                )
            status_messages.append("Calculated straight-line distances (m), handling missing/empty origins/destinations.")

            # Create LineString geometry for visualization, checking for valid, non-empty points
            routing_dataset_full_gdf['od_line'] = routing_dataset_full_gdf.apply(
                lambda row: LineString([row['origin_geometry'], row['geometry']]) 
                          if row['origin_geometry'] and not row['origin_geometry'].is_empty and 
                             row['geometry'] and not row['geometry'].is_empty 
                          else None, 
                axis=1
            )
            status_messages.append("Created O-D LineString geometries, handling missing/empty points.")
            
            # Filter out rows with missing lines before setting geometry and reprojecting
            valid_lines_gdf = routing_dataset_full_gdf[routing_dataset_full_gdf['od_line'].notna()].copy()
            if not valid_lines_gdf.empty:
                valid_lines_gdf = valid_lines_gdf.set_geometry('od_line') # Set line as active geometry
                valid_lines_gdf = valid_lines_gdf.to_crs(WGS84) # Convert back to WGS84
            else:
                 st.warning("No valid O-D lines could be generated after matching.")
                 # Define empty GDF with expected columns and geometry column name
                 expected_cols = list(routing_dataset_full_gdf.columns) # Get columns from original df
                 if 'od_line' not in expected_cols: expected_cols.append('od_line') # Ensure geometry col exists
                 valid_lines_gdf = gpd.GeoDataFrame(columns=expected_cols, geometry='od_line', crs=WGS84)

            # --- Create Final Routing CSV for specific hour --- 
            target_hour = od_config.get('simulation_hour_for_matching', 15)
            status_messages.append(f"Filtering final dataset for target hour: {target_hour}")
            st.write(f"Filtering final routing dataset for hour {target_hour}...")
            
            # Ensure timestamp is datetime in the GDF containing valid lines
            if not valid_lines_gdf.empty:
                valid_lines_gdf['timestamp'] = pd.to_datetime(valid_lines_gdf['timestamp'])
                routing_dataset_target_hour = valid_lines_gdf[valid_lines_gdf['timestamp'].dt.hour == target_hour].copy()
            else:
                routing_dataset_target_hour = gpd.GeoDataFrame() # Start with empty if no valid lines
            
            if routing_dataset_target_hour.empty:
                st.warning(f"No deliveries found for the target hour ({target_hour}). Final CSV will be empty.")
                status_messages.append(f"WARN: No deliveries found for target hour {target_hour}.")
                routing_dataset_df = pd.DataFrame(columns=['order_id', 'timestamp', 'origin_id', 'origin_coordinates', 'destination_id', 'destination_coordinates'])
            else:
                status_messages.append(f"Filtered {len(routing_dataset_target_hour)} deliveries for target hour {target_hour}.")
                # Prepare final columns
                routing_dataset_df = pd.DataFrame({
                    'order_id': routing_dataset_target_hour['order_id'],
                    'timestamp': routing_dataset_target_hour['timestamp'].astype(int) // 10**9, # Convert to Unix timestamp (seconds)
                    'origin_id': routing_dataset_target_hour['origin_store_id'],
                    # Robust coordinate extraction for origin (Point)
                    'origin_coordinates': routing_dataset_target_hour['origin_geometry'].apply(
                        lambda p: [round(p.y, 6), round(p.x, 6)] if p and not p.is_empty else [None, None]
                    ), # Lat, Lon
                    'destination_id': routing_dataset_target_hour['building_unique_id'],
                    # Robust coordinate extraction for destination (from LineString)
                    'destination_coordinates': routing_dataset_target_hour.geometry.apply(
                        lambda line: [round(line.coords[1][1], 6), round(line.coords[1][0], 6)] if line and not line.is_empty and len(line.coords) > 1 else [None, None]
                    ) # Extract dest point (lat, lon)
                })

            # Save Final Routing CSV
            try:
                routing_csv_path = os.path.join(data_subdir, 'routing_dataset.csv')
                routing_dataset_df.to_csv(routing_csv_path, index=False)
                status_messages.append(f"Saved: {os.path.basename(routing_csv_path)}")
            except Exception as e_save:
                st.warning(f"Could not save routing_dataset.csv: {e_save}")
                status_messages.append(f"WARN: Failed to save {os.path.basename(routing_csv_path)}: {e_save}")
                
            # Save Full Detailed GeoJSON (Optional)
            # Consider adding a config flag for this? It can be large.
            save_full_geojson = od_config.get('save_full_detailed_geojson', False) # Use config flag
            if save_full_geojson:
                try:
                    routing_geojson_path = os.path.join(data_subdir, 'routing_dataset_detailed_weekly.geojson')
                    # Prepare for saving - drop complex intermediate columns, convert types
                    cols_to_drop = ['geometry', 'origin_geometry', 'store_index', 'assigned_store_type'] # Drop original points, index
                    # Convert remaining object cols (e.g., dicts/lists from earlier modules) to str
                    gdf_to_save = routing_dataset_full_gdf.drop(columns=cols_to_drop, errors='ignore').copy()
                    for col in gdf_to_save.select_dtypes(include=['object']).columns:
                         try: gdf_to_save[col] = gdf_to_save[col].astype(str) 
                         except Exception: pass # Ignore conversion errors 
                         
                    gdf_to_save.to_file(routing_geojson_path, driver="GeoJSON")
                    status_messages.append(f"Saved: {os.path.basename(routing_geojson_path)}")
                except Exception as e_save:
                    st.warning(f"Could not save routing_dataset_detailed_weekly.geojson: {e_save}")
                    status_messages.append(f"WARN: Failed to save {os.path.basename(routing_geojson_path)}: {e_save}")
                    routing_dataset_full_gdf = gpd.GeoDataFrame() # Ensure empty if save failed
            else:
                 routing_dataset_full_gdf = gpd.GeoDataFrame() # Don't return it if not saved

        except Exception as e:
            st.error(f"Error finalizing datasets or calculating distances: {e}")
            status_messages.append(f"ERROR: Finalizing datasets failed: {e}")
            module_error = True
            routing_dataset_df = pd.DataFrame()
            routing_dataset_full_gdf = gpd.GeoDataFrame()

    # --- 4. Calculate and Save Statistics --- 
    if not module_error:
        st.markdown("--- *Calculating O-D Matching Statistics* ---")
        try:
            total_matched_pairs = len(routing_dataset_full_gdf)
            stats_f = {"Total Matched O-D Pairs": total_matched_pairs}
            
            if total_matched_pairs > 0:
                dist_stats = routing_dataset_full_gdf['distance_m'].describe()
                stats_f["Avg Distance (m)"] = f"{dist_stats.get('mean', 0):.1f}"
                stats_f["Median Distance (m)"] = f"{dist_stats.get('50%', 0):.1f}"
                stats_f["Min Distance (m)"] = f"{dist_stats.get('min', 0):.1f}"
                stats_f["Max Distance (m)"] = f"{dist_stats.get('max', 0):.1f}"
                
                # Counts per origin store
                origin_counts = routing_dataset_full_gdf['origin_store_id'].value_counts()
                stats_f["Avg Deliveries per Origin"] = f"{origin_counts.mean():.2f}"
                stats_f["Max Deliveries from Single Origin"] = origin_counts.max()
                # Could add top N origins if needed
                
                # Counts per destination building
                dest_counts = routing_dataset_full_gdf['building_unique_id'].value_counts()
                stats_f["Avg Deliveries per Destination"] = f"{dest_counts.mean():.2f}"
                stats_f["Max Deliveries to Single Destination"] = dest_counts.max()

            stats_f_df = pd.DataFrame.from_dict(stats_f, orient='index', columns=['Value'])
            stats_f_path = os.path.join(stats_subdir, 'od_matching_stats.csv')
            stats_f_df.to_csv(stats_f_path)
            status_messages.append(f"Saved: {os.path.basename(stats_f_path)}")
            st.write("O-D Matching Statistics:")
            st.dataframe(stats_f_df)
            
        except Exception as e_stat:
            st.warning(f"Could not calculate/save O-D stats: {e_stat}")
            status_messages.append(f"WARN: O-D stats calculation failed: {e_stat}")
            stats_f_df = pd.DataFrame() # Ensure empty on error

    # --- 5. Generate Visualizations (Static Map & Animation Frames) --- 
    if not module_error:
        st.markdown("--- *Generating O-D Visualizations* ---")
        # --- Static O-D Map (for target hour) --- 
        if not routing_dataset_target_hour.empty:
            try:
                st.write(f"Generating Static O-D Map (Hour: {target_hour})...")
                fig_odmap, ax_odmap = plt.subplots(figsize=(10, 10))
                
                plot_gdf = routing_dataset_target_hour.to_crs(WEB_MERCATOR) # Use constant
                
                # Plot O-D lines
                plot_gdf.plot(ax=ax_odmap, color='purple', linewidth=0.5, alpha=0.4)
                
                # Plot Origins (stores used in this hour)
                origin_ids = plot_gdf['origin_store_id'].unique()
                origins_plot = stores_proj[stores_proj['store_id'].isin(origin_ids)].to_crs(WEB_MERCATOR) # Use constant
                if not origins_plot.empty:
                    origins_plot.plot(ax=ax_odmap, color='red', marker='*', markersize=60, label='Origins (used this hour)') # Adjusted label
                
                # Plot Destinations (buildings receiving in this hour)
                dest_ids = plot_gdf['building_unique_id'].unique()
                # Need to load buildings_with_demand? Or just use event points?
                # Using event points for simplicity now
                dest_points = gpd.GeoDataFrame(
                     plot_gdf.drop_duplicates(subset=['building_unique_id']),
                     geometry=gpd.points_from_xy(plot_gdf.longitude, plot_gdf.latitude),
                     crs=WGS84
                ).to_crs(WEB_MERCATOR) # Use constant
                if not dest_points.empty:
                     dest_points.plot(ax=ax_odmap, color='blue', marker='o', markersize=10, alpha=0.7, label='Destinations (this hour)') # Adjusted label

                try:
                    cx.add_basemap(ax_odmap, crs=plot_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
                except Exception as e_base:
                    st.warning(f"Basemap for O-D Map failed: {e_base}")
                
                style_map(ax_odmap, f'O-D Pairs (Hour: {target_hour})')
                # Add legend if needed
                ax_odmap.legend(fontsize=9) # Added legend back
                odmap_path = os.path.join(viz_subdir, 'od_map_static.png')
                plt.savefig(odmap_path, dpi=150, bbox_inches='tight')
                plt.close(fig_odmap)
                status_messages.append(f"Saved: {os.path.basename(odmap_path)}")
            except Exception as e_fig:
                st.warning(f"Could not generate/save Static O-D Map: {e_fig}")
                status_messages.append(f"WARN: Static O-D Map generation failed: {e_fig}")
        else:
            status_messages.append(f"Skipped Static O-D Map - No deliveries in target hour {target_hour}.")

        # --- O-D Animation Frames --- 
        if not routing_dataset_full_gdf.empty:
            st.write("Generating O-D Animation Frames (this might take time)...")
            anim_prog = st.progress(0.0)
            try:
                # Use full dataset projected to UTM
                anim_gdf = routing_dataset_full_gdf.to_crs(utm_crs)
                anim_gdf['hour_of_sim'] = (pd.to_datetime(anim_gdf['timestamp']) - pd.to_datetime(anim_gdf['timestamp']).min()).dt.total_seconds() // 3600
                max_hour = int(anim_gdf['hour_of_sim'].max())
                hourly_groups = anim_gdf.groupby('hour_of_sim')
                status_messages.append(f"Generating {max_hour + 1} hourly O-D animation frames...")

                # Get origin points projected
                origins_anim_plot = stores_proj.copy()

                # Create frames hour by hour
                for hour in range(max_hour + 1):
                    fig_anim_od, ax_anim_od = plt.subplots(figsize=(10, 10))
                    
                    # Plot origins
                    origins_anim_plot.plot(ax=ax_anim_od, color='red', marker='*', markersize=40, alpha=0.8)
                    
                    # Plot O-D lines *for this specific hour*
                    if hour in hourly_groups.groups:
                        hour_data = hourly_groups.get_group(hour)
                        if not hour_data.empty:
                            hour_data.set_geometry('od_line').to_crs(WEB_MERCATOR).plot( # Use constant
                                ax=ax_anim_od, color='purple', linewidth=0.7, alpha=0.5
                            )
                    
                    # Add basemap and style
                    try: cx.add_basemap(ax_anim_od, crs=origins_anim_plot.crs.to_string(), source=cx.providers.CartoDB.Positron) 
                    except Exception: pass # Ignore basemap errors for frames
                    style_map(ax_anim_od, f'O-D Pairs (Hour: {hour:03d})')
                    ax_anim_od.set_xticks([])
                    ax_anim_od.set_yticks([])
                    
                    # Save frame
                    frame_path = os.path.join(anim_subdir, f'od_frame_{hour:03d}.png')
                    plt.savefig(frame_path, dpi=100, bbox_inches='tight') # Lower DPI for frames
                    plt.close(fig_anim_od)
                    
                    anim_prog.progress((hour + 1) / (max_hour + 1))
                
                status_messages.append(f"Finished generating O-D animation frames in {anim_subdir}.")
                anim_prog.progress(1.0)

            except Exception as e_anim:
                st.error(f"Error generating O-D animation frames: {e_anim}")
                status_messages.append(f"ERROR: O-D animation frame generation failed: {e_anim}")
        else:
             status_messages.append("Skipped O-D Animation Frames - No matched O-D pairs.")

    # --- Completion --- 
    module_F_time = time.time() - module_F_start
    status_messages.append(f"Module F completed in {module_F_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module F finished in {module_F_time:.2f}s.")
    else:
        st.error(f"Module F finished with errors in {module_F_time:.2f}s.")

    # --- Return Results --- 
    return {
        "routing_dataset_df": routing_dataset_df, 
        "routing_dataset_full_gdf": routing_dataset_full_gdf, 
        "stats_f_df": stats_f_df,
        "status_messages": status_messages,
        "error": module_error
    } 