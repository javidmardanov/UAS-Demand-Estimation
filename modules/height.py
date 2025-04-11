# modules/height.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import json
import re # For parsing height tags
import streamlit as st
import matplotlib.pyplot as plt
import contextily as cx
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Import utilities
try:
    from utils.helpers import save_empty_gdf, save_empty_df, style_map
    from utils.crs import estimate_utm_crs # Might need this if UTM not passed
except ImportError:
    st.error("Failed to import utility functions. Ensure utils/helpers.py and utils/crs.py exist.")
    def save_empty_gdf(f, **kwargs): pass
    def save_empty_df(f, **kwargs): pass
    def style_map(ax, title): pass
    def estimate_utm_crs(gdf): return 'EPSG:3857' # Fallback

WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857' # Added constant

# --- Helper Functions for Height Extraction/Estimation (Adapted from Colab) ---

def extract_known_height(row, meters_per_level):
    """Extracts height from OSM tags if available."""
    height = np.nan
    source = 'unknown'

    # 1. Try 'height' tag (parse numeric value)
    if pd.notna(row.get('height')):
        try:
            # Use regex to find the first number (int or float)
            match = re.search(r'[+-]?(\d*\.)?\d+', str(row['height']))
            if match:
                height = float(match.group(0))
                source = 'height_tag'
        except Exception:
            pass # Ignore parsing errors

    # 2. Try 'building:levels' tag if height not found
    if pd.isna(height) and pd.notna(row.get('building:levels')):
        try:
            # Use regex to find the first number (int or float)
            match = re.search(r'[+-]?(\d*\.)?\d+', str(row['building:levels']))
            if match:
                 levels = float(match.group(0))
                 # Apply sanity check for levels (e.g., > 0 and < 200)
                 if 0 < levels < 200:
                     height = levels * meters_per_level
                     source = 'levels_tag'
        except Exception:
            pass # Ignore parsing errors
            
    # Apply sanity check for height (e.g., > 0 and < 1000m)
    if pd.notna(height) and not (0 < height < 1000):
        height = np.nan # Discard unreasonable values
        source = 'unknown' # Reset source if height is discarded

    return pd.Series([height, source], index=['height_m', 'height_source'])

# --- Main Module Function ---
def run_module_c(config, classified_buildings_gdf, union_gdf, utm_crs):
    """Executes Module C: Height Estimation.

    Args:
        config (dict): The configuration dictionary.
        classified_buildings_gdf (gpd.GeoDataFrame): Buildings from Module B.
        union_gdf (gpd.GeoDataFrame): Union of tracts boundary from Module A.
        utm_crs (str): Estimated UTM CRS from Module A.

    Returns:
        dict: A dictionary containing results:
              'buildings_with_heights_gdf', 'stats_c_df', 
              'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module C: Building Height Estimation")
    st.markdown("--- *Initializing* ---")
    module_C_start = time.time()
    height_config = config['height_estimation']
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")

    # Initialize return values
    buildings_with_heights_gdf = gpd.GeoDataFrame()
    stats_c_df = pd.DataFrame()
    status_messages = []
    module_error = False

    # --- Check Inputs --- 
    if classified_buildings_gdf is None or classified_buildings_gdf.empty:
        st.warning("Module C: Input classified buildings GeoDataFrame is empty. Skipping height estimation.")
        status_messages.append("WARN: Skipping Module C - No classified buildings provided.")
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson"))
        save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'))
        return {
            "buildings_with_heights_gdf": buildings_with_heights_gdf, "stats_c_df": stats_c_df,
            "status_messages": status_messages, "error": False
        }
    if union_gdf is None or union_gdf.empty:
        st.warning("Module C: Input union GeoDataFrame is empty. Cannot filter buildings. Skipping height estimation.")
        status_messages.append("WARN: Skipping Module C - No union boundary provided.")
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson"))
        save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'))
        return {
            "buildings_with_heights_gdf": buildings_with_heights_gdf, "stats_c_df": stats_c_df,
            "status_messages": status_messages, "error": False
        }
    if not utm_crs:
         st.warning("Module C: UTM CRS not provided. Needed for KNN features. Skipping k-NN estimation.")
         status_messages.append("WARN: Skipping k-NN height estimation - UTM CRS missing.")
         # Allow proceeding without KNN?
         # module_error = True # Set error if KNN is critical

    # --- 1. Filter Buildings to Union Area --- 
    st.markdown("--- *Filtering Buildings to Union Area* ---")
    try:
        buildings_in_union_gdf = gpd.clip(classified_buildings_gdf, union_gdf)
        # Keep only buildings whose representative point is within the union (handles edge cases)
        # Need to project for representative_point()
        if not utm_crs:
            st.warning("Estimating temporary UTM for filtering, original UTM preferred.")
            temp_utm = estimate_utm_crs(buildings_in_union_gdf)
        else:
            temp_utm = utm_crs
        
        if temp_utm:
             buildings_in_union_proj = buildings_in_union_gdf.to_crs(temp_utm)
             union_proj = union_gdf.to_crs(temp_utm)
             buildings_in_union_proj['repr_point'] = buildings_in_union_proj.geometry.representative_point()
             within_mask = buildings_in_union_proj.repr_point.within(union_proj.geometry.iloc[0])
             buildings_in_union_gdf = buildings_in_union_gdf.loc[within_mask[within_mask].index]
             del buildings_in_union_proj # Free memory
             status_messages.append(f"Filtered to {len(buildings_in_union_gdf)} buildings within the union of tracts.")
        else:
             st.warning("Could not create projected GDF for precise filtering. Using simple clip.")
             status_messages.append(f"Using {len(buildings_in_union_gdf)} buildings from simple clip (might include edge overlaps).")

        if buildings_in_union_gdf.empty:
            st.warning("Module C: No buildings found within the union area. Skipping height estimation.")
            status_messages.append("WARN: Skipping Module C - No buildings found in union area.")
            save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson"))
            save_empty_df(os.path.join(stats_subdir, 'height_stats.csv'))
            return {
                "buildings_with_heights_gdf": buildings_with_heights_gdf, "stats_c_df": stats_c_df,
                "status_messages": status_messages, "error": False
            }
        
        # Ensure a copy for modifications
        buildings_to_process = buildings_in_union_gdf.copy()

    except Exception as e:
        st.error(f"Error filtering buildings to union area: {e}")
        status_messages.append(f"ERROR: Filtering buildings failed: {e}")
        module_error = True
        buildings_to_process = gpd.GeoDataFrame() # Ensure empty on error

    # --- 2. Extract Known Heights from Tags --- 
    if not module_error:
        st.markdown("--- *Extracting Heights from OSM Tags* ---")
        try:
            meters_per_level = height_config.get('meters_per_level', 3.5)
            status_messages.append(f"Using {meters_per_level} m/level for 'building:levels' tag.")
            
            # Apply the helper function row-wise
            extracted_heights = buildings_to_process.apply(
                lambda row: extract_known_height(row, meters_per_level),
                axis=1
            )
            buildings_to_process[['height_m', 'height_source']] = extracted_heights
            
            known_count = buildings_to_process['height_m'].notna().sum()
            status_messages.append(f"Extracted known height for {known_count} buildings from OSM tags.")
            st.write(f"Extracted known height for {known_count} buildings.")

        except Exception as e:
            st.error(f"Error extracting known heights: {e}")
            status_messages.append(f"ERROR: Extracting known heights failed: {e}")
            module_error = True
            # Initialize columns if they failed to create
            if 'height_m' not in buildings_to_process.columns:
                 buildings_to_process['height_m'] = np.nan
            if 'height_source' not in buildings_to_process.columns:
                 buildings_to_process['height_source'] = 'unknown'

    # --- 3. Estimate Missing Heights with k-NN (Optional) --- 
    knn_enabled = height_config.get('use_knn_estimation', False)
    if not module_error and knn_enabled:
        st.markdown("--- *Estimating Missing Heights via k-NN* ---")
        knn_details = {}
        try:
            if not utm_crs:
                st.warning("Skipping k-NN estimation: UTM CRS is required but missing.")
                status_messages.append("WARN: Skipped k-NN - UTM CRS missing.")
            else:
                st.write("Preparing data for k-NN...")
                # Prepare data: Need projected coordinates and area
                buildings_proj = buildings_to_process.to_crs(utm_crs)
                buildings_proj['building_area_m2'] = buildings_proj.geometry.area
                buildings_proj['centroid_x'] = buildings_proj.geometry.centroid.x
                buildings_proj['centroid_y'] = buildings_proj.geometry.centroid.y

                # Separate known and unknown heights
                known_height_df = buildings_proj[buildings_proj['height_m'].notna()].copy()
                unknown_height_df = buildings_proj[buildings_proj['height_m'].isna()].copy()

                status_messages.append(f"{len(known_height_df)} buildings with known height available for training.")
                status_messages.append(f"{len(unknown_height_df)} buildings require height estimation.")
                
                # Check if enough data to train
                min_train_samples = height_config.get('knn_min_samples_train', 30)
                if len(known_height_df) < min_train_samples:
                    st.warning(f"Skipping k-NN: Insufficient buildings with known height ({len(known_height_df)} < {min_train_samples}) for training.")
                    status_messages.append(f"WARN: Skipped k-NN - Insufficient training data ({len(known_height_df)} < {min_train_samples}).")
                elif unknown_height_df.empty:
                     st.info("Skipping k-NN: No buildings require height estimation.")
                     status_messages.append("INFO: Skipped k-NN - All building heights known.")
                else:
                    # Define features based on config
                    features = ['centroid_x', 'centroid_y']
                    if height_config.get('knn_use_area_feature', True):
                        features.append('building_area_m2')
                    target = 'height_m'
                    status_messages.append(f"Using k-NN features: {features}")

                    X_known = known_height_df[features]
                    y_known = known_height_df[target]
                    X_unknown = unknown_height_df[features]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_known_scaled = scaler.fit_transform(X_known)
                    X_unknown_scaled = scaler.transform(X_unknown)

                    # Train KNN model
                    n_neighbors = height_config.get('knn_n_neighbors', 5)
                    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
                    st.write(f"Training k-NN model (k={n_neighbors})...")
                    knn.fit(X_known_scaled, y_known)
                    
                    # Predict heights for unknown buildings
                    st.write("Predicting heights for remaining buildings...")
                    predicted_heights = knn.predict(X_unknown_scaled)
                    
                    # Apply predicted heights and source
                    buildings_to_process.loc[unknown_height_df.index, 'height_m'] = predicted_heights
                    buildings_to_process.loc[unknown_height_df.index, 'height_source'] = 'knn_estimated'
                    knn_estimated_count = len(unknown_height_df)
                    status_messages.append(f"Estimated height for {knn_estimated_count} buildings using k-NN.")
                    
                    # Optional: Calculate and store KNN performance metrics (e.g., on a test split)
                    try: 
                        X_train, X_test, y_train, y_test = train_test_split(X_known_scaled, y_known, test_size=0.2, random_state=config.get('random_seed', 42))
                        if len(X_train) >= n_neighbors and len(X_test) > 0: # Ensure test set is usable
                             knn_test = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
                             y_pred_test = knn_test.predict(X_test)
                             rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                             knn_details['RMSE (test split)'] = f"{rmse:.2f}"
                             knn_details['Training Samples'] = len(X_train)
                             knn_details['Test Samples'] = len(X_test)
                             status_messages.append(f"k-NN RMSE on test split: {rmse:.2f}m")
                    except Exception as e_test:
                        st.warning(f"Could not evaluate k-NN performance: {e_test}")
        
        except Exception as e:
            st.error(f"Error during k-NN height estimation: {e}")
            status_messages.append(f"ERROR: k-NN estimation failed: {e}")
            # Don't set module_error=True? Allow fallback to default?
        
        # Store KNN details if any were generated
        if knn_details:
             buildings_to_process['knn_details'] = buildings_to_process.index.map(lambda x: json.dumps(knn_details) if x in unknown_height_df.index else None)
    
    elif knn_enabled:
        status_messages.append("INFO: Skipping k-NN estimation due to previous module error.")

    # --- 4. Apply Default Height for Remaining Unknowns --- 
    if not module_error:
        st.markdown("--- *Applying Default Heights* ---")
        try:
            unknown_mask = buildings_to_process['height_m'].isna()
            remaining_unknown_count = unknown_mask.sum()
            
            if remaining_unknown_count > 0:
                default_height = height_config.get('default_height_m', 5.0)
                status_messages.append(f"Applying default height ({default_height}m) to {remaining_unknown_count} buildings.")
                st.write(f"Applying default height ({default_height}m) to {remaining_unknown_count} buildings.")
                buildings_to_process.loc[unknown_mask, 'height_m'] = default_height
                buildings_to_process.loc[unknown_mask, 'height_source'] = 'default'
            else:
                status_messages.append("No buildings remaining require default height.")

            # Final check for any NaNs (shouldn't happen but good practice)
            final_nan_count = buildings_to_process['height_m'].isna().sum()
            if final_nan_count > 0:
                 st.warning(f"Found {final_nan_count} buildings still missing height after all steps. Filling with 1m.")
                 status_messages.append(f"WARN: Filling {final_nan_count} residual missing heights with 1m.")
                 buildings_to_process['height_m'].fillna(1.0, inplace=True)
                 buildings_to_process['height_source'].fillna('fallback_1m', inplace=True)
            
            # Ensure height is positive
            buildings_to_process['height_m'] = buildings_to_process['height_m'].clip(lower=0.1) # Ensure positive height

        except Exception as e:
            st.error(f"Error applying default heights: {e}")
            status_messages.append(f"ERROR: Applying default heights failed: {e}")
            module_error = True

    # --- 5. Save Results --- 
    if not module_error:
        st.markdown("--- *Saving Buildings with Heights* ---")
        buildings_with_heights_gdf = buildings_to_process # Assign the final GDF
        try:
            # Prepare for saving (handle complex types)
            save_columns = buildings_with_heights_gdf.columns.tolist()
            for col in save_columns:
                # Check if column dtype is object and contains lists/dicts, convert to string
                 if buildings_with_heights_gdf[col].dtype == 'object':
                    is_list_or_dict = buildings_with_heights_gdf[col].apply(lambda x: isinstance(x, (list, dict))).any()
                    if is_list_or_dict:
                        st.write(f"Converting object column '{col}' to string for saving.")
                        buildings_with_heights_gdf[col] = buildings_with_heights_gdf[col].astype(str)
            
            height_file = os.path.join(data_subdir, "buildings_with_heights.geojson")
            if not buildings_with_heights_gdf.empty:
                 buildings_with_heights_gdf.to_file(height_file, driver="GeoJSON")
            else:
                 save_empty_gdf(height_file)
            status_messages.append(f"Saved: {os.path.basename(height_file)}")
        except Exception as e_save:
            st.warning(f"Could not save buildings_with_heights.geojson: {e_save}")
            status_messages.append(f"WARN: Failed to save {os.path.basename(height_file)}: {e_save}")
            buildings_with_heights_gdf = gpd.GeoDataFrame() # Ensure empty GDF if save failed
    else:
        # If error occurred earlier, ensure we save an empty file
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_heights.geojson"))
        buildings_with_heights_gdf = gpd.GeoDataFrame()

    # --- 6. Calculate and Save Statistics --- 
    if not module_error:
        st.markdown("--- *Calculating Height Statistics* ---")
        try:
            total_buildings_in_union = len(buildings_with_heights_gdf)
            stats_c = {"Total Buildings (Union)": total_buildings_in_union}
            
            if total_buildings_in_union > 0:
                # Counts by source
                source_counts = buildings_with_heights_gdf['height_source'].value_counts()
                for source, count in source_counts.items():
                    stats_c[f"Count - {source}"] = count
                
                # Descriptive stats for height
                height_stats = buildings_with_heights_gdf['height_m'].describe()
                stats_c["Height Mean (m)"] = f"{height_stats.get('mean', 0):.2f}"
                stats_c["Height Median (m)"] = f"{height_stats.get('50%', 0):.2f}"
                stats_c["Height Std Dev (m)"] = f"{height_stats.get('std', 0):.2f}"
                stats_c["Height Min (m)"] = f"{height_stats.get('min', 0):.2f}"
                stats_c["Height Max (m)"] = f"{height_stats.get('max', 0):.2f}"
            
            stats_c_df = pd.DataFrame.from_dict(stats_c, orient='index', columns=['Value'])
            stats_c_path = os.path.join(stats_subdir, 'height_stats.csv')
            stats_c_df.to_csv(stats_c_path)
            status_messages.append(f"Saved: {os.path.basename(stats_c_path)}")
            st.write("Height Estimation Statistics:")
            st.dataframe(stats_c_df)
            
            # Optional: Plot height distribution
            try:
                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                buildings_with_heights_gdf['height_m'].plot(kind='hist', bins=50, ax=ax_hist, title='Distribution of Estimated Building Heights')
                ax_hist.set_xlabel("Height (m)")
                plt.tight_layout()
                hist_path = os.path.join(viz_subdir, 'height_distribution.png')
                plt.savefig(hist_path, dpi=100)
                plt.close(fig_hist)
                status_messages.append(f"Saved: {os.path.basename(hist_path)}")
            except Exception as e_hist:
                 st.warning(f"Could not generate height distribution plot: {e_hist}")

        except Exception as e_stat:
            st.warning(f"Could not calculate/save height stats: {e_stat}")
            status_messages.append(f"WARN: Height stats calculation failed: {e_stat}")
            stats_c_df = pd.DataFrame() # Ensure empty on error

    # --- 7. Generate Visualization (Height Map) --- 
    if not module_error and not buildings_with_heights_gdf.empty:
        st.markdown("--- *Generating Height Map* ---")
        try:
            st.write("Generating Height Map...")
            fig_hmap, ax_hmap = plt.subplots(figsize=(10, 10))
            
            plot_gdf = buildings_with_heights_gdf.to_crs(WEB_MERCATOR) # Use constant
            plot_gdf.plot(column='height_m', cmap='viridis', legend=True,
                          ax=ax_hmap, 
                          legend_kwds={'label': "Estimated Height (m)",
                                       'orientation': "horizontal",
                                       'shrink': 0.5})
            
            # Add union boundary for context
            if not union_gdf.empty:
                 union_gdf.to_crs(WEB_MERCATOR).boundary.plot(ax=ax_hmap, color='red', linewidth=1, linestyle=':') # Use constant

            try:
                cx.add_basemap(ax_hmap, crs=plot_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
            except Exception as e_base:
                st.warning(f"Basemap for Height Map failed: {e_base}")
            
            style_map(ax_hmap, 'Estimated Building Heights (within Union)')
            hmap_path = os.path.join(viz_subdir, 'height_map.png')
            plt.savefig(hmap_path, dpi=150, bbox_inches='tight')
            plt.close(fig_hmap)
            status_messages.append(f"Saved: {os.path.basename(hmap_path)}")
        except Exception as e_fig:
            st.warning(f"Could not generate/save Height Map: {e_fig}")
            status_messages.append(f"WARN: Height Map generation failed: {e_fig}")

    # --- Completion --- 
    module_C_time = time.time() - module_C_start
    status_messages.append(f"Module C completed in {module_C_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module C finished in {module_C_time:.2f}s.")
    else:
        st.error(f"Module C finished with errors in {module_C_time:.2f}s.")

    # --- Return Results --- 
    return {
        "buildings_with_heights_gdf": buildings_with_heights_gdf,
        "stats_c_df": stats_c_df,
        "status_messages": status_messages,
        "error": module_error
    } 