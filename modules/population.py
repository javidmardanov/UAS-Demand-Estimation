# modules/population.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import json
import streamlit as st
import matplotlib.pyplot as plt
import contextily as cx

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

# --- Main Module Function ---
def run_module_d(config, buildings_with_heights_gdf, census_data_df, tracts_gdf):
    """Executes Module D: Population Allocation.

    Args:
        config (dict): The configuration dictionary.
        buildings_with_heights_gdf (gpd.GeoDataFrame): Buildings with height info from Module C.
        census_data_df (pd.DataFrame): Filtered Census demographic data from Module A.
        tracts_gdf (gpd.GeoDataFrame): Census Tract boundaries from Module A.

    Returns:
        dict: A dictionary containing results:
              'buildings_with_population_gdf', 'stats_d_df',
              'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module D: Population Allocation")
    st.markdown("--- *Initializing* ---")
    module_D_start = time.time()
    pop_config = config['population_allocation']
    census_vars = config['data_acquisition']['census_variables'] # Get var names from config
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")

    # Initialize return values
    buildings_with_population_gdf = gpd.GeoDataFrame()
    stats_d_df = pd.DataFrame()
    status_messages = []
    module_error = False

    # Define expected column names from Census data (based on typical config)
    pop_col = 'B01003_001E' # Total Population
    hh_size_col = 'B25010_001E' # Average Household Size

    # --- Check Inputs --- 
    if buildings_with_heights_gdf is None or buildings_with_heights_gdf.empty:
        st.warning("Module D: Input buildings GeoDataFrame is empty. Skipping.") # Simplified warning
        module_error = True
    # Removed check for GEOID here, will be added via spatial join
    elif not all(col in buildings_with_heights_gdf.columns for col in ['unique_id', 'height_m', 'geometry']): 
         st.error("Module D: Input buildings GDF missing required columns from Module C (unique_id, height_m, geometry). Skipping.")
         module_error = True
    elif 'residential' not in buildings_with_heights_gdf.columns:
         st.error("Module D: Input buildings GDF missing 'residential' column. Skipping population allocation.")
         status_messages.append("ERROR: Skipping Module D - Missing 'residential' column in input buildings.")
         module_error = True
    elif 'height_m' not in buildings_with_heights_gdf.columns:
         st.error("Module D: Input buildings GDF missing 'height_m' column. Skipping population allocation.")
         status_messages.append("ERROR: Skipping Module D - Missing 'height_m' column in input buildings.")
         module_error = True
    
    if census_data_df is None or census_data_df.empty:
        st.warning("Module D: Input Census data DataFrame is empty. Skipping population allocation.")
        status_messages.append("WARN: Skipping Module D - No Census data provided.")
        module_error = True
    elif pop_col not in census_data_df.columns:
        st.error(f"Module D: Input Census data missing Population column '{pop_col}'. Check config/CSV. Skipping.")
        status_messages.append(f"ERROR: Skipping Module D - Census data missing '{pop_col}'.")
        module_error = True
    # Household size is needed unless overridden
    elif hh_size_col not in census_data_df.columns and pop_config.get('avg_household_size_override') is None:
         st.error(f"Module D: Input Census data missing Avg Household Size '{hh_size_col}' and no override provided. Check config/CSV. Skipping.")
         status_messages.append(f"ERROR: Skipping Module D - Census data missing '{hh_size_col}' and no override.")
         module_error = True
         
    if tracts_gdf is None or tracts_gdf.empty:
        st.warning("Module D: Input tracts GeoDataFrame is empty. Skipping population allocation.")
        status_messages.append("WARN: Skipping Module D - No Census tracts provided.")
        module_error = True
    elif 'GEOID' not in tracts_gdf.columns:
        st.error("Module D: Input tracts GDF missing 'GEOID' column. Skipping population allocation.")
        status_messages.append("ERROR: Skipping Module D - Missing 'GEOID' column in input tracts.")
        module_error = True
    
    if not config.get('utm_crs'):
         st.warning("Module D: UTM CRS not provided. Needed for area calculation. Skipping.")
         status_messages.append("WARN: Skipping Module D - UTM CRS missing.")
         module_error = True

    # Save empty files and return if critical error found
    if module_error:
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_population.geojson"))
        save_empty_df(os.path.join(stats_subdir, 'population_stats.csv'))
        return {
            "buildings_with_population_gdf": buildings_with_population_gdf, "stats_d_df": stats_d_df,
            "status_messages": status_messages, "error": True
        }

    # --- 1. Prepare Data for Allocation --- 
    st.markdown("--- *Preparing Data for Population Allocation* ---")
    residential_buildings = gpd.GeoDataFrame() # Initialize
    try:
        # --- Add GEOID via Spatial Join --- 
        st.write("Spatially joining buildings to tracts to add GEOID...")
        if buildings_with_heights_gdf.crs != tracts_gdf.crs:
             # Assuming tracts_gdf is WGS84, project buildings GDF if needed
             st.warning(f"CRS mismatch between buildings ({buildings_with_heights_gdf.crs}) and tracts ({tracts_gdf.crs}). Reprojecting buildings to tracts CRS.")
             buildings_with_heights_gdf = buildings_with_heights_gdf.to_crs(tracts_gdf.crs)
        
        # Perform the spatial join (use 'intersects' for robustness)
        # Keep essential columns from buildings, add GEOID from tracts
        buildings_with_geoid = gpd.sjoin(
            buildings_with_heights_gdf[['unique_id', 'height_m', 'residential', 'geometry']], # Keep residential flag
            tracts_gdf[['GEOID', 'geometry']], 
            how='left', 
            predicate='intersects' # Or 'within'? Intersects might be safer.
        )
        # Handle buildings potentially overlapping multiple tracts - keep first match
        buildings_with_geoid = buildings_with_geoid[~buildings_with_geoid.index.duplicated(keep='first')]
        # Drop the spatial join index column
        if 'index_right' in buildings_with_geoid.columns:
            buildings_with_geoid = buildings_with_geoid.drop(columns=['index_right'])
        status_messages.append(f"Spatially joined {len(buildings_with_geoid)} buildings to tracts to assign GEOID.")
        
        # Check for buildings that didn't get a GEOID
        missing_geoid_count = buildings_with_geoid['GEOID'].isna().sum()
        if missing_geoid_count > 0:
            st.warning(f"{missing_geoid_count} buildings could not be assigned a GEOID via spatial join (potentially outside all tract boundaries). These buildings will be excluded from allocation.")
            status_messages.append(f"WARN: {missing_geoid_count} buildings missing GEOID after spatial join.")
            buildings_with_geoid = buildings_with_geoid.dropna(subset=['GEOID'])

        # --- Filter to Residential Buildings --- 
        if 'residential' not in buildings_with_geoid.columns:
             st.warning("Module D: 'residential' column not found in buildings GDF. Assuming ALL buildings are non-residential.")
             residential_buildings = buildings_with_geoid[buildings_with_geoid['residential'] == True].copy() # Will likely be empty
        else:
             residential_buildings = buildings_with_geoid[buildings_with_geoid['residential'] == True].copy()
        
        status_messages.append(f"Processing {len(residential_buildings)} residential buildings with GEOID.")
        st.write(f"Processing {len(residential_buildings)} residential buildings with GEOID.")
        
        if residential_buildings.empty:
             st.warning("Module D: No residential buildings found. Skipping population allocation.")
             status_messages.append("WARN: Skipping Module D - No residential buildings.")
             module_error = True # Treat as error for downstream?
             # Save empty files and return
             save_empty_gdf(os.path.join(data_subdir, "buildings_with_population.geojson"))
             save_empty_df(os.path.join(stats_subdir, 'population_stats.csv'))
             return { # Return empty results
                 "buildings_with_population_gdf": buildings_with_population_gdf,
                 "stats_d_df": stats_d_df,
                 "status_messages": status_messages,
                 "error": False # Not a code error, just no data to process
             }
             
        # Ensure valid geometries and calculate area
        residential_buildings = residential_buildings[residential_buildings.geometry.is_valid].copy()
        if residential_buildings.crs != config.get('utm_crs'):
            residential_buildings = residential_buildings.to_crs(config.get('utm_crs'))
        residential_buildings['building_area_m2'] = residential_buildings.geometry.area
        
        # Calculate building volume (ensure height and area >= 1 for stability)
        residential_buildings['height_m_alloc'] = residential_buildings['height_m'].clip(lower=1.0)
        residential_buildings['building_area_m2_alloc'] = residential_buildings['building_area_m2'].clip(lower=1.0)
        residential_buildings['building_volume_m3'] = residential_buildings['building_area_m2_alloc'] * residential_buildings['height_m_alloc']
        
        # Calculate area and volume for allocation
        utm_crs = config.get('utm_crs') # Get UTM from config if possible
        if not utm_crs:
            # Estimate if not provided (should come from Module A ideally)
            st.warning("UTM CRS not found in config, estimating based on centroid.")
            if not residential_buildings.empty:
                avg_lon = residential_buildings.geometry.centroid.x.mean()
                avg_lat = residential_buildings.geometry.centroid.y.mean()
                utm_zone = int(np.floor((avg_lon + 180) / 6) + 1)
                utm_crs = f"EPSG:326{utm_zone}" if avg_lat >= 0 else f"EPSG:327{utm_zone}"
            else: # Should not happen if check above passed
                 utm_crs = 'EPSG:3857' # Fallback, less accurate for area
            st.write(f"Estimated UTM CRS: {utm_crs}")
            status_messages.append(f"Estimated UTM CRS: {utm_crs} for area calculations.")
            
        residential_buildings_proj = residential_buildings.to_crs(utm_crs)
        residential_buildings_proj['building_area_m2'] = residential_buildings_proj.geometry.area
        residential_buildings_proj['building_volume_m3'] = residential_buildings_proj['building_area_m2'] * residential_buildings_proj['height_m_alloc']
        residential_buildings = residential_buildings_proj.copy() # Assign back with new columns
        del residential_buildings_proj
        status_messages.append("Calculated building area and volume for allocation.")
        
    except Exception as e_prep:
        st.error(f"Error preparing data for population allocation: {e_prep}")
        status_messages.append(f"ERROR: Data preparation failed: {e_prep}")
        module_error = True

    # --- 2. Allocate Population --- 
    if not module_error:
        st.markdown("--- *Allocating Population to Buildings* ---")
        try:
            pop_config = config['population_allocation']
            pop_col = 'B01003_001E' # Total Population
            hh_size_col = 'B25010_001E' # Avg Household Size
            scale_factor = pop_config.get('population_scale_factor', 1.0)
            avg_hh_override = pop_config.get('avg_household_size_override')

            # --- Perform Allocation Steps --- 
            st.write("Merging Census population data...")
            # Ensure required columns exist in census_data_df
            if pop_col not in census_data_df.columns:
                 raise ValueError(f"Required population column '{pop_col}' not found in census_data_df")
            if avg_hh_override is None and hh_size_col not in census_data_df.columns:
                 raise ValueError(f"Required household size column '{hh_size_col}' not found in census_data_df (and override not set)")
                 
            census_subset = census_data_df[['GEOID', pop_col] + ([hh_size_col] if avg_hh_override is None else [])].copy()
            census_subset[pop_col] = pd.to_numeric(census_subset[pop_col], errors='coerce').fillna(0).astype(int)
            if avg_hh_override is None:
                census_subset[hh_size_col] = pd.to_numeric(census_subset[hh_size_col], errors='coerce').fillna(1.0).clip(lower=0.1)

            buildings_to_allocate = residential_buildings.merge(census_subset, on='GEOID', how='left')
            buildings_to_allocate[pop_col] = buildings_to_allocate[pop_col].fillna(0)
            if avg_hh_override is None:
                 fallback_hh_size = census_subset[hh_size_col].mean() if not census_subset.empty else 1.0
                 buildings_to_allocate[hh_size_col].fillna(fallback_hh_size, inplace=True)
            status_messages.append("Merged Census data onto residential buildings.")
            
            # Calculate tract volume, volume share, initial pop, scaling factor, final pop
            tract_total_volume = buildings_to_allocate.groupby('GEOID')['building_volume_m3'].sum().reset_index()
            tract_total_volume.rename(columns={'building_volume_m3': 'tract_total_volume_m3'}, inplace=True)
            buildings_to_allocate = buildings_to_allocate.merge(tract_total_volume, on='GEOID', how='left')
            buildings_to_allocate['tract_total_volume_m3'] = buildings_to_allocate['tract_total_volume_m3'].fillna(1.0).clip(lower=1.0)
            buildings_to_allocate['volume_share'] = buildings_to_allocate['building_volume_m3'] / buildings_to_allocate['tract_total_volume_m3']
            status_messages.append("Calculated volume-based allocation weights.")

            buildings_to_allocate['initial_allocated_pop'] = buildings_to_allocate[pop_col] * buildings_to_allocate['volume_share'] * scale_factor # Apply scale factor here
            
            st.write("Scaling allocated population to match Census totals...")
            tract_allocated_sum = buildings_to_allocate.groupby('GEOID')['initial_allocated_pop'].sum().reset_index()
            tract_allocated_sum.rename(columns={'initial_allocated_pop': 'tract_allocated_pop_sum'}, inplace=True)
            scaling_df = tract_allocated_sum.merge(census_subset, on='GEOID', how='left')
            scaling_df[pop_col] = scaling_df[pop_col].fillna(0) 
            scaling_df['scaling_factor'] = np.where(
                scaling_df['tract_allocated_pop_sum'] > 1e-6, 
                scaling_df[pop_col] / scaling_df['tract_allocated_pop_sum'], 
                0 
            )
            status_messages.append("Calculated scaling factors for each tract.")
            
            buildings_to_allocate = buildings_to_allocate.merge(scaling_df[['GEOID', 'scaling_factor']], on='GEOID', how='left')
            buildings_to_allocate['scaling_factor'] = buildings_to_allocate['scaling_factor'].fillna(0) 
            buildings_to_allocate['allocated_population'] = (buildings_to_allocate['initial_allocated_pop'] * buildings_to_allocate['scaling_factor']).round().astype(int)
            buildings_to_allocate['allocated_population'] = buildings_to_allocate['allocated_population'].clip(lower=0)
            status_messages.append("Applied scaling factors to finalize allocated population.")
            
            total_allocated_pop = buildings_to_allocate['allocated_population'].sum()
            st.write(f"Total allocated population across buildings: {total_allocated_pop}")
            status_messages.append(f"Total allocated population: {total_allocated_pop}")

            # --- Calculate Estimated Households ---
            st.write("Calculating estimated households...")
            if avg_hh_override is not None:
                st.write(f"Using overridden average household size: {avg_hh_override}")
                status_messages.append(f"Using overridden Avg HH Size: {avg_hh_override}")
                avg_hh_size = float(avg_hh_override)
                if avg_hh_size <= 0:
                    st.warning("Household size override is <= 0, using 1.0 instead.")
                    avg_hh_size = 1.0
                buildings_to_allocate['avg_household_size'] = avg_hh_size # Assign directly
            else:
                # hh_size_col (e.g., 'B25010_001E') should already be present from the initial merge
                st.write(f"Using Census tract average household size (Variable: {hh_size_col})...")
                if hh_size_col not in buildings_to_allocate.columns:
                     # Error if the column is unexpectedly missing
                     raise KeyError(f"Internal Error: Expected household size column '{hh_size_col}' not found in buildings_to_allocate DataFrame before renaming.") 
                
                # Simply rename the existing column 
                buildings_to_allocate.rename(columns={hh_size_col: 'avg_household_size'}, inplace=True)
                status_messages.append(f"Using tract-level Avg HH Size (renamed from {hh_size_col})." )           

            # Check if the target column exists now (either from override or rename)
            if 'avg_household_size' not in buildings_to_allocate.columns:
                 raise KeyError("Internal Error: 'avg_household_size' column not present after processing override/rename.")
                 
            # Ensure the column is numeric before calculation
            buildings_to_allocate['avg_household_size'] = pd.to_numeric(buildings_to_allocate['avg_household_size'], errors='coerce').fillna(1.0).clip(lower=0.1)

            buildings_to_allocate['estimated_households'] = np.where(
                buildings_to_allocate['allocated_population'] > 0,
                (buildings_to_allocate['allocated_population'] / buildings_to_allocate['avg_household_size']).round().clip(lower=1),
                0 
            ).astype(int)
            status_messages.append("Calculated estimated households per building.")
            total_est_hh = buildings_to_allocate['estimated_households'].sum()
            st.write(f"Total estimated households across buildings: {total_est_hh}")
            status_messages.append(f"Total estimated households: {total_est_hh}")
            
            # Assign results back for saving/return
            buildings_with_population_gdf = buildings_to_allocate.copy() 

            status_messages.append("Population allocation calculations complete.")
            
        except Exception as e_alloc:
            st.error(f"Error during population allocation calculation: {e_alloc}")
            status_messages.append(f"ERROR: Population allocation failed: {e_alloc}")
            module_error = True
            
    # --- 3. Prepare Final Output GDF (already done in allocation block) --- 
    if not module_error:
        status_messages.append("Prepared final buildings GDF with population.")

    # --- 4. Calculate and Save Statistics --- 
    if not module_error:
        st.markdown("--- *Calculating Population Statistics* ---")
        try:
            total_res_buildings = len(buildings_with_population_gdf)
            total_pop_alloc = buildings_with_population_gdf['allocated_population'].sum() if 'allocated_population' in buildings_with_population_gdf.columns else 0
            total_hh_est = buildings_with_population_gdf['estimated_households'].sum() if 'estimated_households' in buildings_with_population_gdf.columns else 0
            
            stats_d = {
                "Total Residential Buildings (Union)": total_res_buildings,
                "Total Allocated Population": total_pop_alloc,
                "Total Estimated Households": total_hh_est,
            }
            
            if total_res_buildings > 0:
                pop_stats = buildings_with_population_gdf['allocated_population'].describe()
                hh_stats = buildings_with_population_gdf['estimated_households'].describe()
                stats_d["Avg Pop per Building"] = f"{pop_stats.get('mean', 0):.2f}"
                stats_d["Median Pop per Building"] = f"{pop_stats.get('50%', 0):.1f}"
                stats_d["Max Pop per Building"] = f"{pop_stats.get('max', 0):.0f}"
                stats_d["Avg HH per Building"] = f"{hh_stats.get('mean', 0):.2f}"
                stats_d["Median HH per Building"] = f"{hh_stats.get('50%', 0):.1f}"
                stats_d["Max HH per Building"] = f"{hh_stats.get('max', 0):.0f}"
                # Compare allocated vs census sum again
                original_total_pop_check = census_data_df[pop_col].sum()
                stats_d["Original Census Pop (Intersecting Tracts)"] = original_total_pop_check
                if original_total_pop_check > 0: 
                    diff_pct = abs(total_pop_alloc-original_total_pop_check)/original_total_pop_check*100
                    stats_d["Allocation Difference (%)"] = f"{diff_pct:.2f}%"

            stats_d_df = pd.DataFrame.from_dict(stats_d, orient='index', columns=['Value'])
            stats_d_path = os.path.join(stats_subdir, 'population_stats.csv')
            stats_d_df.to_csv(stats_d_path)
            status_messages.append(f"Saved: {os.path.basename(stats_d_path)}")
            st.write("Population Allocation Statistics:")
            st.dataframe(stats_d_df)
            
        except Exception as e_stats:
            st.warning(f"Could not calculate/save population stats: {e_stats}")
            status_messages.append(f"WARN: Population stats calculation failed: {e_stats}")
            stats_d_df = pd.DataFrame() # Ensure empty on error

    # --- 5. Generate Visualizations --- 
    if not module_error:
        st.markdown("--- *Generating Population Visualizations* ---")
        
        # --- Population Map --- 
        # Define path before try block
        map_path = os.path.join(viz_subdir, 'population_map.png') 
        try:
            st.write("Generating Population Map...") 
            fig_map, ax_map = plt.subplots(figsize=(10, 10))

            # Plotting logic using buildings_with_population_gdf...
            if not buildings_with_population_gdf.empty:
                 plot_gdf = buildings_with_population_gdf.to_crs(WEB_MERCATOR)
                 plot_gdf.plot(column='allocated_population', cmap='plasma', legend=True,
                               ax=ax_map, 
                               legend_kwds={'label': "Allocated Population",
                                            'orientation': "horizontal",
                                            'shrink': 0.5})
            # ... (Add union boundary, basemap, style_map) ...

            plt.savefig(map_path, dpi=150, bbox_inches='tight')
            plt.close(fig_map)
            status_messages.append(f"Saved: {os.path.basename(map_path)}")
        except Exception as e_map:
            st.warning(f"Could not generate/save population map: {e_map}")
            status_messages.append(f"WARN: Population map generation failed: {e_map}")
            # Check and remove exists check is safe now
            if os.path.exists(map_path): os.remove(map_path)

        # --- Population Distribution Plot ---
        # Define path before try block
        plot_path = os.path.join(viz_subdir, 'population_distribution.png') 
        try:
            st.write("Generating Population Distribution Plot...") 
            fig_plot, ax_plot = plt.subplots(figsize=(8, 4))

            # Plotting logic...
            if not buildings_with_population_gdf.empty and 'allocated_population' in buildings_with_population_gdf.columns:
                 buildings_with_population_gdf['allocated_population'].plot(kind='hist', bins=50, ax=ax_plot, title='Distribution of Allocated Population per Building')
                 ax_plot.set_xlabel("Allocated Population")
                 plt.tight_layout()
            else:
                 # Handle empty df or missing column case for plot
                 ax_plot.text(0.5, 0.5, 'No population data to plot', horizontalalignment='center', verticalalignment='center')

            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig_plot)
            status_messages.append(f"Saved: {os.path.basename(plot_path)}")
        except Exception as e_plot:
            st.warning(f"Could not generate/save population distribution plot: {e_plot}")
            status_messages.append(f"WARN: Population plot generation failed: {e_plot}")
            # Check and remove exists check is safe now
            if os.path.exists(plot_path): os.remove(plot_path)
            
    # --- 6. Save Final GDF --- 
    st.markdown("--- *Saving Buildings with Population* ---")
    try:
        pop_file = os.path.join(data_subdir, "buildings_with_population.geojson")
        if not module_error and not buildings_with_population_gdf.empty:
             # Prepare for saving (handle complex types)
             save_columns = buildings_with_population_gdf.columns.tolist()
             cols_to_drop = ['repr_point', 'tract_index', 'tract_total_volume_m3', 'volume_share', 'initial_allocated_pop', 'scaling_factor', 'height_m_alloc', 'building_area_m2_alloc', 'building_volume_m3'] # Drop intermediate cols
             # Keep avg_household_size if needed downstream, otherwise drop
             # cols_to_drop.append('avg_household_size') 
             save_columns = [col for col in save_columns if col not in cols_to_drop]
             buildings_to_save = buildings_with_population_gdf[save_columns].copy()
 
             for col in buildings_to_save.columns:
                  if buildings_to_save[col].dtype == 'object':
                     is_list_or_dict = buildings_to_save[col].apply(lambda x: isinstance(x, (list, dict))).any()
                     if is_list_or_dict:
                         st.write(f"Converting object column '{col}' to string for saving.")
                         buildings_to_save[col] = buildings_to_save[col].astype(str)
             
             buildings_to_save = buildings_to_save.to_crs(WGS84) # Ensure saving in WGS84
             buildings_to_save.to_file(pop_file, driver="GeoJSON")
             status_messages.append(f"Saved: {os.path.basename(pop_file)}")
        else:
             save_empty_gdf(pop_file)
             status_messages.append(f"Saved empty: {os.path.basename(pop_file)} (due to error or no residential buildings)")
    except Exception as e_save:
        st.warning(f"Could not save buildings_with_population.geojson: {e_save}")
        status_messages.append(f"WARN: Failed to save {os.path.basename(pop_file)}: {e_save}")

    # --- Return Results --- 
    # ... (Return dictionary)

    # --- Completion --- 
    module_D_time = time.time() - module_D_start
    status_messages.append(f"Module D completed in {module_D_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module D finished in {module_D_time:.2f}s.")
    else:
        st.error(f"Module D finished with errors in {module_D_time:.2f}s.")

    return {
        "buildings_with_population_gdf": buildings_with_population_gdf,
        "stats_d_df": stats_d_df,
        "status_messages": status_messages,
        "error": module_error
    } 