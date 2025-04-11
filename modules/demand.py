# modules/demand.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import json
import streamlit as st
import matplotlib.pyplot as plt
import contextily as cx
from datetime import datetime, timedelta

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

# --- Helper Function for Demand Rate Calculation ---
# Note: This helper exists for clarity/potential row-wise use, 
# but the main calculation below uses a faster vectorized approach.
def calculate_demand_rate(building_row, census_row, config):
    """Calculates the base daily demand rate for a single residential building.
       Merges building and relevant census data before calling.
    """
    demand_config = config['demand_model']
    base_rate = demand_config['base_deliveries_per_household_per_day']
    coeffs = demand_config['coefficients']
    refs = demand_config['reference_values']
    est_hh = building_row.get('estimated_households', 0)

    if est_hh <= 0:
        return 0.0

    # Calculate adjustment factors
    adj_factors = {}
    for factor_name, var_coef in coeffs.items(): # e.g., factor_name='income', var_coef={'variable': 'median_income', 'coefficient': 0.1}
        var = var_coef['variable']
        coef = var_coef['coefficient']
        ref_val = refs.get(var, 1) # Default ref value to 1 if not found
        val = census_row.get(var, ref_val) # Use ref_val if census value missing
        
        # Avoid division by zero for ref_val
        if ref_val == 0:
            if val == 0:
                adj = 1.0 # If both are zero, no adjustment
            else:
                adj = 1.0 + coef # Or some large number? Let's use 1+coef for now.
        else:
            adj = 1.0 + coef * ((val / ref_val) - 1.0)
        
        # Apply floor to adjustment factor (e.g., min 0.2)
        adj = max(demand_config.get('adjustment_floor', 0.2), adj)
        adj_factors[f"adj_{factor_name}"] = adj

    # Calculate final demand rate
    final_rate = base_rate * est_hh
    for adj in adj_factors.values():
        final_rate *= adj
        
    return max(0.0, final_rate) # Ensure non-negative rate

# --- Main Module Function ---
def run_module_e(config, buildings_with_population_gdf, census_data_df):
    """Executes Module E: Demand Modeling & Simulation.

    Args:
        config (dict): The configuration dictionary.
        buildings_with_population_gdf (gpd.GeoDataFrame): Buildings with population from Module D.
        census_data_df (pd.DataFrame): Filtered Census demographic data from Module A.

    Returns:
        dict: A dictionary containing results:
              'buildings_with_demand_gdf', 'delivery_events_df',
              'stats_e_df', 'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module E: Demand Modeling & Simulation")
    st.markdown("--- *Initializing* ---")
    module_E_start = time.time()
    demand_config = config['demand_model']
    output_dir = config['output_dir']
    data_subdir = os.path.join(output_dir, "data")
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")
    anim_subdir = os.path.join(output_dir, "animation_frames", "demand") # Subdir for frames
    os.makedirs(anim_subdir, exist_ok=True) # Create anim frame dir

    # Initialize return values
    buildings_with_demand_gdf = gpd.GeoDataFrame()
    delivery_events_df = pd.DataFrame()
    stats_e_df = pd.DataFrame()
    status_messages = []
    module_error = False

    # --- Check Inputs --- 
    if buildings_with_population_gdf is None or buildings_with_population_gdf.empty:
        st.warning("Module E: Input buildings GeoDataFrame is empty. Skipping demand modeling.")
        status_messages.append("WARN: Skipping Module E - No buildings with population provided.")
        module_error = True
    elif 'estimated_households' not in buildings_with_population_gdf.columns:
         st.error("Module E: Input buildings GDF missing 'estimated_households' column. Skipping.")
         status_messages.append("ERROR: Skipping Module E - Missing 'estimated_households' column.")
         module_error = True
    elif 'GEOID' not in buildings_with_population_gdf.columns:
         st.error("Module E: Input buildings GDF missing 'GEOID' column. Skipping.")
         status_messages.append("ERROR: Skipping Module E - Missing 'GEOID' column.")
         module_error = True
         
    if census_data_df is None or census_data_df.empty:
        st.warning("Module E: Input Census data DataFrame is empty. Needed for demand factors. Skipping.")
        status_messages.append("WARN: Skipping Module E - No Census data provided.")
        module_error = True
    elif 'GEOID' not in census_data_df.columns:
         st.error("Module E: Input Census data missing 'GEOID' column. Skipping.")
         status_messages.append("ERROR: Skipping Module E - Census data missing 'GEOID' column.")
         module_error = True
         
    # Check if all variables needed for coefficients exist in census_data_df
    required_vars = {vc['variable'] for vc in demand_config.get('coefficients', {}).values()}    
    missing_vars = required_vars - set(census_data_df.columns)
    if missing_vars:
        st.error(f"Module E: Census data missing required variables for demand coefficients: {missing_vars}. Skipping.")
        status_messages.append(f"ERROR: Skipping Module E - Census data missing vars: {missing_vars}")
        module_error = True

    # Save empty files and return if critical error found
    if module_error:
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_demand.geojson"))
        save_empty_df(os.path.join(data_subdir, 'delivery_events.csv'))
        save_empty_df(os.path.join(stats_subdir, 'demand_stats.csv'))
        return {
            "buildings_with_demand_gdf": buildings_with_demand_gdf, "delivery_events_df": delivery_events_df,
            "stats_e_df": stats_e_df, "status_messages": status_messages, "error": True
        }

    # --- 1. Prepare Data & Calculate Base Demand Rate --- 
    st.markdown("--- *Calculating Base Demand Rates* ---")
    buildings_for_demand = gpd.GeoDataFrame() # Initialize in case of error
    try:
        # Ensure we have the necessary columns from previous steps
        cols_needed = ['unique_id', 'GEOID', 'estimated_households', 'geometry']
        # Check if all needed columns exist in the input GDF
        missing_cols = [col for col in cols_needed if col not in buildings_with_population_gdf.columns]
        if missing_cols:
            raise ValueError(f"Input buildings_with_population_gdf is missing required columns: {missing_cols}")
            
        buildings_for_demand = buildings_with_population_gdf[cols_needed].copy()
        # KEEP 'unique_id' for now, rename later just before use
        
        status_messages.append(f"Preparing {len(buildings_for_demand)} residential buildings for demand modeling.")
        
        # Merge required census variables for demand calculation
        census_vars_needed = list(required_vars) + ['GEOID']
        census_subset = census_data_df[census_vars_needed].copy()
        
        # Convert census vars to numeric, fill NaN with reference values for calculation safety
        refs = demand_config.get('reference_values', {})
        for var in required_vars:
            ref_val = refs.get(var, 0) # Use 0 if no ref value? Maybe 1 is safer? Check if ref=0 causes issues.
            census_subset[var] = pd.to_numeric(census_subset[var], errors='coerce').fillna(ref_val)
            
        buildings_for_demand = buildings_for_demand.merge(census_subset, on='GEOID', how='left')
        status_messages.append("Merged required Census variables onto buildings.")
        
        # Handle buildings where merge failed (shouldn't happen if previous steps ok)
        missing_census_merge = buildings_for_demand[list(required_vars)].isna().any(axis=1)
        if missing_census_merge.any():
             st.warning(f"{missing_census_merge.sum()} buildings missing Census data needed for demand factors. Demand rate will be 0.")
             status_messages.append(f"WARN: {missing_census_merge.sum()} buildings missing Census data for demand factors.")
             # Fill missing census values with reference values *again* after merge
             for var in required_vars:
                 ref_val = refs.get(var, 0)
                 buildings_for_demand[var].fillna(ref_val, inplace=True)

        # --- Vectorized Demand Rate Calculation --- 
        st.write("Calculating base daily demand rate per building (vectorized)...")
        coeffs = demand_config.get('coefficients', {})
        refs = demand_config.get('reference_values', {})
        base_rate_per_hh = demand_config['base_deliveries_per_household_per_day']
        adj_floor = demand_config.get('adjustment_floor', 0.2)
        
        buildings_for_demand['base_demand'] = buildings_for_demand['estimated_households'] * base_rate_per_hh
        final_demand = buildings_for_demand['base_demand'].copy()

        # Calculate adjustment factors vectorized
        calculated_adjustments = {}
        for factor_name, var_coef in coeffs.items():
            var = var_coef['variable']
            coef = var_coef['coefficient']
            ref_val = refs.get(var, 1.0) # Default ref val to 1
            col_name = f"adj_{factor_name}"
            
            building_vals = buildings_for_demand[var].astype(float) # Ensure float for calc
            
            if ref_val == 0:
                adj = np.where(building_vals == 0, 1.0, 1.0 + coef)
            else:
                adj = 1.0 + coef * ((building_vals / ref_val) - 1.0)
            
            adj = np.maximum(adj_floor, adj) # Apply floor
            calculated_adjustments[col_name] = adj # Store for inspection/saving
            final_demand *= adj # Apply adjustment multiplicatively
            status_messages.append(f"Applied adjustment factor: {col_name}")

        buildings_for_demand['demand_rate'] = np.maximum(0.0, final_demand) # Ensure non-negative
        # Add adjustment factors as columns for potential saving/inspection
        for adj_col, adj_values in calculated_adjustments.items():
            buildings_for_demand[adj_col] = adj_values
            
        status_messages.append("Calculated base demand rate for all buildings.")
        
    except Exception as e_rate:
        st.error(f"Error calculating demand rates: {e_rate}")
        status_messages.append(f"ERROR: Demand rate calculation failed: {e_rate}")
        module_error = True

    # --- 2. Simulate Delivery Events --- 
    if not module_error:
        st.markdown("--- *Simulating Delivery Events* ---")
        try:
            sim_days = demand_config.get('simulation_days', 7)
            start_date_str = demand_config.get('simulation_start_date', '2023-10-27')
            seed = config.get('random_seed', 42)
            np.random.seed(seed)
            status_messages.append(f"Simulating deliveries for {sim_days} days starting from {start_date_str}.")
            st.write(f"Simulating deliveries for {sim_days} days...")

            # Prepare temporal factors (normalize if needed)
            hourly_dist = demand_config.get('hourly_distribution', {h: 1/24 for h in range(24)})
            daily_factors = demand_config.get('daily_factors', {d: 1.0 for d in range(7)})
            monthly_factors = demand_config.get('monthly_factors', {m: 1.0 for m in range(1, 13)})
            # Ensure factors are complete and numeric
            hourly_dist = {int(h): float(v) for h, v in hourly_dist.items()} 
            daily_factors = {int(d): float(v) for d, v in daily_factors.items()} 
            monthly_factors = {int(m): float(v) for m, v in monthly_factors.items()} 
            # Normalize hourly distribution (should already be done in config loading, but double check)
            hourly_sum = sum(hourly_dist.values())
            if not abs(hourly_sum - 1.0) < 1e-5 and hourly_sum > 1e-9:
                 hourly_dist = {h: v / hourly_sum for h, v in hourly_dist.items()}
            hours = list(hourly_dist.keys())
            hour_probs = list(hourly_dist.values())
            status_messages.append("Prepared temporal factors.")

            all_events = []
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

            # Progress bar
            sim_progress = st.progress(0.0)

            for i in range(sim_days):
                current_date = start_date + timedelta(days=i)
                day_of_week = current_date.weekday() # Monday = 0, Sunday = 6
                month_of_year = current_date.month
                
                # Get temporal adjustment for the current day
                day_factor = daily_factors.get(day_of_week, 1.0)
                month_factor = monthly_factors.get(month_of_year, 1.0)
                daily_adj_rate = buildings_for_demand['demand_rate'] * day_factor * month_factor
                
                # Simulate number of deliveries for each building for this day (Poisson)
                num_deliveries_today = np.random.poisson(daily_adj_rate.clip(lower=0))
                buildings_with_deliveries = buildings_for_demand[num_deliveries_today > 0].copy()
                buildings_with_deliveries['num_events_today'] = num_deliveries_today[num_deliveries_today > 0]
                
                # Expand buildings dataframe for each delivery event
                events_today_df = buildings_with_deliveries.loc[buildings_with_deliveries.index.repeat(buildings_with_deliveries['num_events_today'])].reset_index(drop=True)
                
                if not events_today_df.empty:
                    # Assign random hour based on distribution
                    assigned_hour = np.random.choice(hours, size=len(events_today_df), p=hour_probs)
                    # Assign random minute/second
                    random_minute = np.random.randint(0, 60, size=len(events_today_df))
                    random_second = np.random.randint(0, 60, size=len(events_today_df))
                    
                    # Create precise timestamp using pd.to_datetime with DataFrame components
                    timestamp_df = pd.DataFrame({
                        'year': current_date.year,
                        'month': current_date.month,
                        'day': current_date.day,
                        'hour': assigned_hour,
                        'minute': random_minute,
                        'second': random_second
                    })
                    events_today_df['timestamp'] = pd.to_datetime(timestamp_df)                 
                                       
                    # Add necessary columns (lat/lon from geometry CENTROID)
                    # Calculate centroid first, as geometry is likely Polygon
                    centroids = events_today_df.geometry.centroid
                    events_today_df['latitude'] = centroids.y
                    events_today_df['longitude'] = centroids.x
                    # Add unique order ID - using index from expanded df before appending
                    base_order_id = len(all_events) # Get current total before appending
                    events_today_df['order_id'] = base_order_id + events_today_df.index 
                    
                    # Select the columns needed and rename 'unique_id' on the fly for appending
                    cols_to_append = ['order_id', 'unique_id', 'timestamp', 'latitude', 'longitude']
                    df_to_append = events_today_df[cols_to_append].rename(columns={'unique_id': 'building_unique_id'})
                    
                    # Append the newly created DataFrame with correct names
                    all_events.append(df_to_append)
                
                # Update progress bar
                sim_progress.progress((i + 1) / sim_days)

            sim_progress.progress(1.0)

            if all_events:
                # Concatenate all daily events
                delivery_events_df = pd.concat(all_events, ignore_index=True)
                # Ensure the concatenated DataFrame has the correct column name if rename didn't carry over somehow (defensive)
                if 'building_unique_id' not in delivery_events_df.columns and 'unique_id' in delivery_events_df.columns:
                    delivery_events_df.rename(columns={'unique_id': 'building_unique_id'}, inplace=True)
                
                delivery_events_df.sort_values(by='timestamp', inplace=True)
                delivery_events_df['order_id'] = range(len(delivery_events_df)) # Re-assign contiguous order ID
                status_messages.append(f"Generated {len(delivery_events_df)} total delivery events over {sim_days} days.")
                st.write(f"Generated {len(delivery_events_df)} total delivery events.")
            else:
                st.warning("Simulation resulted in zero delivery events.")
                status_messages.append("WARN: Simulation generated zero delivery events.")
                # Define empty DF with the FINAL desired columns
                delivery_events_df = pd.DataFrame(columns=['order_id', 'building_unique_id', 'timestamp', 'latitude', 'longitude'])
            
            # Store total simulated deliveries per building
            # Use the FINAL delivery_events_df which should have 'building_unique_id'
            if not delivery_events_df.empty and 'building_unique_id' in delivery_events_df.columns:
                total_sim_deliveries = delivery_events_df.groupby('building_unique_id').size()
                # Map back to buildings_for_demand (which still has unique_id)
                buildings_for_demand['simulated_total_deliveries'] = buildings_for_demand['unique_id'].map(total_sim_deliveries).fillna(0).astype(int)
            else:
                buildings_for_demand['simulated_total_deliveries'] = 0

        except Exception as e_sim:
            st.error(f"Error during delivery event simulation: {e_sim}")
            status_messages.append(f"ERROR: Delivery simulation failed: {e_sim}")
            module_error = True
            delivery_events_df = pd.DataFrame() # Ensure empty on error

    # --- 3. Prepare Final Outputs (GDFs, Stats, Viz) --- 
    buildings_with_demand_gdf = gpd.GeoDataFrame() # Initialize
    if not module_error:
        try:
            # Merge demand rate back onto original buildings GDF
            buildings_with_demand_gdf = buildings_for_demand.copy()
            
            # Calculate Stats
            total_res_buildings = len(buildings_with_demand_gdf)
            total_demand_rate = buildings_with_demand_gdf['demand_rate'].sum() if 'demand_rate' in buildings_with_demand_gdf.columns else 0
            total_sim_deliveries = buildings_with_demand_gdf['simulated_total_deliveries'].sum() if 'simulated_total_deliveries' in buildings_with_demand_gdf.columns else 0
            
            stats_e = {
                "Total Residential Buildings (Demand Calc)": total_res_buildings,
                "Total Base Daily Demand Rate (Sum)": f"{total_demand_rate:.2f}",
                "Total Simulated Deliveries (Across Period)": total_sim_deliveries,
            }
            
            if total_res_buildings > 0:
                rate_stats = buildings_with_demand_gdf['demand_rate'].describe()
                sim_stats = buildings_with_demand_gdf['simulated_total_deliveries'].describe()
                stats_e["Avg Demand Rate / Bldg"] = f"{rate_stats.get('mean', 0):.4f}"
                stats_e["Median Demand Rate / Bldg"] = f"{rate_stats.get('50%', 0):.4f}"
                stats_e["Avg Sim Deliveries / Bldg"] = f"{sim_stats.get('mean', 0):.2f}"
                stats_e["Median Sim Deliveries / Bldg"] = f"{sim_stats.get('50%', 0):.1f}"
                stats_e["Max Sim Deliveries / Bldg"] = f"{sim_stats.get('max', 0):.0f}"
                
            if not delivery_events_df.empty:
                 stats_e["Avg Deliveries / Day"] = f"{(total_sim_deliveries / sim_days):.2f}"
                 # Hourly distribution summary
                 hourly_counts = delivery_events_df['timestamp'].dt.hour.value_counts().sort_index()
                 peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else 'N/A'
                 peak_count = hourly_counts.max() if not hourly_counts.empty else 0
                 stats_e["Peak Hour (Simulation)"] = peak_hour
                 stats_e["Deliveries in Peak Hour"] = peak_count
                 # Daily distribution summary
                 daily_counts = delivery_events_df['timestamp'].dt.dayofweek.value_counts().sort_index()
                 # ... add more detailed stats if needed

            stats_e_df = pd.DataFrame.from_dict(stats_e, orient='index', columns=['Value'])
            stats_e_path = os.path.join(stats_subdir, 'demand_stats.csv')
            stats_e_df.to_csv(stats_e_path)
            status_messages.append(f"Saved: {os.path.basename(stats_e_path)}")
            st.write("Demand Modeling Statistics:")
            st.dataframe(stats_e_df)
            
            # Generate Visualizations (Map, Timeseries, Animation Frames)
            # --- Static Demand Rate Map --- 
            if not buildings_with_demand_gdf.empty:
                try:
                    st.write("Generating Static Demand Rate Map...")
                    fig_dmap, ax_dmap = plt.subplots(figsize=(10, 10))
                    
                    plot_gdf = buildings_with_demand_gdf.to_crs(WEB_MERCATOR) # Use constant
                    plot_gdf.plot(column='demand_rate', cmap='YlOrRd', legend=True,
                                  ax=ax_dmap, 
                                  legend_kwds={'label': "Base Daily Demand Rate",
                                               'orientation': "horizontal",
                                               'shrink': 0.5})
                    
                    # Add union boundary from config if available (might need to reload/pass)
                    # union_gdf.to_crs(WEB_MERCATOR).boundary.plot(ax=ax_dmap, color='blue', linewidth=1, linestyle=':') # Use constant

                    try:
                        cx.add_basemap(ax_dmap, crs=plot_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
                    except Exception as e_base:
                        st.warning(f"Basemap for Demand Map failed: {e_base}")
                    
                    style_map(ax_dmap, 'Base Daily Demand Rate per Residential Building')
                    dmap_path = os.path.join(viz_subdir, 'demand_rate_map.png')
                    plt.savefig(dmap_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_dmap)
                    status_messages.append(f"Saved: {os.path.basename(dmap_path)}")
                except Exception as e_fig:
                    st.warning(f"Could not generate/save Demand Rate Map: {e_fig}")
                    status_messages.append(f"WARN: Demand Rate Map generation failed: {e_fig}")
            else:
                status_messages.append("Skipped Demand Rate Map - No buildings with demand.")

            # --- Demand Animation Frames --- 
            if not delivery_events_df.empty and not buildings_with_demand_gdf.empty:
                st.write("Generating Demand Animation Frames (this might take time)...")
                anim_prog = st.progress(0.0)
                try:
                    # Get building centroids for plotting points
                    buildings_plot_pts = buildings_with_demand_gdf.copy()
                    buildings_plot_pts['geometry'] = buildings_plot_pts.geometry.centroid
                    buildings_plot_pts = buildings_plot_pts.to_crs(WEB_MERCATOR) # Use constant

                    # Prepare events data with hour
                    events_anim_df = delivery_events_df.copy()
                    events_anim_df['hour_of_sim'] = (events_anim_df['timestamp'] - events_anim_df['timestamp'].min()).dt.total_seconds() // 3600
                    max_hour = int(events_anim_df['hour_of_sim'].max())
                    hourly_events = events_anim_df.groupby('hour_of_sim')
                    status_messages.append(f"Generating {max_hour + 1} hourly demand animation frames...")

                    # Create frames hour by hour
                    for hour in range(max_hour + 1):
                        fig_anim, ax_anim = plt.subplots(figsize=(10, 10))
                        
                        # Plot all building centroids lightly
                        buildings_plot_pts.plot(ax=ax_anim, color='gray', markersize=1, alpha=0.3)
                        
                        # Plot deliveries up to this hour
                        events_so_far = events_anim_df[events_anim_df['hour_of_sim'] <= hour]
                        if not events_so_far.empty:
                             gpd.GeoDataFrame(
                                 events_so_far, 
                                 geometry=gpd.points_from_xy(events_so_far.longitude, events_so_far.latitude),
                                 crs=WGS84
                             ).to_crs(WEB_MERCATOR).plot(ax=ax_anim, color='red', markersize=5, alpha=0.6) # Use constant
                        
                        # Add basemap and style
                        try: cx.add_basemap(ax_anim, crs=buildings_plot_pts.crs.to_string(), source=cx.providers.CartoDB.Positron) 
                        except Exception: pass # Ignore basemap errors for frames
                        style_map(ax_anim, f'Simulated Deliveries (Hour: {hour:03d})')
                        ax_anim.set_xticks([])
                        ax_anim.set_yticks([])
                        
                        # Save frame
                        frame_path = os.path.join(anim_subdir, f'demand_frame_{hour:03d}.png')
                        plt.savefig(frame_path, dpi=100, bbox_inches='tight') # Lower DPI for frames
                        plt.close(fig_anim)
                        
                        anim_prog.progress((hour + 1) / (max_hour + 1))
                    
                    status_messages.append(f"Finished generating demand animation frames in {anim_subdir}.")
                    anim_prog.progress(1.0)

                except Exception as e_anim:
                    st.error(f"Error generating demand animation frames: {e_anim}")
                    status_messages.append(f"ERROR: Demand animation frame generation failed: {e_anim}")
                    # Don't set module error? Maybe static map is enough?
            else:
                 status_messages.append("Skipped Demand Animation Frames - No delivery events simulated.")

        except Exception as e_output:
            st.error(f"Error preparing Module E outputs (stats/viz): {e_output}")
            status_messages.append(f"ERROR: Preparing Module E outputs failed: {e_output}")
            # Don't set module_error=True here, let simulation results still be returned if available
            
    # --- 4. Save Data Files --- 
    if not module_error:
        st.markdown("--- *Saving Demand Modeling Results* ---")
        # Save buildings with demand rate
        try:
            demand_file = os.path.join(data_subdir, "buildings_with_demand.geojson")
            # Prepare for saving - select relevant cols, convert types
            cols_to_keep = ['unique_id', 'GEOID', 'estimated_households', 'demand_rate', 'simulated_total_deliveries', 'geometry']
            # Add back adjustment factors if needed for inspection
            adj_cols = [col for col in buildings_with_demand_gdf.columns if col.startswith('adj_')]
            buildings_to_save = buildings_with_demand_gdf[cols_to_keep + adj_cols].copy()
            
            for col in buildings_to_save.columns:
                 if buildings_to_save[col].dtype == 'object':
                    is_list_or_dict = buildings_to_save[col].apply(lambda x: isinstance(x, (list, dict))).any()
                    if is_list_or_dict:
                        buildings_to_save[col] = buildings_to_save[col].astype(str)
                        
            if not buildings_to_save.empty:
                 buildings_to_save.to_file(demand_file, driver="GeoJSON")
            else:
                 save_empty_gdf(demand_file)
            status_messages.append(f"Saved: {os.path.basename(demand_file)}")
        except Exception as e_save:
            st.warning(f"Could not save buildings_with_demand.geojson: {e_save}")
            status_messages.append(f"WARN: Failed to save {os.path.basename(demand_file)}: {e_save}")

        # Save delivery events
        try:
            events_file = os.path.join(data_subdir, 'delivery_events.csv')
            if not delivery_events_df.empty:
                delivery_events_df.to_csv(events_file, index=False)
            else:
                save_empty_df(events_file, columns=['order_id', 'building_unique_id', 'timestamp', 'latitude', 'longitude'])
            status_messages.append(f"Saved: {os.path.basename(events_file)}")
        except Exception as e_save:
            st.warning(f"Could not save delivery_events.csv: {e_save}")
            status_messages.append(f"WARN: Failed to save {os.path.basename(events_file)}: {e_save}")
    else:
        # Save empty files if error occurred
        save_empty_gdf(os.path.join(data_subdir, "buildings_with_demand.geojson"))
        save_empty_df(os.path.join(data_subdir, 'delivery_events.csv'))

    # --- Completion --- 
    module_E_time = time.time() - module_E_start
    status_messages.append(f"Module E completed in {module_E_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module E finished in {module_E_time:.2f}s.")
    else:
        st.error(f"Module E finished with errors in {module_E_time:.2f}s.")

    # --- Return Results --- 
    return {
        "buildings_with_demand_gdf": buildings_with_demand_gdf, 
        "delivery_events_df": delivery_events_df, 
        "stats_e_df": stats_e_df,
        "status_messages": status_messages,
        "error": module_error
    } 