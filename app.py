import streamlit as st
import os
import yaml
import pandas as pd
import geopandas as gpd
import time
import urllib.request # Added for User-Agent fix
from PIL import Image
import folium # Added for interactive map
from streamlit_folium import st_folium # Added for interactive map
from folium.plugins import Draw # Added for drawing tool

# --- Set page config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="UAS Demand Modeler",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Set User-Agent for libraries like cenpy/pandas/geopandas that use urllib ---
# This helps avoid 403 Forbidden errors when downloading data (e.g., TIGER/Line files)
try:
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    print("Installed custom urllib opener with User-Agent.")
except Exception as e:
    print(f"Could not set custom urllib opener: {e}")
# --- End User-Agent --- 

# Import module functions (use try-except for robustness)
try:
    from modules import area_selection
    from modules import classification
    from modules import height
    from modules import population
    from modules import demand
    from modules import od_matching
    from modules import reporting
    # We will add other modules here later
    # from modules import reporting
except ImportError as e:
    st.error(f"Failed to import modules: {e}. Make sure module files exist in the 'modules' directory.")
    # Define dummy functions if import fails
    class DummyModule:
        def run_module_a(self, config): return {"status_messages": ["Error: Module A not found"], "error": True}
        def run_module_b(self, *args): return {"status_messages": ["Error: Module B not found"], "error": True}
        def run_module_c(self, *args): return {"status_messages": ["Error: Module C not found"], "error": True}
        def run_module_d(self, *args): return {"status_messages": ["Error: Module D not found"], "error": True}
        def run_module_e(self, *args): return {"status_messages": ["Error: Module E not found"], "error": True}
        def run_module_f(self, *args): return {"status_messages": ["Error: Module F not found"], "error": True}
        def run_module_g(self, *args): return {"status_messages": ["Error: Module G not found"], "error": True}
        # Add dummies for other modules if needed
    area_selection = DummyModule()
    classification = DummyModule()
    height = DummyModule()
    population = DummyModule()
    demand = DummyModule()
    od_matching = DummyModule()
    reporting = DummyModule()

# --- Configuration Loading ---
@st.cache_resource # Cache the loaded config resource
def load_config(config_path="code_v20_3_config.yaml"):
    """Loads the configuration file and performs initial processing."""
    if not os.path.exists(config_path):
        st.error(f"Default config file '{config_path}' not found! Cannot proceed.")
        return None # Return None to indicate failure
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # --- Perform initial processing like in Colab --- 
        # Handle Null conversion for avg_household_size_override
        if 'population_allocation' in config and config['population_allocation'].get('avg_household_size_override') == 'Null':
            config['population_allocation']['avg_household_size_override'] = None

        # Process and normalize hourly distribution
        if 'demand_model' in config and 'hourly_distribution' in config['demand_model']:
            try:
                hourly_dist = config['demand_model']['hourly_distribution']
                print(f"[Debug] Raw hourly dist from YAML: {hourly_dist}") # DEBUG PRINT
                # Ensure keys are integers and values are floats
                hourly_dist = {int(k): float(v) if isinstance(v, (int, float)) else 0.0 for k, v in hourly_dist.items()}
                # Ensure all hours 0-23 are present
                for h in range(24): hourly_dist.setdefault(h, 0.0)
                # Normalize if sum is not close to 1.0
                print(f"[Debug] Hourly dist before sum: {hourly_dist}") # DEBUG PRINT
                hourly_sum = sum(hourly_dist.values())
                # Increase tolerance for floating point comparison
                if not abs(hourly_sum - 1.0) < 1e-9: # Increased tolerance from 1e-5
                    st.warning(f"Hourly distribution sum in config ({hourly_sum:.4f}) is not 1.0. Normalizing.")
                    if hourly_sum > 1e-9:
                        hourly_dist = {k: v / hourly_sum for k, v in hourly_dist.items()}
                    else:
                        st.warning("Hourly distribution sum is zero. Using uniform distribution.")
                        hourly_dist = {h: 1/24 for h in range(24)}
                config['demand_model']['hourly_distribution'] = dict(sorted(hourly_dist.items()))
            except Exception as e:
                st.warning(f"Error processing hourly distribution in config: {e}. Using uniform fallback.")
                config['demand_model']['hourly_distribution'] = {h: 1/24 for h in range(24)}

        # Ensure output_dir exists, default if not specified
        if 'output_dir' not in config:
            config['output_dir'] = './uas_analysis_output'
        os.makedirs(config['output_dir'], exist_ok=True)

        print(f"[Debug] Config loaded. Keys: {list(config.keys())}") # DEBUG PRINT
        return config

    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file '{config_path}': {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading configuration: {e}")
        return None

# Load configuration or stop if failed
config_initial = load_config()

# --- Initialize Session State --- 
# Use session state to store results between runs if needed, and config edits
if 'config' not in st.session_state:
    st.session_state.config = config_initial.copy() if config_initial else {}
    print(f"[Debug] Session state config initialized. Keys: {list(st.session_state.config.keys())}") # DEBUG PRINT
else:
    print(f"[Debug] Session state config already exists. Keys: {list(st.session_state.config.keys())}") # DEBUG PRINT

# Initialize session state for module results
for mod_key in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    if f'module_{mod_key}_results' not in st.session_state:
        st.session_state[f'module_{mod_key}_results'] = {}

# Initialize specific state variables
if 'drawn_bounds' not in st.session_state:
    st.session_state.drawn_bounds = None # To store bounds from map
if 'area_input_method' not in st.session_state:
    # Default to coordinate input matching original config values
    st.session_state.area_input_method = "Enter Coordinates" 

# --- Page Title ---
st.title("✈️ UAS Last-Mile Delivery Demand Modeler")
st.markdown("A tool for simulating and visualizing UAS delivery demand based on OSM and Census data.")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Configuration")

run_simulation = False # Default
if not st.session_state.config: # Check if config loaded
    st.sidebar.error("Configuration failed to load. Please check the `code_v20_3_config.yaml` file.")
else:
    st.sidebar.subheader("Area Selection")
    
    # --- Input Method Selection ---
    st.sidebar.radio(
        "Area Input Method", 
        ["Enter Coordinates", "Draw on Map"], 
        key='area_input_method',
        help="Choose how to define the study area."
    )

    # --- Conditional Coordinate Inputs (Sidebar) ---
    if st.session_state.area_input_method == "Enter Coordinates":
        st.sidebar.warning("Ensure Min < Max for latitude and longitude.", icon="⚠️")
        coords = st.session_state.config['area_selection']['coordinates']
        # Use columns for better layout
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lat_min = st.number_input("Min Latitude", value=coords[0], min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="lat_min")
            lon_min = st.number_input("Min Longitude", value=coords[1], min_value=-180.0, max_value=180.0, step=0.001, format="%.5f", key="lon_min")
        with col2:
            lat_max = st.number_input("Max Latitude", value=coords[2], min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="lat_max")
            lon_max = st.number_input("Max Longitude", value=coords[3], min_value=-180.0, max_value=180.0, step=0.001, format="%.5f", key="lon_max")
        # Ensure drawn_bounds is cleared if user switches back to coordinates
        st.session_state.drawn_bounds = None
        
    # Buffer input always visible under Area Selection
    buffer_km = st.session_state.config['area_selection']['buffer_km']
    buffer_km_input = st.sidebar.number_input("Buffer (km)", value=buffer_km, min_value=0.1, max_value=5.0, step=0.1, key="buffer_km")

    # --- Other Sidebar Inputs (Demand Model, etc.) ---
    st.sidebar.subheader("Demand Model (Example)")
    base_rate = st.session_state.config['demand_model']['base_deliveries_per_household_per_day']
    base_rate_input = st.sidebar.slider("Base Deliveries/HH/Day", min_value=0.01, max_value=1.0, value=base_rate, step=0.01, key="base_rate")

    # TODO: Add widgets for other configurable parameters (Census vars, height defaults, demand coeffs, etc.)

    st.sidebar.divider()
    # Button to trigger the simulation
    run_simulation = st.sidebar.button("Run Simulation Pipeline", type="primary")

# --- Main Area --- 

# Create placeholders for outputs in the main area
status_area = st.expander("Status Log", expanded=False)
results_area = st.container()

# --- Conditional Map Display (Main Area) ---
if st.session_state.area_input_method == "Draw on Map":
    results_area.subheader("1. Define Study Area")
    results_area.warning("Draw a rectangle on the map. Please avoid excessively large areas, as this can significantly increase processing time and data downloads.", icon="⚠️")

    # Use initial coordinates from config for map center if available
    map_center = [38.98, -95.67] # Default center (US)
    zoom_start = 4
    if st.session_state.config:
        initial_coords = st.session_state.config.get('area_selection', {}).get('coordinates')
        if initial_coords and len(initial_coords) == 4:
            map_center = [(initial_coords[0] + initial_coords[2]) / 2, (initial_coords[1] + initial_coords[3]) / 2]
            zoom_start = 12

    m = folium.Map(location=map_center, zoom_start=zoom_start)
    draw_plugin = Draw(
        export=False,
        draw_options={
            'polyline': False, 'polygon': False, 'circle': False, 
            'marker': False, 'circlemarker': False,
            'rectangle': {'shapeOptions': {'color': '#FF0000'}}
        }
    )
    m.add_child(draw_plugin)
    map_output = st_folium(m, width=700, height=500, key="map_area_selector") # Added key for stability

    # Process map output to get bounds
    drawn_rectangle_coords = None
    if map_output and map_output.get("last_active_drawing"):
        geometry = map_output["last_active_drawing"]["geometry"]
        if geometry and geometry["type"] == "Polygon":
            coords_lon_lat = geometry["coordinates"][0]
            lons = [c[0] for c in coords_lon_lat]
            lats = [c[1] for c in coords_lon_lat]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            min_lat = max(-90.0, min_lat)
            max_lat = min(90.0, max_lat)
            min_lon = max(-180.0, min_lon)
            max_lon = min(180.0, max_lon)
            drawn_rectangle_coords = [min_lat, min_lon, max_lat, max_lon]
            st.session_state.drawn_bounds = drawn_rectangle_coords
            
    # Display the selected bounds (optional feedback)
    if st.session_state.drawn_bounds:
        results_area.info(f"Selected Bounds: Lat [{st.session_state.drawn_bounds[0]:.4f}, {st.session_state.drawn_bounds[2]:.4f}], Lon [{st.session_state.drawn_bounds[1]:.4f}, {st.session_state.drawn_bounds[3]:.4f}]")
    else:
         results_area.info("No area selected yet. Draw a rectangle on the map.")

# Display placeholder/results area title
results_area.subheader("Simulation Results")

if run_simulation:
    results_area.info("Simulation pipeline starting...")
    status_area.empty() 
    status_messages_all = []
    # Clear previous module results
    for mod_key in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
        st.session_state[f'module_{mod_key}_results'] = {}

    # --- Get Coordinates based on Input Method --- 
    selected_coordinates = None
    if st.session_state.area_input_method == "Draw on Map":
        if st.session_state.drawn_bounds:
            selected_coordinates = st.session_state.drawn_bounds
        else:
            results_area.error("Error: Input method is 'Draw on Map', but no rectangle was drawn. Please draw an area.")
            st.stop()
    elif st.session_state.area_input_method == "Enter Coordinates":
        # Read from sidebar state
        lat_min_val = st.session_state.lat_min
        lon_min_val = st.session_state.lon_min
        lat_max_val = st.session_state.lat_max
        lon_max_val = st.session_state.lon_max
        # Validate coordinates
        if lat_min_val >= lat_max_val or lon_min_val >= lon_max_val:
            results_area.error("Error: Invalid coordinates entered. Ensure Min Latitude < Max Latitude and Min Longitude < Max Longitude.")
            st.stop()
        selected_coordinates = [lat_min_val, lon_min_val, lat_max_val, lon_max_val]
    else:
        results_area.error("Error: Invalid area input method selected.")
        st.stop()

    # --- Update config with chosen coordinates and sidebar values before running ---
    current_config = st.session_state.config.copy()
    current_config['area_selection']['coordinates'] = selected_coordinates
    current_config['area_selection']['buffer_km'] = st.session_state.buffer_km 
    current_config['demand_model']['base_deliveries_per_household_per_day'] = st.session_state.base_rate 
    # Add updates for other sidebar widgets here if added later...

    # --- Dictionary to hold stats from all modules ---
    all_stats = {}

    # --- Call Module A --- 
    st.session_state.module_a_results = area_selection.run_module_a(current_config)
    status_messages_all.extend(st.session_state.module_a_results.get("status_messages", []))
    all_stats['A'] = st.session_state.module_a_results.get("stats_a_df")
    
    # ---> Add estimated UTM CRS from Module A results to the current config <--- 
    utm_crs = st.session_state.module_a_results.get("utm_crs")
    if utm_crs:
        current_config['utm_crs'] = utm_crs 
        status_messages_all.append(f"Added estimated UTM CRS ({utm_crs}) to config for downstream use.")
    else:
        status_messages_all.append("WARN: No UTM CRS returned from Module A.")

    # Display status messages from Module A
    with status_area:
        st.subheader("Module A Log")
        for msg in st.session_state.module_a_results.get("status_messages", []):
            if "WARN:" in msg or "Warning:" in msg:
                st.warning(msg)
            elif "ERROR:" in msg or "Error:" in msg:
                st.error(msg)
            else:
                st.info(msg)

    if st.session_state.module_a_results.get("error"):
        results_area.error("Module A failed. Cannot proceed.")
        st.stop()
    else:
        results_area.success("Module A completed successfully.")

        # --- Display Module A Results --- 
        results_area.subheader("Module A Outputs")

        # Display Stats
        stats_a_df = st.session_state.module_a_results.get("stats_a_df")
        if stats_a_df is not None and not stats_a_df.empty:
            results_area.dataframe(stats_a_df)
        else:
            results_area.write("No statistics generated by Module A.")

        # Display Maps
        viz_dir = os.path.join(current_config['output_dir'], "visualizations")
        map_files = {
            "Fig 1: Selected Area": os.path.join(viz_dir, 'setup_map_1.png'),
            "Fig 2: Tracts & Union": os.path.join(viz_dir, 'setup_map_2.png'),
            "Fig 3: Buffer & Buildings": os.path.join(viz_dir, 'setup_map_3_initial.png')
        }
        map_cols = results_area.columns(len(map_files))
        for idx, (caption, file_path) in enumerate(map_files.items()):
            if os.path.exists(file_path):
                try:
                    image = Image.open(file_path)
                    map_cols[idx].image(image, caption=caption, use_container_width=True)
                except Exception as e:
                    map_cols[idx].warning(f"Could not load map: {os.path.basename(file_path)}\n{e}")
            else:
                map_cols[idx].warning(f"Map not found: {os.path.basename(file_path)}")

        # --- Call Module B (Classification & Store ID) --- 
        results_area.markdown("--- *Starting Module B* ---")
        osm_gdf = st.session_state.module_a_results.get("osm_gdf")
        union_gdf = st.session_state.module_a_results.get("union_gdf")
        buffer_gdf = st.session_state.module_a_results.get("buffered_gdf")

        # Check if necessary inputs from Module A exist
        if osm_gdf is not None and union_gdf is not None and buffer_gdf is not None:
            st.session_state.module_b_results = classification.run_module_b(current_config, osm_gdf, buffer_gdf, union_gdf)
            status_messages_all.extend(st.session_state.module_b_results.get("status_messages", []))
            all_stats['B'] = st.session_state.module_b_results.get("stats_b_df")

            # Display status messages from Module B
            with status_area:
                st.subheader("Module B Log")
                for msg in st.session_state.module_b_results.get("status_messages", []):
                    if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                    elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                    else: st.info(msg)

            if st.session_state.module_b_results.get("error"):
                results_area.error("Module B failed. Subsequent modules may be affected.")
            else:
                results_area.success("Module B completed successfully.")

                # --- Display Module B Results --- 
                results_area.subheader("Module B Outputs")

                # Display Stats
                stats_b_df = st.session_state.module_b_results.get("stats_b_df")
                if stats_b_df is not None and not stats_b_df.empty:
                    results_area.dataframe(stats_b_df)
                else:
                    results_area.write("No statistics generated by Module B.")
                
                # Display Map (Fig 4)
                fig4_path = os.path.join(viz_dir, 'classification_map_4.png')
                if os.path.exists(fig4_path):
                    try:
                        image4 = Image.open(fig4_path)
                        # Use a new column or display below previous maps
                        results_area.image(image4, caption="Fig 4: Classification & Stores", use_container_width=True)
                    except Exception as e:
                        results_area.warning(f"Could not load map: {os.path.basename(fig4_path)}\n{e}")
                else:
                    results_area.warning(f"Map not found: {os.path.basename(fig4_path)}")

                # --- >>> ADD MODULE C CALL HERE <<< --- 
                if not st.session_state.module_b_results.get("error"):
                    # --- Call Module C (Height Estimation) --- 
                    results_area.markdown("--- *Starting Module C* ---")
                    classified_buildings_gdf = st.session_state.module_b_results.get("classified_buildings_gdf")
                    # union_gdf is already available from above
                    utm_crs = st.session_state.module_a_results.get("utm_crs") # From Module A

                    # Check if necessary inputs exist
                    if classified_buildings_gdf is not None and not classified_buildings_gdf.empty and union_gdf is not None:
                        st.session_state.module_c_results = height.run_module_c(current_config, classified_buildings_gdf, union_gdf, utm_crs)
                        status_messages_all.extend(st.session_state.module_c_results.get("status_messages", []))
                        all_stats['C'] = st.session_state.module_c_results.get("stats_c_df")

                        # Display status messages from Module C
                        with status_area:
                            st.subheader("Module C Log")
                            for msg in st.session_state.module_c_results.get("status_messages", []):
                                if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                                elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                                else: st.info(msg)

                        if st.session_state.module_c_results.get("error"):
                            results_area.error("Module C failed. Subsequent modules may be affected.")
                        else:
                            results_area.success("Module C completed successfully.")

                            # --- Display Module C Results --- 
                            results_area.subheader("Module C Outputs")
                            
                            # Display Stats
                            stats_c_df = st.session_state.module_c_results.get("stats_c_df")
                            if stats_c_df is not None and not stats_c_df.empty:
                                results_area.dataframe(stats_c_df)
                            else:
                                results_area.write("No statistics generated by Module C.")
                            
                            # Display Maps/Plots
                            map_c_files = {
                                "Height Map": os.path.join(viz_dir, 'height_map.png'),
                                "Height Distribution": os.path.join(viz_dir, 'height_distribution.png')
                            }
                            map_c_cols = results_area.columns(len(map_c_files))
                            for idx, (caption, file_path) in enumerate(map_c_files.items()):
                                if os.path.exists(file_path):
                                    try:
                                        image_c = Image.open(file_path)
                                        map_c_cols[idx].image(image_c, caption=caption, use_container_width=True)
                                    except Exception as e:
                                        map_c_cols[idx].warning(f"Could not load image: {os.path.basename(file_path)}\n{e}")
                                else:
                                     map_c_cols[idx].warning(f"Image not found: {os.path.basename(file_path)}")
                    else:
                         results_area.warning("Skipping Module C: Missing required GeoDataFrames (classified_buildings_gdf, union_gdf) from previous modules.")
                         status_messages_all.append("WARN: Skipping Module C due to missing inputs.")

                # --- Update Placeholder for subsequent modules --- 
                results_area.markdown("--- *End of Module C* ---")
                results_area.info("Subsequent modules (D, E, F, G) are not yet implemented.")
                # TODO: Call Module D ...
                
        else:
             results_area.warning("Skipping Module B & C: Missing required GeoDataFrames (osm_gdf, union_gdf, buffer_gdf) from Module A.")
             status_messages_all.append("WARN: Skipping Module B & C due to missing inputs from Module A.")
             # Update placeholder since B was skipped
             results_area.markdown("--- *End of Module A (Skipped B, C)* ---")
             results_area.info("Subsequent modules (D, E, F, G) are not yet implemented.")

    # --- Final completion message (adjust if needed) --- 
    if not st.session_state.module_a_results.get("error"):
        results_area.success("Pipeline run complete (up to implemented modules). Check logs and outputs.")

    if not st.session_state.module_b_results.get("error"):
        if not st.session_state.module_c_results.get("error"):
             # --- Call Module D (Population Allocation) --- 
            results_area.markdown("--- *Starting Module D* ---")
            buildings_with_heights_gdf = st.session_state.module_c_results.get("buildings_with_heights_gdf")
            census_data_df = st.session_state.module_a_results.get("census_data_df") # Get from Module A
            tracts_gdf = st.session_state.module_a_results.get("tracts_gdf") # Get from Module A

            # Check if necessary inputs exist
            if buildings_with_heights_gdf is not None and census_data_df is not None and tracts_gdf is not None:
                # Call run_module_d with 4 arguments, matching its definition
                st.session_state.module_d_results = population.run_module_d(current_config, buildings_with_heights_gdf, census_data_df, tracts_gdf)
                status_messages_all.extend(st.session_state.module_d_results.get("status_messages", []))
                all_stats['D'] = st.session_state.module_d_results.get("stats_d_df")

                # Display status messages from Module D
                with status_area:
                    st.subheader("Module D Log")
                    for msg in st.session_state.module_d_results.get("status_messages", []):
                        if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                        elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                        else: st.info(msg)
            
                if st.session_state.module_d_results.get("error"):
                    results_area.error("Module D failed. Subsequent modules may be affected.")
                else:
                    results_area.success("Module D completed successfully.")

                    # --- Display Module D Results --- 
                    results_area.subheader("Module D Outputs")
                    
                    # Display Stats
                    stats_d_df = st.session_state.module_d_results.get("stats_d_df")
                    if stats_d_df is not None and not stats_d_df.empty:
                        results_area.dataframe(stats_d_df)
                    else:
                        results_area.write("No statistics generated by Module D.")
                    
                    # Display Maps/Plots
                    map_d_files = {
                        "Population Map": os.path.join(viz_dir, 'population_map.png'),
                        "Population Distribution": os.path.join(viz_dir, 'population_distribution.png')
                    }
                    map_d_cols = results_area.columns(len(map_d_files))
                    for idx, (caption, file_path) in enumerate(map_d_files.items()):
                        if os.path.exists(file_path):
                            try:
                                image_d = Image.open(file_path)
                                map_d_cols[idx].image(image_d, caption=caption, use_container_width=True)
                            except Exception as e:
                                map_d_cols[idx].warning(f"Could not load image: {os.path.basename(file_path)}\n{e}")
                        else:
                             map_d_cols[idx].warning(f"Image not found: {os.path.basename(file_path)}")
            else:
                 results_area.warning("Skipping Module D due to missing inputs from previous modules.")
                 status_messages_all.append("WARN: Skipping Module D - Missing required inputs.")
                 # Ensure results are empty if skipped
                 st.session_state.module_d_results = {"error": True, "status_messages": ["Skipped due to missing inputs."]}

            if not st.session_state.module_d_results.get("error"):
                 # --- Call Module E (Demand Modeling) --- 
                results_area.markdown("--- *Starting Module E* ---")
                buildings_with_population_gdf = st.session_state.module_d_results.get("buildings_with_population_gdf")
                census_data_df = st.session_state.module_a_results.get("census_data_df") # Get from Module A

                # Check if necessary inputs exist
                if buildings_with_population_gdf is not None and census_data_df is not None:
                    st.session_state.module_e_results = demand.run_module_e(current_config, buildings_with_population_gdf, census_data_df)
                    status_messages_all.extend(st.session_state.module_e_results.get("status_messages", []))
                    all_stats['E'] = st.session_state.module_e_results.get("stats_e_df")
                    
                    # Display status messages from Module E
                    with status_area:
                        st.subheader("Module E Log")
                        for msg in st.session_state.module_e_results.get("status_messages", []):
                            if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                            elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                            else: st.info(msg)
                    
                    if st.session_state.module_e_results.get("error"):
                        results_area.error("Module E failed. Subsequent modules may be affected.")
                    else:
                        results_area.success("Module E completed successfully.")

                        # --- Display Module E Results --- 
                        results_area.subheader("Module E Outputs")
                        
                        # Display Stats
                        stats_e_df = st.session_state.module_e_results.get("stats_e_df")
                        if stats_e_df is not None and not stats_e_df.empty:
                            results_area.dataframe(stats_e_df)
                        else:
                            results_area.write("No statistics generated by Module E.")
                        
                        # Display Map
                        dmap_path = os.path.join(viz_dir, 'demand_rate_map.png')
                        if os.path.exists(dmap_path):
                            try:
                                image_e = Image.open(dmap_path)
                                results_area.image(image_e, caption="Demand Rate Map", use_container_width=True)
                            except Exception as e:
                                results_area.warning(f"Could not load image: {os.path.basename(dmap_path)}\n{e}")
                        else:
                             results_area.warning(f"Image not found: {os.path.basename(dmap_path)}")
                            
                        # Note: Animation frames are generated but not displayed here.
                        # They would be used by Module G to create a GIF.
                        anim_frame_dir = os.path.join(current_config['output_dir'], "animation_frames", "demand")
                        if os.path.exists(anim_frame_dir) and len(os.listdir(anim_frame_dir)) > 0:
                             results_area.caption(f"Demand animation frames generated in: {anim_frame_dir}")
                        else:
                             results_area.caption("Demand animation frames were not generated.")

                else:
                     results_area.warning("Skipping Module E due to missing inputs from previous modules.")
                     status_messages_all.append("WARN: Skipping Module E - Errors/missing inputs from Module D.")
                     st.session_state.module_e_results = {"error": True, "status_messages": ["Skipped due to Module D issues."]}

            if not st.session_state.module_e_results.get("error"):
                 # --- Call Module F (O-D Matching) --- 
                results_area.markdown("--- *Starting Module F* ---")
                delivery_events_df = st.session_state.module_e_results.get("delivery_events_df")
                stores_gdf = st.session_state.module_b_results.get("stores_gdf") # From Module B
                utm_crs = st.session_state.module_a_results.get("utm_crs") # From Module A
                
                # Check if necessary inputs exist
                if delivery_events_df is not None and stores_gdf is not None and utm_crs:
                    st.session_state.module_f_results = od_matching.run_module_f(current_config, delivery_events_df, stores_gdf, utm_crs)
                    status_messages_all.extend(st.session_state.module_f_results.get("status_messages", []))
                    all_stats['F'] = st.session_state.module_f_results.get("stats_f_df")

                    # Display status messages from Module F
                    with status_area:
                        st.subheader("Module F Log")
                        for msg in st.session_state.module_f_results.get("status_messages", []):
                            if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                            elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                            else: st.info(msg)
                    
                    if st.session_state.module_f_results.get("error"):
                        results_area.error("Module F failed.") # Final data generation failed
                    else:
                        results_area.success("Module F completed successfully.")

                        # --- Display Module F Results --- 
                        results_area.subheader("Module F Outputs")
                        
                        # Display Stats
                        stats_f_df = st.session_state.module_f_results.get("stats_f_df")
                        if stats_f_df is not None and not stats_f_df.empty:
                            results_area.dataframe(stats_f_df)
                        else:
                            results_area.write("No statistics generated by Module F.")
                        
                        # Display Static O-D Map
                        odmap_path = os.path.join(viz_dir, 'od_map_static.png')
                        if os.path.exists(odmap_path):
                            try:
                                image_f = Image.open(odmap_path)
                                results_area.image(image_f, caption="Static O-D Map (Target Hour)", use_container_width=True)
                            except Exception as e:
                                results_area.warning(f"Could not load image: {os.path.basename(odmap_path)}\n{e}")
                        else:
                             results_area.warning(f"Image not found: {os.path.basename(odmap_path)}")
                            
                        # Note: O-D animation frames are generated but not displayed here.
                        od_anim_frame_dir = os.path.join(current_config['output_dir'], "animation_frames", "od")
                        if os.path.exists(od_anim_frame_dir) and len(os.listdir(od_anim_frame_dir)) > 0:
                             results_area.caption(f"O-D animation frames generated in: {od_anim_frame_dir}")
                        else:
                             results_area.caption("O-D animation frames were not generated.")
                                     
                        # Provide download link for the final CSV
                        routing_csv_path = os.path.join(current_config['output_dir'], "data", 'routing_dataset.csv')
                        if os.path.exists(routing_csv_path):
                             try:
                                 with open(routing_csv_path, "rb") as fp:
                                     results_area.download_button(
                                         label="Download Routing Dataset (CSV)",
                                         data=fp,
                                         file_name="routing_dataset.csv",
                                         mime="text/csv"
                                     )
                             except Exception as e:
                                  results_area.warning(f"Could not offer routing CSV for download: {e}")
                        else:
                             results_area.warning("Final routing_dataset.csv not found.")
                else:
                     results_area.warning("Skipping Module F: Missing required inputs (events, stores, or utm_crs) from previous modules.")
                     status_messages_all.append("WARN: Skipping Module F due to missing inputs.")

            if not st.session_state.module_f_results.get("error"):
                 # --- Call Module G (Reporting / GIF Gen) --- 
                results_area.markdown("--- *Starting Module G* ---")
                
                # Pass the collected stats dictionary
                st.session_state.module_g_results = reporting.run_module_g(current_config, all_stats)
                status_messages_all.extend(st.session_state.module_g_results.get("status_messages", []))
                
                # Display status messages from Module G
                with status_area:
                    st.subheader("Module G Log")
                    for msg in st.session_state.module_g_results.get("status_messages", []):
                        if "WARN:" in msg or "Warning:" in msg: st.warning(msg)
                        elif "ERROR:" in msg or "Error:" in msg: st.error(msg)
                        else: st.info(msg)
                                        
                if st.session_state.module_g_results.get("error"):
                     results_area.warning("Module G encountered errors (GIF/Report generation might have failed).") # Warning, not necessarily fatal
                else:
                     results_area.success("Module G completed successfully.")
                    
                # --- Display Module G Results (GIFs & Report Download) --- 
                results_area.subheader("Module G Outputs - Final Visualizations")
                
                demand_gif_path = st.session_state.module_g_results.get("demand_gif_path")
                od_gif_path = st.session_state.module_g_results.get("od_gif_path")
                report_path = st.session_state.module_g_results.get("summary_report_path")
                
                gif_cols = results_area.columns(2)
                
                if demand_gif_path and os.path.exists(demand_gif_path):
                     gif_cols[0].image(demand_gif_path, caption="Demand Animation")
                else:
                     gif_cols[0].warning("Demand animation GIF not found.")
                                         
                if od_gif_path and os.path.exists(od_gif_path):
                     gif_cols[1].image(od_gif_path, caption="O-D Animation")
                else:
                     gif_cols[1].warning("O-D animation GIF not found.")
                                         
                if report_path and os.path.exists(report_path):
                     try:
                         with open(report_path, "rb") as fp:
                             results_area.download_button(
                                 label="Download Summary Report (MD)",
                                 data=fp,
                                 file_name="summary_report.md",
                                 mime="text/markdown"
                             )
                     except Exception as e:
                          results_area.warning(f"Could not offer summary report for download: {e}")
                else:
                     results_area.warning("Summary report file not found.")

        # --- Final Placeholder --- 
        results_area.markdown("--- *End of Pipeline* ---")
                
    # --- Final completion message --- 
    if not any(st.session_state[f'module_{m}_results'].get("error") for m in ['a', 'b', 'c', 'd', 'e', 'f']): 
         results_area.success("✅ Simulation Pipeline Completed Successfully!")
    else:
         results_area.error("Pipeline finished, but one or more modules encountered errors. Please check the status log.")

else:
    # Initial message when the app loads and simulation hasn't run
    if st.session_state.area_input_method == "Enter Coordinates":
         results_area.info("Configure simulation parameters in the sidebar and click 'Run Simulation Pipeline'.")
    # If map is shown, the map instructions serve as the initial message

# Display final combined status log if needed (optional)
# status_area.subheader("Full Run Log")
# for msg in status_messages_all:
#     status_area.text(msg) 