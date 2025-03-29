import os
import sys
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import contextily as cx
from shapely.geometry import box
import subprocess
import tempfile
import shutil
import glob
from PIL import Image
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="UAS Last-Mile Delivery Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "uas_analysis_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    for subdir in ['data', 'visualizations', 'stats', 'config', 'animation_frames']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# Helper function to load and display an image
def load_and_display_image(image_path, caption=None):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

# Function to convert degrees to decimal coordinates
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# Function to display the various module outputs
def display_module_outputs(module_name, tab):
    with tab:
        st.header(f"Module {module_name}")
        viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
        stats_dir = os.path.join(OUTPUT_DIR, "stats")
        
        # Display visualizations
        if module_name == "A":
            image_path = os.path.join(viz_dir, "setup_map_2.png")
            load_and_display_image(image_path, "Study Area, Tracts, and Buildings")
            
            # Display stats
            stats_path = os.path.join(stats_dir, "setup_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
        
        elif module_name == "B":
            image_path = os.path.join(viz_dir, "classification_map.png")
            load_and_display_image(image_path, "Building Classification")
            
            stats_path = os.path.join(stats_dir, "classification_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
        
        elif module_name == "C":
            image_path = os.path.join(viz_dir, "height_visualization.png")
            load_and_display_image(image_path, "Building Heights")
            
            stats_path = os.path.join(stats_dir, "height_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
        
        elif module_name == "D":
            image_path = os.path.join(viz_dir, "population_visualization.png")
            load_and_display_image(image_path, "Population Allocation")
            
            stats_path = os.path.join(stats_dir, "population_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
        
        elif module_name == "E":
            image_path = os.path.join(viz_dir, "demand_visualization.png")
            load_and_display_image(image_path, "Demand Modeling")
            
            stats_path = os.path.join(stats_dir, "demand_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
            
            # Display the GIF animation if available
            gif_path = os.path.join(viz_dir, "delivery_animation.gif")
            if os.path.exists(gif_path):
                st.image(gif_path, caption="Delivery Animation")
        
        elif module_name == "F":
            image_path = os.path.join(viz_dir, "od_map.png")
            load_and_display_image(image_path, "Origin-Destination Matching")
            
            stats_path = os.path.join(stats_dir, "od_stats.csv")
            if os.path.exists(stats_path):
                df = pd.read_csv(stats_path)
                st.dataframe(df)
            
            # Show the routing dataset
            data_path = os.path.join(OUTPUT_DIR, "data", "routing_dataset.csv")
            if os.path.exists(data_path):
                st.subheader("Routing Dataset")
                df = pd.read_csv(data_path)
                st.dataframe(df.head(10))
                st.download_button(
                    label="Download Full Dataset",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="routing_dataset.csv",
                    mime="text/csv"
                )

# Function to run the simulation and update the UI
def run_simulation(config):
    st.session_state.running = True
    st.session_state.progress = 0
    
    # Save the config to a YAML file
    config_path = os.path.join(OUTPUT_DIR, "config", "input_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create a modified version of code_v18.py that doesn't use the pip install line
    with open('code_v18.py', 'r') as f:
        code = f.read()
    
    # Replace the pip install line and set up a progress function
    code = code.replace('!pip install geopandas osmnx cenpy contextily matplotlib seaborn scikit-learn requests scipy fuzzywuzzy python-levenshtein pyyaml Pillow pyogrio --quiet', '# pip dependencies already installed')
    
    # Insert progress reporting
    progress_code = """
# Progress reporting for Streamlit
def update_progress(message, progress_value):
    with open('progress.txt', 'w') as f:
        f.write(f"{message}\\n{progress_value}")
"""
    code = code.replace('# -----------------------------------------------------------------------------\n# 0. Setup Environment\n# -----------------------------------------------------------------------------', 
                       '# -----------------------------------------------------------------------------\n# 0. Setup Environment\n# -----------------------------------------------------------------------------\n' + progress_code)
    
    # Add progress updates at key points
    code = code.replace('print("## 0. Setting up Environment...")', 'print("## 0. Setting up Environment..."); update_progress("Setting up environment...", 5)')
    code = code.replace('print("\\n## Module A: Area Selection & Data Acquisition...")', 'print("\\n## Module A: Area Selection & Data Acquisition..."); update_progress("Module A: Area Selection & Data Acquisition...", 10)')
    code = code.replace('print("\\n## Module B: Building Classification & Store Identification...")', 'print("\\n## Module B: Building Classification & Store Identification..."); update_progress("Module B: Building Classification & Store Identification...", 30)')
    code = code.replace('print("\\n## Module C: Building Height Estimation...")', 'print("\\n## Module C: Building Height Estimation..."); update_progress("Module C: Building Height Estimation...", 45)')
    code = code.replace('print("\\n## Module D: Population Allocation...")', 'print("\\n## Module D: Population Allocation..."); update_progress("Module D: Population Allocation...", 60)')
    code = code.replace('print("\\n## Module E: Demand Modeling...")', 'print("\\n## Module E: Demand Modeling..."); update_progress("Module E: Demand Modeling...", 75)')
    code = code.replace('print("\\n## Module F: Origin-Destination Matching & Dataset Generation...")', 'print("\\n## Module F: Origin-Destination Matching & Dataset Generation..."); update_progress("Module F: Origin-Destination Matching & Dataset Generation...", 85)')
    code = code.replace('print("\\n## Module G: Reporting...")', 'print("\\n## Module G: Reporting..."); update_progress("Module G: Reporting...", 95)')
    code = code.replace('print("\\nProcessing Complete. Check the output directory structure for results.")', 'print("\\nProcessing Complete. Check the output directory structure for results."); update_progress("Complete!", 100)')
    
    # Replace the config YAML with our config path
    code = code.replace('config_yaml = """', f'with open("{config_path}", "r") as f:\n    config_yaml = f.read()\n')
    code = code.replace('"""', '', 1)  # Remove the first triple quote
    # Find the line that ends the YAML and remove it
    yaml_end_idx = code.find('"""', code.find('config_yaml'))
    if yaml_end_idx != -1:
        code = code[:yaml_end_idx] + code[yaml_end_idx+3:]
    
    # Write the modified code to a temporary file
    temp_file = 'temp_code_v18.py'
    with open(temp_file, 'w') as f:
        f.write(code)
    
    # Create a progress file
    with open('progress.txt', 'w') as f:
        f.write("Initializing...\n0")
    
    # Run the code in a separate process
    process = subprocess.Popen([sys.executable, temp_file], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update the progress bar while the process is running
    while process.poll() is None:
        try:
            with open('progress.txt', 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    message = lines[0].strip()
                    progress = int(lines[1].strip())
                    progress_bar.progress(progress)
                    status_text.text(message)
                    st.session_state.progress = progress
        except:
            pass
        time.sleep(0.5)
    
    # Read stdout and stderr
    stdout, stderr = process.communicate()
    
    # Clean up temporary files
    if os.path.exists(temp_file):
        os.remove(temp_file)
    if os.path.exists('progress.txt'):
        os.remove('progress.txt')
    
    # Display process output
    if process.returncode != 0:
        st.error("Error during simulation:")
        st.code(stderr)
    else:
        progress_bar.progress(100)
        status_text.text("Simulation completed successfully!")
        st.success("Simulation completed successfully!")
        
        # Create tabs for the different module outputs
        module_tabs = st.tabs(["Module A", "Module B", "Module C", "Module D", "Module E", "Module F"])
        display_module_outputs("A", module_tabs[0])
        display_module_outputs("B", module_tabs[1])
        display_module_outputs("C", module_tabs[2])
        display_module_outputs("D", module_tabs[3])
        display_module_outputs("E", module_tabs[4])
        display_module_outputs("F", module_tabs[5])
        
        # Display the summary report
        st.subheader("Summary Report")
        report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()
                st.text_area("Report", report_text, height=400)
    
    st.session_state.running = False

# Main UI
st.title("UAS Last-Mile Delivery Simulator")
st.write("This tool simulates last-mile delivery demand for Unmanned Aerial Systems (UAS).")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Area Selection
    st.subheader("Area Selection")
    area_method = "coordinates"  # Currently only supporting coordinates
    
    # Coordinate inputs
    col1, col2 = st.columns(2)
    with col1:
        sw_lat = st.number_input("SW Latitude", value=30.265, format="%.6f")
        sw_lon = st.number_input("SW Longitude", value=-97.745, format="%.6f")
    with col2:
        ne_lat = st.number_input("NE Latitude", value=30.270, format="%.6f")
        ne_lon = st.number_input("NE Longitude", value=-97.740, format="%.6f")
    
    buffer_km = st.slider("Buffer (km)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    # Data Acquisition
    st.subheader("Data Acquisition")
    census_product = st.selectbox("Census Product", ["ACSDT5Y2022", "ACSDT5Y2021", "ACSDT5Y2020"], index=0)
    state_code = st.text_input("State Code", "48")  # Default is Texas
    county_code = st.text_input("County Code", "453")  # Default is Travis County
    
    # Building Classification
    st.subheader("Building Classification")
    min_residential_area = st.number_input("Min Residential Area (m²)", value=30, step=5)
    max_residential_area = st.number_input("Max Residential Area (m²)", value=1000, step=50)
    
    # Height Estimation
    st.subheader("Height Estimation")
    default_height = st.number_input("Default Building Height (m)", value=3.5, step=0.5)
    meters_per_level = st.number_input("Meters per Level", value=3.5, step=0.5)
    
    # Demand Model
    st.subheader("Demand Model")
    base_deliveries = st.number_input("Base Deliveries per Household per Day", value=0.18, format="%.2f", step=0.01)
    income_coef = st.slider("Income Coefficient", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    pop_density_coef = st.slider("Population Density Coefficient", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # Simulation Parameters
    st.subheader("Simulation Parameters")
    sim_start_date = st.date_input("Simulation Start Date", datetime.now())
    sim_duration = st.slider("Simulation Duration (days)", min_value=1, max_value=14, value=7, step=1)
    random_seed = st.number_input("Random Seed", value=42, step=1)
    
    # OD Matching
    st.subheader("Origin-Destination Matching")
    match_method = st.selectbox("Matching Method", ["Market Share Weighted", "Proximity-Based", "Random"], index=0)
    
    # Run button
    run_button = st.button("Run Simulation", disabled=st.session_state.running)

# Main content area
if run_button or st.session_state.running:
    if not st.session_state.running:
        # Build the configuration from inputs
        config = {
            'area_selection': {
                'method': 'coordinates',
                'coordinates': [sw_lat, sw_lon, ne_lat, ne_lon],
                'buffer_km': buffer_km
            },
            'data_acquisition': {
                'osm_tags': {'building': True, 'shop': True, 'amenity': True, 'landuse': True},
                'census_variables': [
                    'B19013_001E',  # Median household income
                    'B01003_001E',  # Total population
                    'B25010_001E',  # Average household size
                    'B23025_004E',  # Employed population (16+)
                    'B23025_002E',  # Population in labor force (16+)
                    'B25001_001E',  # Total housing units
                ],
                'census_product': census_product,
                'state_code': state_code,
                'county_code': county_code,
                'tract_year': int(census_product[-4:])
            },
            'building_classification': {
                'method': 'rule_based',
                'rule_based_parameters': {
                    'min_residential_area_m2': min_residential_area,
                    'max_residential_area_m2': max_residential_area,
                    'likely_max_residential_area_m2': 500,
                    'min_nonresidential_area_m2': 1500,
                    'residential_building_tags': ['residential', 'house', 'apartments', 'detached', 'terrace', 'semidetached_house', 'bungalow', 'dormitory'],
                    'nonresidential_building_tags': ['commercial', 'retail', 'industrial', 'warehouse', 'office', 'supermarket', 'shop', 'mall', 'store', 'school', 'hospital', 'church', 'public', 'civic', 'government', 'hotel', 'motel']
                },
                'store_tags': {
                    'shop': ['supermarket', 'department_store', 'convenience', 'mall', 'wholesale', 'grocery'],
                    'building': ['warehouse', 'retail', 'commercial'],
                    'amenity': ['marketplace', 'fast_food'],
                    'name_keywords': ["walmart", "target", "amazon", "kroger", "H-E-B", "heb", "costco", "distribution center", "fulfillment", "supercenter", "warehouse", "grocery", "market", "cvs", "walgreens"]
                }
            },
            'height_estimation': {
                'default_height_m': default_height,
                'meters_per_level': meters_per_level,
                'knn_neighbors': 5,
                'use_area_feature': True,
                'max_height_cap_m': 150
            },
            'population_allocation': {
                'population_scale_factor': 1.0,
                'avg_household_size_override': None
            },
            'demand_model': {
                'base_deliveries_per_household_per_day': base_deliveries,
                'income_coef': income_coef,
                'pop_density_coef': pop_density_coef,
                'household_size_coef': -0.15,
                'employment_coef': 0.1,
                'reference_values': {
                    'ref_median_income': 75000,
                    'ref_pop_density': 3000,
                    'ref_avg_household_size': 2.5,
                    'ref_employment_rate': 95
                },
                'daily_variation': {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.2, 5: 0.8, 6: 0.5},
                'hourly_distribution': {
                    0:0.0, 1:0.0, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.01, 7:0.02, 8:0.04, 9:0.07, 
                    10:0.09, 11:0.10, 12:0.10, 13:0.11, 14:0.12, 15:0.12, 16:0.11, 17:0.09, 
                    18:0.07, 19:0.05, 20:0.02, 21:0.01, 22:0.0, 23:0.0
                },
                'monthly_factors': {
                    1: 0.8, 2: 0.85, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.0, 7: 1.0, 
                    8: 1.05, 9: 1.0, 10: 1.1, 11: 1.5, 12: 1.75
                },
                'simulation_start_date': sim_start_date.strftime('%Y-%m-%d'),
                'simulation_duration_days': sim_duration
            },
            'origin_destination_matching': {
                'method': match_method,
                'market_shares': {
                    'Walmart': 0.65, 
                    'Target': 0.20, 
                    'Retail Hub': 0.10, 
                    'Kroger': 0.0, 
                    'H-E-B': 0.05, 
                    'Warehouse/Fulfillment': 0.0, 
                    'Other': 0.0
                },
                'simulation_hour_for_matching': 17
            },
            'random_seed': random_seed,
            'output_dir': OUTPUT_DIR
        }
        
        # Run the simulation
        run_simulation(config)
    
    # If already running, show the progress
    elif st.session_state.running:
        progress_bar = st.progress(st.session_state.progress)
        st.info("Simulation in progress...")
        
# Show previous results button
if not st.session_state.running and os.path.exists(os.path.join(OUTPUT_DIR, "summary_report.txt")):
    if st.button("Show Previous Results"):
        st.session_state.show_results = True

# Show previous results
if st.session_state.show_results:
    st.success("Displaying previous simulation results")
    
    # Create tabs for the different module outputs
    module_tabs = st.tabs(["Module A", "Module B", "Module C", "Module D", "Module E", "Module F"])
    display_module_outputs("A", module_tabs[0])
    display_module_outputs("B", module_tabs[1])
    display_module_outputs("C", module_tabs[2])
    display_module_outputs("D", module_tabs[3])
    display_module_outputs("E", module_tabs[4])
    display_module_outputs("F", module_tabs[5])
    
    # Display the summary report
    st.subheader("Summary Report")
    report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_text = f.read()
            st.text_area("Report", report_text, height=400) 