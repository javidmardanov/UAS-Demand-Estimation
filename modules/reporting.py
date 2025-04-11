# modules/reporting.py
import os
import glob
import imageio # For GIF creation
import pandas as pd
import time
import streamlit as st
import datetime # Added import

# Import utilities (optional, may not be needed directly)
# try:
#     from utils.helpers import ...
# except ImportError:
#     pass

def create_gif(frame_folder, output_gif_path, frame_prefix, duration=0.2):
    """Creates a GIF from a folder of PNG frames."""
    filenames = sorted(glob.glob(os.path.join(frame_folder, f'{frame_prefix}*.png')))
    if not filenames:
        return False # No frames found
    
    images = []
    for filename in filenames:
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            st.warning(f"Could not read frame {os.path.basename(filename)} for GIF: {e}")
            continue # Skip problematic frame
            
    if not images:
         return False # No valid images read
         
    try:
        imageio.mimsave(output_gif_path, images, duration=duration, loop=0) # loop=0 means infinite loop
        return True
    except Exception as e:
        st.error(f"Failed to create GIF '{os.path.basename(output_gif_path)}': {e}")
        return False

# --- Main Module Function ---
def run_module_g(config, all_stats_dfs):
    """Executes Module G: Reporting & GIF Generation.

    Args:
        config (dict): The configuration dictionary.
        all_stats_dfs (dict): A dictionary containing stats DataFrames 
                              from previous modules (e.g., {'A': stats_a_df, 'B': stats_b_df, ...}).

    Returns:
        dict: A dictionary containing results:
              'demand_gif_path', 'od_gif_path', 'summary_report_path' (optional),
              'status_messages' (list), 'error' (bool)
    """
    st.subheader("Module G: Reporting & GIF Generation")
    st.markdown("--- *Initializing* ---")
    module_G_start = time.time()
    output_dir = config['output_dir']
    viz_subdir = os.path.join(output_dir, "visualizations")
    stats_subdir = os.path.join(output_dir, "stats")
    demand_anim_dir = os.path.join(output_dir, "animation_frames", "demand")
    od_anim_dir = os.path.join(output_dir, "animation_frames", "od")
    
    # Initialize return values
    demand_gif_path = None
    od_gif_path = None
    summary_report_path = None # Placeholder for report file path
    status_messages = []
    module_error = False # Less critical if only reporting fails?

    gif_duration = config.get('reporting', {}).get('gif_frame_duration', 0.2)

    # --- 1. Create Demand GIF --- 
    st.markdown("--- *Creating Demand Animation GIF* ---")
    try:
        demand_gif_file = os.path.join(viz_subdir, "demand_animation.gif")
        st.write("Generating demand GIF (may take a moment)...")
        if create_gif(demand_anim_dir, demand_gif_file, "demand_frame_", duration=gif_duration):
            demand_gif_path = demand_gif_file
            status_messages.append(f"Successfully created: {os.path.basename(demand_gif_file)}")
        else:
            status_messages.append("WARN: Demand GIF creation failed or no frames found.")
    except Exception as e:
        st.error(f"Error creating demand GIF: {e}")
        status_messages.append(f"ERROR: Demand GIF creation failed: {e}")
        # Continue to OD GIF?

    # --- 2. Create O-D GIF --- 
    st.markdown("--- *Creating O-D Animation GIF* ---")
    try:
        od_gif_file = os.path.join(viz_subdir, "od_animation.gif")
        st.write("Generating O-D GIF (may take a moment)...")
        if create_gif(od_anim_dir, od_gif_file, "od_frame_", duration=gif_duration):
            od_gif_path = od_gif_file
            status_messages.append(f"Successfully created: {os.path.basename(od_gif_file)}")
        else:
            status_messages.append("WARN: O-D GIF creation failed or no frames found.")
    except Exception as e:
        st.error(f"Error creating O-D GIF: {e}")
        status_messages.append(f"ERROR: O-D GIF creation failed: {e}")

    # --- 3. Generate Summary Report (Basic Example) --- 
    st.markdown("--- *Generating Summary Report* ---")
    try:
        report_path = os.path.join(output_dir, "summary_report.md") # Markdown report
        st.write("Generating summary report...")
        with open(report_path, 'w') as f:
            f.write("# Simulation Summary Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Include key config settings (example)
            f.write("## Key Configuration\n")
            f.write(f"- Area Coordinates: {config['area_selection']['coordinates']}\n")
            f.write(f"- Buffer: {config['area_selection']['buffer_km']} km\n")
            f.write(f"- Simulation Days: {config['demand_model'].get('simulation_days', 'N/A')}\n")
            f.write(f"- O-D Matching Method: {config['origin_destination_matching'].get('method', 'N/A')}\n")
            f.write(f"- Target Hour for CSV: {config['origin_destination_matching'].get('simulation_hour_for_matching', 'N/A')}\n\n")

            # Include summary stats from each module
            f.write("## Summary Statistics\n")
            for module_code, stats_df in all_stats_dfs.items():
                 if stats_df is not None and not stats_df.empty:
                     f.write(f"### Module {module_code} Statistics\n")
                     f.write(stats_df.to_markdown())
                     f.write("\n\n")
                 else:
                     f.write(f"*No statistics available for Module {module_code}.*\n\n")
            
            summary_report_path = report_path
            status_messages.append(f"Successfully generated: {os.path.basename(report_path)}")

    except Exception as e:
        st.error(f"Error generating summary report: {e}")
        status_messages.append(f"ERROR: Summary report generation failed: {e}")

    # --- Completion --- 
    module_G_time = time.time() - module_G_start
    status_messages.append(f"Module G completed in {module_G_time:.2f} seconds.")
    if not module_error:
        st.success(f"Module G finished in {module_G_time:.2f}s.")
    else:
        st.error(f"Module G finished with errors in {module_G_time:.2f}s.")

    # --- Return Results --- 
    return {
        "demand_gif_path": demand_gif_path,
        "od_gif_path": od_gif_path,
        "summary_report_path": summary_report_path,
        "status_messages": status_messages,
        "error": module_error # Decide if reporting errors should halt anything
    } 