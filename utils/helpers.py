import geopandas as gpd
import pandas as pd
import numpy as np
import os

WGS84 = 'EPSG:4326'

def save_empty_gdf(filepath, driver="GeoJSON", crs=WGS84):
    """Saves an empty GeoDataFrame to a file."""
    print(f"Saving empty GDF: {filepath}")
    gdf = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs=crs)
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        gdf.to_file(filepath, driver=driver)
    except Exception as e:
        print(f"Error saving empty GDF {filepath}: {e}")

def save_empty_df(filepath, columns=None):
    """Saves an empty DataFrame to a CSV file."""
    print(f"Saving empty DF: {filepath}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.DataFrame(columns=columns if columns else []).to_csv(filepath, index=False)
    except Exception as e:
        print(f"Error saving empty DF {filepath}: {e}")

def style_map(ax, title):
    """Applies standard styling to a matplotlib map axis."""
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight='bold')
    # Set aspect ratio to equal only if geometry is present
    if not ax.collections and not ax.lines:
        print("  style_map: No geometry plotted, skipping aspect ratio.")
    else:
        ax.set_aspect('equal', adjustable='box')

    # Attempt to add North arrow
    try:
        ax.text(0.06, 0.94, 'N\n^', transform=ax.transAxes, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.7))
    except Exception as e:
        print(f"  style_map: Error adding North arrow: {e}")

    # Attempt to add scale bar
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        if x_range > 0 and y_range > 0 and np.isfinite(x_range) and np.isfinite(y_range):
            # Base scale bar length on ~10% of map width
            scale_len_map = x_range * 0.1
            # Sensible scale values in meters
            possible_scales = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            if scale_len_map > 1e-6: # Check if map width is reasonably large
                # Find the closest sensible scale
                scale_meters = min(possible_scales, key=lambda x: abs(x - scale_len_map))
                # Calculate the length of the scale bar in map units
                scale_len_display = scale_meters # Direct mapping since axis units are meters (Web Mercator)

                # Position the scale bar at the bottom left
                x_pos = xlim[0] + 0.05 * x_range
                y_pos = ylim[0] + 0.05 * y_range

                # Draw the scale bar line
                ax.plot([x_pos, x_pos + scale_len_display], [y_pos, y_pos], color='black', linewidth=3, transform=ax.transData)
                # Add text label below the center of the bar
                ax.text(x_pos + scale_len_display / 2,
                        y_pos - 0.02 * y_range, # Adjust vertical position slightly
                        f'{scale_meters}m',
                        horizontalalignment='center',
                        verticalalignment='top', # Align top of text with bottom edge of position
                        fontsize=10,
                        transform=ax.transData,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", lw=0, alpha=0.7))
            else:
                print("  style_map: Map width too small for scale bar.")
        else:
            print("  style_map: Invalid map limits for scale bar.")
    except Exception as e:
        print(f"  style_map: Error adding scale bar: {e}") 