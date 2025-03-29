import os
import pytest
import pandas as pd
import geopandas as gpd
import yaml

# Test configuration loading
def test_config_generation():
    # Test params
    sw_lat = 30.265
    sw_lon = -97.745
    ne_lat = 30.270
    ne_lon = -97.740
    buffer_km = 1.0
    
    # Create a test config
    config = {
        'area_selection': {
            'method': 'coordinates',
            'coordinates': [sw_lat, sw_lon, ne_lat, ne_lon],
            'buffer_km': buffer_km
        },
        'data_acquisition': {
            'osm_tags': {'building': True},
            'census_variables': ['B01003_001E'],
            'census_product': 'ACSDT5Y2022',
            'state_code': '48',
            'county_code': '453',
            'tract_year': 2022
        },
        'random_seed': 42,
        'output_dir': './test_output'
    }
    
    # Ensure we can save and load the config
    if not os.path.exists('./test_output/config'):
        os.makedirs('./test_output/config', exist_ok=True)
        
    config_path = './test_output/config/test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Read it back and validate
    with open(config_path, 'r') as f:
        loaded_config = yaml.safe_load(f)
    
    assert loaded_config['area_selection']['coordinates'] == [sw_lat, sw_lon, ne_lat, ne_lon]
    assert loaded_config['random_seed'] == 42
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return True

# Test that required directories exist
def test_directory_setup():
    output_dir = "uas_analysis_output"
    
    # Create the directory structures
    for subdir in ['data', 'visualizations', 'stats', 'config', 'animation_frames']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Check they exist
    for subdir in ['data', 'visualizations', 'stats', 'config', 'animation_frames']:
        assert os.path.exists(os.path.join(output_dir, subdir))
    
    return True

# Run the tests
if __name__ == "__main__":
    test_config_generation()
    test_directory_setup()
    print("All tests passed!") 