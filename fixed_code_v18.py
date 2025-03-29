import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping, LineString
from shapely.ops import transform, unary_union
import osmnx as ox
import cenpy
from cenpy import products
from datetime import datetime, timedelta
import random
import math
from scipy.stats import poisson
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
import time
import json
import pyproj
from fuzzywuzzy import fuzz, process
import yaml
from PIL import Image
import glob
import requests
import zipfile
import io
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

# --- Optional: Add your Census API Key ---
print("Checking for Census API Key...")
census_api_key = os.environ.get("CENSUS_API_KEY", None) # Standard env variable name
# census_api_key = "YOUR_API_KEY_HERE" # Uncomment and paste key here
if census_api_key and census_api_key != "YOUR_API_KEY_HERE":
    # Cenpy uses the environment variable directly if set
    print("Using Census API Key found in environment variable.")
else:
    census_api_key = None # Explicitly set to None if not found/set
    print("Census API Key not set. Using default access (may be rate-limited).")

print("Environment setup complete.") 