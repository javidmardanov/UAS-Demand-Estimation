**MEMORANDUM**

**To:** Future Gemini Instance (Working on UAS Demand/Routing Streamlit Tool)
**From:** Past Gemini Instance (Completed Colab Prototyping - v20.2 / Streamlit Refactor v0.1)
**Date:** April 10, 2025 (Simulated)
**Subject:** Project Handover & Lessons Learned - UAS Delivery Demand Modeling & O-D Matching (Streamlit Version)

**1. Project Overview & Goal**

Welcome back! You're picking up the project to create a comprehensive, configurable Streamlit application for simulating and visualizing Unmanned Aerial System (UAS) last-mile delivery demand and generating origin-destination (O-D) datasets. The goal remains an award-winning, scientific tool suitable for PhD research and presentations. We successfully prototyped the core workflow in Google Colab (v20.3 code) and have begun refactoring into a modular Streamlit app.

**2. Core Workflow & Modules (Streamlit Structure)**

The application (`app.py`) calls functions sequentially from the `modules/` directory. Understand this flow:

*   **A. Setup & Area Definition (`modules/area_selection.py`):** 
    *   User defines study rectangle **either by drawing on an interactive map or entering coordinates** via sidebar widgets (controlled by a radio button). Includes a warning about selecting large areas.
    *   Code finds intersecting Census tracts by downloading TIGER/Line shapefiles for the specified state/year. **Crucially, it handles varying shapefile formats** by searching for common `GEOID` column names (e.g., `GEOID`, `GEOID10`, `GEOID20`) or constructing it from `STATEFP`, `COUNTYFP`, `TRACTCE`, ensuring it's a string.
    *   Calculates the **union** of valid intersecting tracts. If union fails or no tracts are found, **it robustly falls back** to using the original selection rectangle as the union area for subsequent steps (like buffering).
    *   Defines a **buffered union area** based on the calculated (or fallback) union and user-specified buffer distance.
    *   Fetches OSM features (**via `osmnx` API**) within the buffer. Handles potential `MultiIndex` columns from `osmnx` and **standardizes a `unique_id`** for each feature (checking `osmid`, `element_type_osmid`, `id`, or generating one). Applies basic geometry cleaning (`buffer(0)`).
    *   Loads/filters demographic data for intersecting tracts from a user-provided local CSV (`us_census_tracts_2023.csv`). **It identifies the `GEOID` column in the CSV** by checking common names in the header (e.g., `GEOID`, `geoid`, `Id`) and enforces string `dtype`. *(Current limitation: CSV path is hardcoded in the module, should be moved to config).* Checks for required data columns specified in the config. Calculates derived metrics like population density (using tract area from shapefile) and employment rate, handling missing source columns.
    *   Outputs key GeoDataFrames (`selected_gdf`, `tracts_gdf`, `union_gdf`, `buffered_gdf`, `osm_gdf`), the filtered demographic `census_data_df`, estimated `utm_crs`, statistics, and status messages.
*   **B. Data Acquisition:** Largely integrated into Module A now. OSM features are fetched in Module A. Census *boundaries* are fetched in Module A. Census *demographics* are loaded from the local CSV in Module A.
*   **C. Building Classification & Store ID (`modules/classification.py` - TBD):** Classifies *all* fetched buildings (in buffer) as 'residential'/'non-residential' using rules. Identifies potential stores ('origins') within the *buffered union area* based on tags/keywords.
*   **D. Height Estimation (`modules/height.py` - TBD):** Estimates height (`height_m`) for buildings located *within the union of tracts*, using tags, k-NN, or defaults. Records the estimation source.
*   **E. Population Allocation (`modules/population.py` - TBD):** Allocates Census tract population (from the filtered local CSV data) to *residential* buildings *within the union of tracts*, based on relative building volume (`building_area_m2 * height_m`). Calculates `estimated_households`.
*   **F. Demand Modeling (`modules/demand.py` - TBD):** Calculates base demand rate (`demand_rate`) for residential buildings *within the union* using demographics (from CSV) and configurable coefficients. Simulates detailed delivery events (`delivery_events.csv`) over a specified period. Generates hourly *demand animation frames*.
*   **G. O-D Matching & Dataset Generation (`modules/od_matching.py` - TBD):** Matches *all* simulated delivery events (destinations *within the union*) to origin stores (*within the buffer*) using a chosen method. Calculates straight-line distances. Generates the final `routing_dataset.csv` for a specific user-selected hour. Generates hourly *O-D animation frames*.
*   **H. Reporting & Visualization (`modules/reporting.py` or integrated - TBD):** Consolidates statistics, generates maps/plots/GIFs, provides download links.

**3. Data: Sources & Key Files**

*   **Inputs:**
    *   User Area Definition (Coordinates via `st.sidebar` widgets)
    *   User Configuration (`code_v20_3_config.yaml` - base, potentially modified by sidebar)
    *   **User Provided Census Data:** `us_census_tracts_2023.csv` (Assumed to be in the root directory, containing GEOID and necessary ACS variables for *all* US tracts).
    *   OSM API (via `osmnx` in `modules/area_selection.py`)
    *   Census TIGER/Line Shapefiles (via direct download URL in `modules/area_selection.py`)
*   **Key Intermediate/Output Files (in `output_dir/data/`):**
    *   `area_selection.geojson`: User's initial rectangle.
    *   `census_tracts.geojson`: Tract boundaries intersecting the selection.
    *   `union_of_tracts.geojson`: Polygon(s) defining the core analysis area.
    *   `buffered_area.geojson`: Polygon used for OSM fetching.
    *   `initial_osm_features.geojson`: Raw OSM data within the buffer.
    *   `census_data.csv`: **Filtered and processed** demographic data for *intersecting* tracts (derived from the user's `us_census_tracts_2023.csv`).
    *   `classified_buildings.geojson`: *All* buildings from buffer + `residential` (yes/no), `is_store` (bool), `store_type` attributes. Includes `unique_id`.
    *   `stores.geojson`: **Point** features (centroids) for stores in buffer + `store_id`, `store_type`. `unique_id` links back to `classified_buildings.geojson` for footprint geometry.
    *   `buildings_with_heights.geojson`: Buildings *within the union* + `height_m`, `height_source`, `knn_details`.
    *   `buildings_with_population.geojson`: Buildings *within the union* + `allocated_population`, `estimated_households`.
    *   `buildings_with_demand.geojson`: Buildings *within the union* + `demand_rate`, adjustment factors, `simulated_total_deliveries`.
    *   `delivery_events.csv`: The core simulation output: `building_unique_id`, `timestamp` (datetime), `latitude`, `longitude`.
    *   `routing_dataset_detailed_weekly.geojson`: **LineString** features for *all* matched O-D pairs over the simulation week, with detailed attributes.
    *   `routing_dataset.csv`: **Final deliverable.** Subset of matched deliveries for the user-specified hour (`simulation_hour_for_matching`), formatted as: `order_id, timestamp (unix), origin_id (store_id), origin_coordinates ([lat,lon]), destination_id (building_unique_id), destination_coordinates ([lat,lon])`.

**4. Key Logic & Formulae**

*   **Height Estimation (`extract_known_height`, `estimate_missing_heights`):**
    *   Priority: `height` tag (parsed robustly) > `building:levels` tag (`levels * meters_per_level`) > k-NN > default.
    *   k-NN: Uses `KNeighborsRegressor`, features typically `building_area_m2`, `centroid_x`, `centroid_y` (in UTM). Configurable `k` and `use_area_feature`.
*   **Population Allocation (`allocate_population`):**
    *   For residential buildings within the union only.
    *   Volume: `Vol = Area * Height` (ensure >= 1).
    *   Share: `Share_bldg = Vol_bldg / Sum(Vol_bldgs_in_tract)`.
    *   Initial Alloc: `AllocPop_bldg = TractPop * Share_bldg * PopScaleFactor`.
    *   Scale to Match Census: `ScaleFactor_tract = TargetPop_tract / Sum(AllocPop_bldg_in_tract)`, `FinalPop_bldg = AllocPop_bldg * ScaleFactor_tract`.
    *   Households: `EstHH = round(FinalPop_bldg / AvgHHSize).clip(lower=1)`.
*   **Demand Rate (`calculate_demand_rate`):**
    *   For residential buildings within the union only.
    *   **Formula:** `DemandRate = BaseRatePerHH * EstHH * Adj_Inc * Adj_Dens * Adj_HHSize * Adj_Emp`
    *   Where `Adj = max(0.2, 1 + Coef * (Value/RefValue - 1))` (handles division by zero if RefValue=0).
    *   Inputs: `estimated_households`, demographic values (income, density, etc.) joined from Census data.
    *   Configurable: `BaseRatePerHH`, all `Coef` values, all `RefValue` values.
*   **Demand Simulation (`simulate_deliveries`):**
    *   Calculates `AdjDailyRate = DemandRate * DayFactor * MonthFactor`.
    *   Generates `NumDeliveries = np.random.poisson(AdjDailyRate.clip(lower=0))`.
    *   Assigns hour using `np.random.choice` based on normalized `hourly_distribution`.
    *   Assigns precise timestamp (random minute/second).
*   **O-D Matching (`match_deliveries` - Market Share Weighted):**
    *   Normalize configured `market_shares` based on *available store types* found in `stores.geojson`.
    *   For each delivery event:
        1.  Assign `assigned_store_type` via `np.random.choice` using normalized shares.
        2.  Filter `stores_gdf` to get `possible_origins` of that type (must have valid geometry).
        3.  If `possible_origins` exist: Calculate distances (UTM) from delivery point to all `possible_origins`. Find the index label of the minimum distance: `min_dist_idx_label = distances.idxmin()`. Select `chosen_origin = possible_origins.loc[min_dist_idx_label]`.
        4.  Else (Fallback): Find the overall nearest *valid* store of *any* type.
        5.  Record `origin_id`, `origin_geometry`, etc.

**5. Configuration & User Control (`config.yaml` & Streamlit Widgets)**

*   The Streamlit app loads `code_v20_3_config.yaml` as a base.
*   Key parameters (area coords, buffer, base demand rate - more to be added) are exposed in the `st.sidebar`.
*   The `app.py` updates a *copy* of the loaded config with sidebar values before passing it to the modules for a specific run.
*   Saving the *run-specific* config is handled by the Colab code logic (saving `used_config.yaml`).

**6. Visualizations & Outputs**

*   **Static Maps (PNG):**
    *   Fig 1: Selection Rectangle
    *   Fig 2: Tracts/Union/BuildingsInUnion/Selection
    *   Fig 3: Buffer/AllBuildings/StoreFootprints
    *   Fig 4: Classification Map (Buildings in Union, colored Res/NonRes; Store Points in Buffer)
    *   Height Map (Buildings in Union, colored by height_m)
    *   Population Map (Res Buildings in Union, colored by allocated_population)
    *   Demand Map (Res Buildings in Union, colored by demand_rate)
    *   O-D Map (Static, for `simulation_hour_for_matching`, showing Store Footprints, Dest Footprints colored by count, connecting lines)
*   **Plots (PNG):** Height Distribution, Population Distribution, Demand Time Series, Temporal Factor bar charts.
*   **Animation Frames (PNG):** Saved to `animation_frames/`. Prefixes `demand_` and `od_`. Include legends/titles *on the frames*.
*   **GIFs (in `visualizations/`):** `demand_animation.gif`, `od_animation.gif`. **These are key final outputs for the user.**
*   **Data Files (in `data/`):** GeoJSONs at each stage, CSVs (Census, Events, Final Routing).
*   **Stats (in `stats/`):** CSV summary for each module.
*   **Config (in `config/`):** Saved `used_config.yaml`.
*   **Report (in `output_dir/`):** `summary_report.txt` (or HTML/Markdown).

**7. IMPORTANT CAVEATS & Lessons Learned (Streamlit Version)**

*   **`cenpy` Dependency Issue:** Direct use of `cenpy` to fetch demographic data caused `urllib.error.HTTPError: 403 Forbidden` during `cenpy`'s own import process (fetching internal FIPS tables). Adding a User-Agent header did not resolve this import-time issue. **Workaround:** Require the user to provide a comprehensive local CSV (`us_census_tracts_2023.csv`) with all necessary ACS data. Module A now reads/filters this file instead of calling the `cenpy` API.
    *   **Implication:** Requires user setup (downloading potentially large file). The application must correctly identify the `GEOID` column (assumed common names like 'GEOID', 'geoid', 'Id') and required data columns (e.g., 'B01003_001E'). Robust error handling is needed if the file or columns are missing/incorrect. Performance reading/filtering the large CSV needs monitoring.
*   **File Size Limits:** Internal tools (like `read_file`) may have size limits. Reading even parts of the large `us_census_tracts_2023.csv` failed. Rely on `pandas` within the module code for reading large files.
*   **CRS:** Still critical. Ensure consistency (WGS84 for storage/input, projected CRS like UTM for planar operations).
*   **YAML Loading:** As before, check indentation and `Null` handling. **Prefer block style over inline `{...}` for complex dictionaries (like `hourly_distribution`) to avoid parsing issues.**
*   **Census Data (Local CSV):** Ensure the user-provided CSV has the correct `GEOID` format (string matching TIGER/Line shapefiles) and includes *all* variables specified in `config['data_acquisition']['census_variables']`. The code now attempts to **match required variables flexibly** (e.g., finding 'B01003_001E' within a column named 'total_pop_B01003_001E') and renames columns internally. However, the user *must* ensure columns containing these codes exist. **(Improvement Needed: The path to this CSV is currently hardcoded in `modules/area_selection.py` and should be moved to `config.yaml`.)**
*   **OSM Data:** Still messy. `osmnx` fetches happen in Module A. The code standardizes `unique_id` creation but relies on OSM tags being present. Handles `MultiIndex` columns from `osmnx`.
*   **Building Classification Config:** The keywords defined in `config.yaml` under `building_classification` (e.g., `residential_building_tags`, `store_tags`) are critical. If these are empty or incorrect, Module B will not identify buildings appropriately, causing downstream modules (D, E, F) to be skipped or produce zero results.
*   **GeoPandas Operations:**
    *   `clip`: Ensure clip mask (e.g., `union_gdf`) is valid.
    *   `sjoin`/`sjoin_nearest`: Watch out for duplicate matches (use `drop_duplicates`). Ensure CRS match. `sjoin_nearest` can be slow on large datasets without spatial indexes (consider adding `.sindex` if needed, though maybe overkill for this scale). `.loc` vs `.iloc`: Remember `idxmin()` returns the *label*, so use `.loc[label]`.
    *   `unary_union`: Can be slow for many complex polygons. Check validity of result.
    *   `buffer(0)`: Useful trick to fix some invalid geometries, but not guaranteed.
*   **File I/O:** Modules save intermediate files to `output_dir`. Ensure paths are correct.
*   **Module Dependencies:** Still sequential. Errors in Module A will prevent subsequent modules.
*   **Visualization:** Module A generates static setup maps. `app.py` displays them. Later modules will need similar patterns.
*   **Performance:** Reading/filtering the large Census CSV in Module A is a new potential bottleneck.
*   **Streamlit Specifics:**
    *   `st.set_page_config()` MUST be the first `st.` call after imports.
    *   Widget tracebacks can be misleading (e.g., `KeyError` masking a type mismatch). Verify config types match widget parameters (`value`, `step`, `min/max_value`).
    *   Address deprecation warnings (e.g., `use_column_width` -> `use_container_width`) promptly.

**8. Next Steps**

*   **Configuration Refinement:**
    *   **Move Census CSV path** from `modules/area_selection.py` to `config.yaml`.
    *   **Populate `building_classification` keywords** in `config.yaml` with sensible defaults/examples.
    *   Consider adding more simulation parameters (e.g., k-NN neighbors, O-D method) to sidebar widgets in `app.py`.
*   **Thorough End-to-End Testing & Refinement.** 
    *   Test with different areas and configurations.
    *   Verify outputs (maps, stats, data files) for correctness.
    *   Review code for clarity and potential optimizations.
*   **Documentation:** Update `README.md` with usage instructions, dependency list (including `streamlit-folium`), and explanation of the required Census CSV format.

---