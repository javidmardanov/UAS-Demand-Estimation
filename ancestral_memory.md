# Ancestral Memory Log

This file records key challenges, bugs, fixes, and lessons learned during the development of the UAS Demand/Routing Streamlit tool.

--- 

**2025-04-10: Cenpy Import Error & Workaround**

*   **Problem:** Encountered `urllib.error.HTTPError: HTTP Error 403: Forbidden` during the import phase (`from cenpy import products`). The traceback indicated the error occurred when `cenpy.explorer.fips_table` tried to fetch an internal lookup table (likely FIPS codes) via `pandas.read_csv`, which uses `urllib`. Adding a User-Agent via `urllib.request.install_opener` in `app.py` did *not* resolve this specific import-time issue, suggesting the fetch happens too early or bypasses the installed opener.
*   **Workaround:** User will manually download the required Census data (ACS 5-Year 2023) for all US tracts into a local CSV file named `us_census_tracts_2023.csv`.
*   **Resolution:** Refactor `modules/area_selection.py` to remove the `cenpy` dependency for fetching demographic data. Instead, it will read the data from `us_census_tracts_2023.csv`, filter it based on the GEOIDs of intersecting tracts, and then process/merge it.
*   **Implication:** Removes a fragile external dependency during runtime/import. Requires user to provide the large base CSV file. Increases importance of documenting the expected CSV format (esp. GEOID). May impact performance if CSV reading/filtering isn't optimized.

**2025-04-10: File Size Limitation & CSV Structure Assumption**

*   **Problem:** The `read_file` tool failed to read the header of `us_census_tracts_2023.csv` because the file exceeded the 2MB size limit.
*   **Resolution:** Modified `modules/area_selection.py` to proceed *without* reading the header directly. It now assumes a standard structure:
    *   It attempts to identify the GEOID column by checking common names (`GEOID`, `geoid`, `Id`, `id`, `geo_id`) by reading *only* the header row using `pd.read_csv(..., nrows=0)`.
    *   If a GEOID column is found, it reads the *entire* CSV using `pandas.read_csv`, explicitly setting the `dtype` of the found GEOID column to `str` for reliable merging.
    *   It adds checks to ensure required data columns (defined in `config.yaml`) exist in the loaded DataFrame.
*   **Implication:** The tool now relies on the user providing a CSV where the GEOID column matches one of the expected names and is formatted correctly for joining with TIGER data (string). Required data columns must also be present. Clear documentation for the user about the expected CSV format is essential. Error handling was added to check for these conditions.

**2025-04-10: `modules/area_selection.py` Refactoring Insights**
*   **Context:** Post-`cenpy` workaround, Module A (`modules/area_selection.py`) was reviewed for robustness and insights.
*   **Key Insights & Refinements:**
    *   **Robust GEOID Handling (TIGER/Line):** Added logic to automatically find the `GEOID` column in downloaded Census tract shapefiles, checking common variations (`GEOID`, `GEOID10`, `GEOID20`) and allowing construction from `STATEFP`, `COUNTYFP`, `TRACTCE`. Ensures `GEOID` is consistently treated as a string.
    *   **Union Fallback:** Implemented a fallback mechanism. If calculating the union of intersecting tracts fails (e.g., no tracts found, invalid geometries), the module now defaults to using the original user-defined `selected_gdf` as the `union_gdf` for subsequent steps (like buffering and clipping). This prevents errors downstream.
    *   **Standardized OSM ID:** Added logic to create a reliable `unique_id` for OSM features fetched via `osmnx`. It checks for `osmid`, then `element_type_osmid`, then `id`, and finally generates a unique ID if none are found. Also handles potential `MultiIndex` columns returned by `osmnx`.
    *   **Robust GEOID Handling (Local CSV):** Added logic to automatically find the `GEOID` column in the user-provided `us_census_tracts_2023.csv`, checking common names (`GEOID`, `geoid`, `Id`, `id`, `geo_id`). Enforces string `dtype` for consistency with shapefile GEOIDs. **Crucially, the *path* to this CSV is currently hardcoded in the module.**
    *   **Derived Metrics Calculation:** Explicitly calculates population density using tract area (from geometry) and handles potential missing columns during other derived metric calculations (e.g., employment rate).
*   **Documentation Needs:**
    *   The user must be clearly informed about the expected `GEOID` formats in both the TIGER/Line shapefiles and the local CSV.
    *   The local CSV path needs to be moved from the module code to the `config.yaml` file for better configuration management.
*   **Implication:** Module A is now more resilient to variations in input data formats (shapefiles, CSVs) and potential processing failures (union calculation). Dependency on correct user-provided CSV structure and required columns remains. Configuration needs improvement (CSV path).

**2025-04-10: Streamlit Startup Errors & Debugging Lessons**
*   **Problem 1:** Application failed on startup with an internal Streamlit error related to `st.set_page_config` (`metrics_util.py`).
    *   **Cause:** `st.set_page_config` was not the absolute first Streamlit command executed in `app.py`, even though preceding code didn't directly use `st`.
    *   **Resolution:** Moved the `st.set_page_config(...)` block to immediately follow the initial `import` statements.
*   **Problem 2:** After fixing Problem 1, the app failed with a `KeyError: 'area_selection'` when defining a sidebar widget (`st.number_input` for `buffer_km`), despite debug prints confirming the `'area_selection'` key existed in `st.session_state.config`.
    *   **Cause:** Misleading traceback. The actual issue was a type mismatch. `buffer_km` was loaded as an integer (`1`) from `config.yaml`, but the `st.number_input` widget was configured with float `min_value`, `max_value`, and `step` (e.g., `0.1`). This type inconsistency likely caused an internal error misinterpreted as a `KeyError`.
    *   **Resolution:** Changed `buffer_km` in `code_v20_3_config.yaml` from `1` to `1.0` (float) to match widget parameters.
*   **Lessons Learned:**
    *   **Strict `set_page_config` Order:** Adhere rigidly to placing `st.set_page_config` as the very first Streamlit command.
    *   **Misleading Tracebacks:** Be skeptical of tracebacks originating from widget rendering or session state if they contradict known variable states. The reported error type (`KeyError`) might mask the real issue (like type mismatch).
    *   **Widget Type Sensitivity:** Ensure data types loaded from configuration files match the types expected by the corresponding Streamlit widget parameters (`value`, `min_value`, `max_value`, `step`).
    *   **Debug Prints:** Use `print()` statements strategically to verify variable state/types immediately before the point of error when tracebacks are confusing.

**2025-04-11: Added Dual Area Input Methods**
*   **Context:** User requested the ability to define the study area either by drawing on a map or by entering coordinates manually.
*   **Implementation:**
    *   Added `streamlit-folium` and `folium` as dependencies (`pip install streamlit-folium folium`).
    *   Added a radio button (`st.sidebar.radio`) to `app.py` allowing user to select "Enter Coordinates" or "Draw on Map".
    *   Conditionally display UI elements:
        *   If "Enter Coordinates" selected: Show Min/Max Lat/Lon `st.number_input` widgets in the sidebar.
        *   If "Draw on Map" selected: Display an interactive Folium map (using `st_folium`) in the main application area, configured with the `folium.plugins.Draw` tool enabled only for rectangles.
    *   Added logic to process the `st_folium` map output, extract bounds from the `last_active_drawing`, and store them in `st.session_state.drawn_bounds`.
    *   Updated the `run_simulation` logic to check the selected input method:
        *   If map: Use `st.session_state.drawn_bounds` (error if None).
        *   If coordinates: Use values from the sidebar `st.number_input` widgets (error if invalid, e.g., min >= max).
        *   Pass the validated coordinates to Module A.
    *   Included an `st.warning` near the map/coordinate inputs advising against selecting excessively large areas.
*   **Implication:** Provides a more flexible and user-friendly way to define the study area. Adds new dependencies. Increases complexity slightly in `app.py` due to conditional UI and logic.

**2025-04-11: YAML Parsing & CSV Column Flexibility Lessons**
*   **Context:** After implementing dual area input, the simulation pipeline ran end-to-end but produced unexpected warnings and skipped core modules.
*   **Problem 1:** Persistent, contradictory warnings about `hourly_distribution` sum despite config values summing to 1.0.
    *   **Cause:** Debug prints revealed `pyyaml` was misinterpreting the inline dictionary format (`{0: 0.0, 1: 0.0, ...}`) in `config.yaml`, reading keys as floats (`0.0`, `60.0`, etc.) and values as `None`. This led to incorrect sum calculations in `app.py`.
    *   **Resolution:** Reformatted `hourly_distribution` in `code_v20_3_config.yaml` to use the standard multi-line block style, which parsed correctly.
    *   **Lesson:** Use standard YAML block format for dictionaries, especially complex ones, as inline formats can be sensitive to parsing nuances.
*   **Problem 2:** Module A failed, reporting missing Census variable columns (e.g., `B01003_001E`) even though the CSV header contained descriptive names including these codes (e.g., `total_population_aka_B01003_001E`).
    *   **Cause:** The check in `modules/area_selection.py` initially required exact column name matches.
    *   **Resolution:** Modified the logic to:
        1. Check for exact match first.
        2. If no exact match, check if the required variable code is a *substring* within any existing column header.
        3. If a containing column is found, *rename* it internally to the standard variable code (e.g., `B01003_001E`).
        4. Raise an error only if the code isn't found exactly or as a substring.
    *   **Lesson:** Input data schemas can vary. Build flexibility into column checking (e.g., substring matching) and internal renaming for robustness, or clearly document exact header requirements.
*   **Problem 3:** Core simulation (Modules D, E, F) was skipped.
    *   **Cause:** Module B ran successfully but classified 0 residential buildings and 0 stores because the relevant keyword lists (`residential_building_tags`, `store_tags`, etc.) in the `building_classification` section of `config.yaml` were empty.
    *   **Lesson:** Correct code execution doesn't guarantee meaningful simulation. The quality and completeness of the configuration file (`config.yaml`) directly drive the simulation's behavior and results.
*   **Other:** Addressed `st.image` deprecation warnings (`use_column_width` -> `use_container_width`).

**2025-04-11: Module E TypeError during Simulation**
*   **Problem:** Module E failed during the delivery event simulation loop with `TypeError: only integer scalar arrays can be converted to a scalar index`.
*   **Cause:** The code attempted to create timestamps using `datetime.replace(hour=hour_array, minute=minute_array, second=second_array)`. The `datetime.replace` method expects scalar integer arguments, not NumPy arrays.
*   **Resolution:** Modified the timestamp creation logic within the simulation loop in `modules/demand.py`. It now constructs a temporary Pandas DataFrame containing separate columns for year, month, day, hour, minute, and second (using the NumPy arrays for the time components). `pd.to_datetime()` is then called on this DataFrame to generate the timestamps correctly in a vectorized manner.
*   **Implication:** Required understanding the specific limitations of `datetime.replace` vs. the capabilities of `pd.to_datetime` for vectorized operations. The fix uses a standard Pandas pattern.

**2025-04-11: Module E Geometry TypeError during Simulation**
*   **Problem:** After fixing the timestamp `TypeError`, Module E failed again during the simulation loop with the error `y attribute access only provided for Point geometries` (or similar, indicating access to .x or .y).
*   **Cause:** The simulation code was attempting to extract latitude (`.y`) and longitude (`.x`) directly from the `geometry` column of the `events_today_df`. This DataFrame contained the original building geometries, which were likely Polygons (footprints), not Points. The `.x` and `.y` attributes are only valid for Point geometries.
*   **Resolution:** Modified the `simulate_deliveries` function in `modules/demand.py`. Before accessing `.y` and `.x`, the code now explicitly calculates the `centroid` of the geometry column: `centroids = events_today_df.geometry.centroid`. The latitude and longitude are then extracted from these `centroids`: `events_today_df['latitude'] = centroids.y`, `events_today_df['longitude'] = centroids.x`.
*   **Implication:** Reinforces the need to be mindful of geometry types in GeoPandas operations. Centroids must be calculated explicitly when point coordinates are needed from polygon features.

**2025-04-11: Module E KeyError during Simulation**
*   **Problem:** After fixing the geometry error, Module E failed again during the simulation loop with `KeyError: "['building_unique_id'] not in index"`.
*   **Cause:** The code selected columns `['order_id', 'building_unique_id', 'timestamp', 'latitude', 'longitude']` from the `events_today_df` to append to the final list of events. However, the DataFrame at that point contained the building identifier under the column name `unique_id` (inherited from earlier modules), not `building_unique_id`.
*   **Resolution:** Modified the beginning of `run_module_e` in `modules/demand.py`. When creating the initial `buildings_for_demand` DataFrame, the `unique_id` column is now explicitly renamed: `.rename(columns={'unique_id': 'building_unique_id'})`. This ensures the correct column name propagates through the simulation logic.
*   **Implication:** Highlights the importance of consistent column naming conventions across modules, especially for key identifiers used in merging or selection.

**2025-04-11: Module E KeyError - Select & Rename On-the-Fly**
*   **Problem:** The `KeyError: 'building_unique_id'` persisted in Module E's simulation loop, even after trying to rename the column inplace just before appending.
*   **Cause:** The exact reason for the column name not being found remained elusive, possibly due to Pandas DataFrame view/copy behavior within the loop.
*   **Resolution:** Implemented a "select and rename on-the-fly" strategy.
    1. Removed the inplace rename of `unique_id` within the loop for `events_today_df`.
    2. Modified the append step: `df_to_append = events_today_df[COLS].rename(columns={'unique_id': 'building_unique_id'})`, where `COLS` includes `'unique_id'`. 
    3. Appended this `df_to_append` to `all_events`: `all_events.append(df_to_append)`.
    4. Ensured subsequent operations (concat, groupby, saving empty DF) correctly handled the `building_unique_id` column name.
*   **Implication:** This avoids modifying `events_today_df` inplace within the loop and creates a new, correctly formatted DataFrame specifically for the append operation. This is a more robust way to handle potential view/copy issues related to column renaming in loops.

**2025-04-11: Module F TypeError - Finalizing Datasets**
*   **Problem:** Module F completed the matching logic (falling back to proximity) but failed during the final dataset preparation with `TypeError: 'NoneType' object is not iterable`.
*   **Cause:** The error occurred when creating the final `routing_dataset_df`, specifically within the `.apply()` lambda functions used to extract `origin_coordinates` and `destination_coordinates`. If either the `origin_geometry` (Point) or the `geometry` (LineString, representing the O-D line) was `None` or invalid for a given row (e.g., due to a failed match or issue creating the LineString), accessing attributes like `.x`, `.y`, or `.coords` on the `None` object triggered the TypeError.
*   **Resolution:** Made the lambda functions more robust. 
    *   For origin points: Changed check to `lambda p: [...] if p and not p.is_empty else [None, None]`.
    *   For destination points (from OD line): Changed check to `lambda line: [...] if line and not line.is_empty and len(line.coords) > 1 else [None, None]`.
*   **Implication:** Ensures that coordinate extraction handles potential missing or invalid geometries gracefully, preventing the TypeError during final dataset creation.

**2025-04-11: Module F TypeError - Enhanced Geometry Checks**
*   **Problem:** The `TypeError: 'NoneType' object is not iterable` persisted in Module F's finalization step, even after adding basic checks for None geometries.
*   **Cause:** Suspected that geometries might be present but invalid (e.g., empty Points or Lines) in a way that caused `.distance()` or `LineString()` to fail, and previous checks were insufficient.
*   **Resolution:** Implemented more rigorous checks:
    *   For distance calculation: Created a boolean mask `valid_geoms` that requires both origin and destination geometries to be `.notna()` AND `~.is_empty`. Distance is only calculated where this mask is True.
    *   For LineString creation: The `.apply()` lambda now checks if both origin and destination geometries are not None AND `not .is_empty` before attempting `LineString` creation.
*   **Implication:** These stricter checks should prevent errors arising from invalid or empty geometries generated during the matching or merging steps.

**2025-04-11: Module F CRS Error on Empty Lines**
*   **Problem:** After fixing geometry checks, Module F failed with `Assigning CRS to a GeoDataFrame without a geometry column is not supported...` when the matching resulted in zero valid O-D LineStrings.
*   **Cause:** The code filtered out all rows where `od_line` was None, resulting in an empty `valid_lines_gdf`. The `else` block then tried to create an empty GeoDataFrame using `gpd.GeoDataFrame(crs=WGS84)`, which failed because GeoPandas requires knowing the geometry column name, even if the DataFrame is empty.
*   **Resolution:** Modified the `else` block to explicitly create the empty GeoDataFrame with the expected columns (including `od_line`) and specifying the geometry column name: `valid_lines_gdf = gpd.GeoDataFrame(columns=expected_cols, geometry='od_line', crs=WGS84)`.
*   **Implication:** Ensures the application handles the edge case where no valid O-D pairs are generated after matching, preventing a crash when trying to define an empty GeoDataFrame with a CRS.

**2025-04-11: Module F - Zero Valid O-D Lines - Destination Point Issue**
*   **Problem:** Even after ensuring origin (store) geometries were valid, Module F still produced zero valid O-D lines.
*   **Cause:** The only remaining possibility was that the destination geometries (derived from `delivery_events_df` latitude/longitude) were systematically invalid (None or empty) for all events, causing `LineString` creation to fail.
*   **Resolution:** Added two checks to `modules/od_matching.py` during the creation of `events_gdf`:
    1.  *Before* `gpd.points_from_xy`: Drop rows from `delivery_events_df` where `latitude` or `longitude` columns contain NaN/None.
    2.  *After* `gpd.points_from_xy`: Filter the resulting `events_gdf` to remove rows where the `geometry` column is None or empty.
    Both steps warn if rows are removed and raise an error if no valid events remain.
*   **Implication:** Ensures that both origin and destination points used for matching and LineString creation are validated (not None, not empty), addressing the root cause of the universal LineString failure.

**2025-04-11: Module F - Zero Valid O-D Lines - `sjoin_nearest` Failure**
*   **Problem:** Despite validating both origin and destination geometries, Module F still produced zero valid O-D lines.
*   **Cause:** The root cause was identified as the `gpd.sjoin_nearest` operation failing to find *any* nearby stores for *any* delivery events. This resulted in the `store_index` (originally `index_right`) column being all `NaN`. The subsequent merge operation correctly produced `None` for all `origin_` columns, leading to the failure of `LineString` creation for all rows.
*   **Resolution:** Added an explicit check immediately after the `sjoin_nearest` call in the `proximity` matching logic within `modules/od_matching.py`. This check verifies if `matched_gdf['store_index'].isna().all()` is True. If so, it raises a `ValueError` and displays messages indicating that no spatial matches were found, advising the user to check the spatial relationship between stores and demand locations or the store identification in Module B.
*   **Implication:** Correctly identifies and reports the failure condition when no origins can be spatially matched to destinations, preventing misleading downstream errors and providing clearer user feedback.

**2025-04-11: Module F - Merge Failure After Successful `sjoin_nearest`**
*   **Problem:** Diagnostic prints revealed that `sjoin_nearest` was successfully finding store indices, but the subsequent merge operation (using `left_on='store_index', right_index=True`) was failing to bring over the `origin_geometry`, resulting in `None`/empty geometries and preventing LineString creation.
*   **Cause:** The exact reason for the merge failure using the index was unclear, possibly related to index data types or subtle inconsistencies between the joined DataFrame and the store DataFrame.
*   **Resolution:** Refactored the `proximity` matching logic in `modules/od_matching.py`. The merge operation (`.merge(stores_proj.add_prefix(...))`) is now performed *immediately* after `sjoin_nearest` successfully identifies matches (and potentially filters out failed matches). The result of this immediate merge is then appended to the `matched_events_list`. The separate merge step after concatenating `matched_events_list` was removed.
*   **Implication:** Performing the merge immediately after the spatial join ensures the correct store details are linked using the index provided by `sjoin_nearest` at that specific point, avoiding potential index inconsistencies later.

**2025-04-11: Module F - KeyError: 'origin_geometry' during Finalization**
*   **Problem:** Module F failed during finalization with `KeyError: 'origin_geometry'` when calculating distances or creating LineStrings.
*   **Cause:** The `pandas.merge` operation combined with `GeoDataFrame.add_prefix()` did not correctly rename the geometry column from the stores GeoDataFrame (`stores_proj`). It remained named 'geometry', colliding with the destination 'geometry' column, and the expected 'origin_geometry' column was never created.
*   **Resolution:** Modified the merge step within the `proximity` block in `modules/od_matching.py`:
    1. Create a copy of `stores_proj` (`stores_to_merge`).
    2. Explicitly rename the `geometry` column to `origin_geometry` and other relevant columns (e.g., `store_id`, `store_type`) to have the `origin_` prefix in `stores_to_merge`.
    3. Removed the `.add_prefix()` call from the merge.
    4. Merged the `matched_gdf` (from `sjoin_nearest`) with the modified `stores_to_merge` DataFrame.
*   **Implication:** Guarantees that the merged DataFrame (`routing_dataset_full_gdf`) contains the correctly named `origin_geometry` column required for subsequent distance and LineString calculations.

**2025-04-11: Module F - Final Fix for `origin_geometry` KeyError**
*   **Problem:** The `KeyError: 'origin_geometry'` persisted. Debugging showed that `pandas.merge` was dropping the explicitly renamed `origin_geometry` column when merging the `sjoin_nearest` results with the store data.
*   **Cause:** Likely internal behavior of `pandas.merge` when merging two GeoDataFrames, where it mishandles the second geometry column.
*   **Resolution:** Completely refactored the proximity matching logic in `modules/od_matching.py`:
    1. Perform `sjoin_nearest` to get `matched_gdf` with `store_index`.
    2. Validate `store_index` and filter `matched_gdf` to keep only valid matches.
    3. **Removed the `.merge()` step.** Instead, directly created the required `origin_` columns (`origin_store_id`, `origin_store_type`, `origin_geometry`) in `matched_gdf` by using `store_index` to look up values in `stores_proj` via `.loc` (e.g., `matched_gdf['origin_geometry'] = stores_proj.loc[matched_gdf['store_index'], 'geometry'].values`).
    4. Directly assigned the resulting `matched_gdf` to `routing_dataset_full_gdf`.
    5. Removed the `matched_events_list` and `pd.concat` logic as it was only needed for multiple merge results.
*   **Implication:** This manual column assignment based on the join index bypasses the problematic `pandas.merge` behavior for geometry columns, ensuring the `origin_geometry` column exists with the correct data for the final processing steps.

**2025-04-11: Module F - Bug in Market Share Fallback Logic**
*   **Problem:** Module F failed with `ValueError: O-D Matching produced no results DataFrame.` even though proximity matching *should* have run as a fallback when 'market share weighted' was selected but shares were invalid.
*   **Cause:** The fallback logic *within* the `elif matching_method == 'market_share_weighted':` block correctly performed the `sjoin_nearest` operation but failed to assign the resulting DataFrame (`matched_gdf`) to the main variable `routing_dataset_full_gdf`. Consequently, `routing_dataset_full_gdf` remained empty, triggering the error check after the matching blocks.
*   **Resolution:** Modified the fallback section (`if not normalized_shares:`) within the `market_share_weighted` block in `modules/od_matching.py`. It now assigns the result of `sjoin_nearest` to `routing_dataset_full_gdf` and includes the necessary subsequent steps (checking join success, directly assigning origin columns using `.loc`) previously implemented only in the main `proximity` block.
*   **Implication:** Corrects the logic flow so that the fallback to proximity matching within the market share method correctly populates the main results DataFrame, allowing the module to proceed.

**2025-04-11: Module B Refactoring for Detailed Classification**
*   **Problem:** Module F consistently failed to produce O-D outputs, traced back to `sjoin_nearest` finding no matches. This suggested the input from Module B (classified buildings and stores) was inadequate, likely due to overly simplistic classification rules compared to the Colab prototype.
*   **Resolution:** Overhauled Module B (`modules/classification.py`) to implement the more sophisticated classification logic from the Colab code (`code_v20_3.py`):
    1.  **Config Update:** Added the detailed `rule_based_parameters` dictionary (containing area thresholds, specific keyword lists for tags, names, landuse, roads) from the Colab config into `code_v20_3_config.yaml`.
    2.  **Function Porting:** Copied the `classify_building_rules_detailed` function (which uses tags, area, road proximity, landuse context) and the more nuanced `identify_store_type` function from the Colab code into `modules/classification.py`.
    3.  **Integration:** Modified `run_module_b` to call these new functions, passing the necessary context (full `osm_gdf`, `buffer_polygon`, `utm_crs`). The simpler, previous classification logic was removed.
    4.  **Bug Fix:** Corrected a `TypeError` (`'GeometryArray' does not support reduction 'sum'`) introduced during integration, ensuring the count of residential buildings was calculated correctly using `.sum()` on the boolean condition.
*   **Implication:** Module B now uses a much more detailed set of rules, closer to the validated Colab prototype. This should significantly improve the accuracy of identifying residential buildings and stores, hopefully resolving the Module F matching failures by providing better quality input data.

**2025-04-11: Debugging Module B Residential Classification**
*   **Problem:** Module B (using detailed classification rules) identified zero residential buildings, causing downstream modules (D, E) to be skipped.
*   **Debugging:** 
    *   Initially suspected context rules (road/landuse). Commenting them out did not resolve the issue (still 0 residential).
    *   Enabled debug print statements within `classify_building_rules_detailed`.
    *   Output (`After Tag/Name Rules: {'no': 11376}`) revealed that **all buildings** were classified as 'no' immediately after the initial tag/name rules.
    *   Identified an overly strict rule: A loop checking `shop`, `amenity`, `office`, `tourism`, `leisure` tags forced any building with *any* value in these tags to 'no', even if the primary `building` tag was residential (e.g., `building=apartments`, `amenity=parking`).
*   **Fix:** Commented out the aggressive loop checking `shop`, `amenity`, etc. This allows buildings tagged as residential (e.g., `building=house`, `building=apartments`) to be correctly identified initially, letting subsequent area rules refine the classification.
*   **Result:** With the fix, the tag/name rules correctly identified residential buildings (`{'yes': 1605, 'unknown': 9503, 'no': 268}`), and subsequent area rules further refined this. The final residential count became non-zero, allowing the pipeline to proceed.

--- 