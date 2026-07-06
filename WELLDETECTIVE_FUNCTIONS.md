# WellDetective Function Reference

**File:** `WellDetective.py`  
**Version:** 2.0  
**Original Authors (v1.0):** Eric Guiltinan, Javier Santos  
**Current Authors (v2.0):** Eric Guiltinan, Nash Taylor, James E. Lee  
**Maintainer:** James E. Lee (Los Alamos National Laboratory)  
**Last Updated:** 2026-06-09

---

## Table of Contents

- [Overview](#overview)
- [Initialization](#initialization)
- [Utilities](#utilities)
- [Data Loading](#data-loading)
- [Data Processing](#data-processing)
- [Filtering](#filtering)
- [Mapping & Gridding](#mapping--gridding)
- [Detection](#detection)
- [Export](#export)
- [Plotting](#plotting)
- [Summary Statistics](#summary-statistics)

---

## Overview

The `WellDetective` class provides comprehensive tools for processing geophysical magnetic survey data. It handles data loading, validation, filtering, gridding, anomaly detection, and visualization.

**Key Features:**
- Automatic file format detection and parsing
- Coordinate transformation and projection
- Flight path filtering and heading correction
- Magnetic field gridding and filtering
- Hotspot and orphan well detection
- Multi-format export (GeoTIFF, NetCDF)
- Advanced visualization tools

**Data Flow:**
1. **data_raw** → Original unprocessed data (preserved)
2. **data** → Working data with calculated fields (heading, mag totals)
3. **data_filtered** → Cleaned data (turns removed, segments filtered, heading corrected, projected)
4. **map** → Gridded and filtered spatial data (xarray and numpy arrays)

---

## Initialization

### `__init__(data: pd.DataFrame)`

**Line:** 112-124  
**Type:** Constructor

Initialize WellDetective with survey data.

**Parameters:**
- `data` (pd.DataFrame): Initial survey data with required columns ('Lat', 'Long', 'Mag')

**Returns:** None

**Creates:**
- `self.data_raw` - Copy of original data
- `self.data` - Working copy for processing
- `self.data_filtered` - Initialized as None
- `self.processing_log` - Empty list for tracking operations

**Example:**
```python
import pandas as pd
data = pd.read_csv('survey.csv')
wd = WellDetective(data)
```

---

## Class Constants

### Default Parameters

**Lines:** 86-111

The `WellDetective` class defines several class-level constants for default processing parameters:

**Column Name Mappings:**
- `LAT_COLUMNS` - Possible latitude column names
- `LON_COLUMNS` - Possible longitude column names  
- `MAG_COLUMNS` - Possible magnetic field column names
- `DATE_COLUMNS` - Possible date column names
- `MAG_X_COLUMNS`, `MAG_Y_COLUMNS`, `MAG_Z_COLUMNS` - Vector magnetometry components

**Processing Parameters:**
- `DEFAULT_HEADING_WINDOW = 20` - Window size for heading calculation
- `DEFAULT_HEADING_TOLERANCE = 20` - Tolerance (degrees) for heading ranges
- `DEFAULT_HEADING_METHOD = 0` - Method for heading correction background calculation
  - 0 = Mean
  - 1 = Median  
  - 2 = Gaussian fit
- `DEFAULT_BASELINE_M = 0.5` - Distance between dual sensors (meters)
- `DEFAULT_MAX_GAP_DIST = 10` - Max gap distance for segmentation (meters)
- `DEFAULT_MIN_SEG_LEN = 150` - Min segment length to keep (meters)

**Gridding Parameters:**
- `DEFAULT_GRID_SIZE = 50` - Grid cell size in meters (smaller = finer resolution)
- `DEFAULT_CUTOFF_WAVELEN = 30` - Filter cutoff wavelength (meters)
- `DEFAULT_PROX_THRESH = 40` - Proximity threshold for masking (meters)

**Plotting Parameters:**
- `DEFAULT_MAP_XRES = 500` - Easting tick increment
- `DEFAULT_MAP_YRES = 500` - Northing tick increment

---

## Utilities

### `check_optional_dependencies()` [static]

**Line:** 132-139

Check which optional dependencies are installed.

**Parameters:** None  
**Returns:** None (prints to console)  
**Internal Calls:** None

**Checks:**
- pyIGRF14 (IGRF calculations)
- harmonica (magnetic field processing)
- rasterio (GeoTIFF export)
- mpl_toolkits (advanced plotting)

---

### `is_numeric(s: str) -> bool` [static]

**Line:** 144-151

Check if string can be converted to a number.

**Parameters:**
- `s` (str): String to test

**Returns:** bool  
**Internal Calls:** None

---

### `to_decimal_year(date)` [static]

**Line:** 153-163

Convert datetime object to decimal year format.

**Parameters:**
- `date` (datetime): Date to convert

**Returns:** float (e.g., 2025.25 for March 31, 2025)  
**Internal Calls:** None

**Example:**
```python
from datetime import datetime
decimal_year = WellDetective.to_decimal_year(datetime(2025, 3, 31))
# Returns: 2025.25
```

---

### `get_matching_columns(df: pd.DataFrame, column_list: list) -> list` [static]

**Line:** 165-168

Return DataFrame columns that exist in the provided list.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to search
- `column_list` (list): List of column names to match

**Returns:** list  
**Internal Calls:** None

---

### `calculate_heading(lat1, lon1, lat2, lon2)` [static]

**Line:** 280-303

Calculate compass heading between two geographical points using haversine formula.

**Parameters:**
- `lat1, lon1, lat2, lon2` (float): Latitude and longitude in degrees

**Returns:** float (heading in degrees, 0° is North)  
**Internal Calls:** None

---

### `get_utm_zone_from_lon_lat(longitude, latitude)` [static]

**Line:** 305-353

Determine UTM zone from longitude and latitude.

**Parameters:**
- `longitude` (float): Longitude in decimal degrees (-180 to 180)
- `latitude` (float): Latitude in decimal degrees (-90 to 90)

**Returns:** tuple (zone_number, hemisphere, epsg_code)  
**Internal Calls:** None

**Example:**
```python
zone, hemi, epsg = WellDetective.get_utm_zone_from_lon_lat(-111.5, 40.7)
# Returns: (12, 'N', 32612)
```

---

### `get_survey_utm_zone(lat_col='Lat', lon_col='Long')`

**Line:** 1441-1477

Automatically determine UTM zone for survey based on coordinate centroid.

**Parameters:**
- `lat_col` (str): Latitude column name (default: 'Lat')
- `lon_col` (str): Longitude column name (default: 'Long')

**Returns:** tuple (zone_number, hemisphere, epsg_code)  
**Internal Calls:** `get_utm_zone_from_lon_lat()`

**Example:**
```python
wd = WellDetective(data)
zone, hemi, epsg = wd.get_survey_utm_zone()
print(f"Survey is in UTM Zone {zone}{hemi} (EPSG:{epsg})")
```

---

### `_get_utm_wkt(zone, northern=True)` [static, private]

**Line:** 356-390

Generate WKT string for UTM zone (workaround for PROJ database issues).

**Parameters:**
- `zone` (int): UTM zone number (1-60)
- `northern` (bool): True for Northern hemisphere

**Returns:** str (WKT string)  
**Internal Calls:** None

---

### `make_utm_easting_formatter()` [static]

**Line:** 396-456

Create matplotlib formatter for UTM easting coordinates with superscripts.

**Parameters:** None  
**Returns:** function (formatter)  
**Internal Calls:** None

**Used by:** `plot_flight_tracks()`, `plot_Mag_Heat()`

---

### `make_utm_northing_formatter()` [static]

**Line:** 458-518

Create matplotlib formatter for UTM northing coordinates with superscripts.

**Parameters:** None  
**Returns:** function (formatter)  
**Internal Calls:** None

**Used by:** `plot_flight_tracks()`, `plot_Mag_Heat()`

---

## Data Loading

### `detect_header_generic(filepath: str, max_lines: int = 50)` [static]

**Line:** 173-276

Automatically detect header line, delimiter, and column count in data file.

**Parameters:**
- `filepath` (str): Path to file
- `max_lines` (int): Maximum lines to check (default: 50)

**Returns:** tuple (header_line_0indexed, delimiter, num_columns)  
**Internal Calls:** `is_numeric()`

**Handles:**
- Tab, semicolon, pipe, comma, space delimiters
- Different delimiters for header vs data
- Whitespace-delimited files

---

### `Load_Process_SingleMagDataFile(s_folderpath, Mag_File, v_plotdata=False, v_runchecks=False)` [static]

**Line:** 528-700

Load and fully process a single magnetometry file.

**Parameters:**
- `s_folderpath` (str): Folder path
- `Mag_File` (str): Filename
- `v_plotdata` (bool): Generate plots (default: False)
- `v_runchecks` (bool): Verbose output (default: False)

**Returns:** WellDetective object

**Internal Calls:**
- `detect_header_generic()`
- `get_matching_columns()`
- `Check_4_DateCol()`
- `Check_4_LatLon()`
- `Check_4_MagTotal()`
- `add_heading_column()`
- `find_primary_secondary_headings()`
- `start_filtering()`
- `remove_turning_data()`
- `segment_and_filter_data()`
- `auto_equalize_heading_correction()`
- `get_survey_utm_zone()`
- `project_coordinates()`
- `plot_flight_tracks()` (if v_plotdata=True)

**Processing Steps:**
1. Detect file format and load data
2. Validate/add date columns
3. Interpolate lat/lon coordinates
4. Calculate magnetic total field
5. Add heading column
6. Detect primary/secondary headings
7. Filter turning data
8. Remove short segments
9. Equalize heading correction
10. Project to UTM coordinates

**Example:**
```python
wd = WellDetective.Load_Process_SingleMagDataFile('data/', 'survey.csv')
```

---

### `Load_Process_MultipleMagDataFiles(s_folderpath, Mag_File_List, v_plotdata=False, v_runchecks=False, skip_errors=False)` [static]

**Line:** 702-860

Load and process multiple magnetometry files, combining them into a single dataset.

**Parameters:**
- `s_folderpath` (str): Folder path
- `Mag_File_List` (list): List of filenames
- `v_plotdata` (bool): Generate plots (default: False)
- `v_runchecks` (bool): Verbose output (default: False)
- `skip_errors` (bool): Skip failed files vs raise error (default: False)

**Returns:** WellDetective object with concatenated data

**Internal Calls:**
- `Load_Process_SingleMagDataFile()` (for each file)
- `create_spatial_map()` (if v_plotdata=True)
- `plot_flight_tracks()` (if v_plotdata=True)
- `plot_Mag_Heat()` (if v_plotdata=True)

**Features:**
- Validates data consistency across files
- Tracks processing status per file
- Reports failed files
- Checks column alignment

**Example:**
```python
files = ['survey1.csv', 'survey2.csv', 'survey3.csv']
wd = WellDetective.Load_Process_MultipleMagDataFiles('data/', files)
```

---

### `Load_from_NetCDF(filepath)` [static]

**Line:** ~906  
**Type:** Static method - File I/O

Load a previously saved WellDetective object from NetCDF file.

**Parameters:**
- `filepath` (str): Path to .nc file created by `export_to_netcdf()`

**Returns:**
- `WellDetective`: Reconstructed object with all data levels and map

**Restores:**
- `data_raw` - Original unprocessed data (if saved)
- `data` - Working data with calculated fields (if saved)
- `data_filtered` - Filtered data (if saved)
- `map` - Spatial grid with metadata (if saved)

**Example:**
```python
# Load previously processed survey
wd = WellDetective.Load_from_NetCDF('survey_data.nc')

# Resume analysis
print(f"Loaded {len(wd.data_filtered):,} points")
wd.plot_Mag_Heat()
```

**Notes:**
- Much faster than reprocessing from CSV files
- NetCDF file must have been created by `export_to_netcdf()`
- Handles missing data groups gracefully
- Useful for sharing processed datasets

---

## Data Processing

### `Check_4_DateCol(v_date, v_deltaT=0.001)`

**Line:** 867-910

Validate/add date information to dataset.

**Parameters:**
- `v_date` (datetime.date): Manual date if missing
- `v_deltaT` (float): Time step (default: 0.001)

**Returns:** self  
**Internal Calls:** `get_matching_columns()`

**Modifies:** `self.data` (adds 'Date' and 'Time' columns if missing)

---

### `Check_4_LatLon(latcolname='Lat', loncolname='Lon')`

**Line:** 912-937

Interpolate lat/lon for sparse recordings (replaces 0,0 values).

**Parameters:**
- `latcolname` (str): Latitude column name
- `loncolname` (str): Longitude column name

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data` (interpolates lat/lon values)

---

### `Check_4_MagTotal()`

**Line:** 939-987

Validate/calculate total magnetic field from vector components.

**Parameters:** None  
**Returns:** self

**Internal Calls:**
- `get_matching_columns()`
- `process_magnetometry_data()` (if single sensor)
- `process_dual_magnetometry_data()` (if dual sensor)

**Modifies:** `self.data` (adds total field columns)

---

### `add_heading_column(lat_col='Lat', lon_col='Long', window=20)`

**Line:** 1025-1076

Add compass heading column using vectorized calculation.

**Parameters:**
- `lat_col` (str): Latitude column
- `lon_col` (str): Longitude column
- `window` (int): Look-back window (default: 20)

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data` (adds 'Heading' column)

**Note:** Replaces deprecated `add_heading_column_old()`

---

### `process_dual_magnetometry_data(...)`

**Line:** 1078-1182

Process dual magnetometer data to calculate field totals, gradients, and differences.

**Parameters:**
- `b1x_col, b1y_col, b1z_col` (str): Sensor 1 vector components
- `b2x_col, b2y_col, b2z_col` (str): Sensor 2 vector components
- `baseline_m` (float): Sensor separation (default: 0.5)
- `add_gradient` (bool): Calculate gradient (default: True)
- `add_vector_diff` (bool): Calculate vector differences (default: False)

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data` (adds Total_B1, Total_B2, Total_Avg, Total_Diff, Gradient_nT_m)

---

### `process_magnetometry_data(bx_col, by_col, bz_col)`

**Line:** 1184-1227

Calculate total field from vector components (single sensor).

**Parameters:**
- `bx_col` (str): X component column
- `by_col` (str): Y component column
- `bz_col` (str): Z component column

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data` (adds 'Total_Field' column)

---

### `project_coordinates(utm_zone=12, lat_col='Lat', lon_col='Long')`

**Line:** 1413-1435

Project geographic coordinates to UTM.

**Parameters:**
- `utm_zone` (int): UTM zone number (default: 12)
- `lat_col` (str): Latitude column
- `lon_col` (str): Longitude column

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data_filtered` (adds 'easting' and 'northing' columns)

**Example:**
```python
wd.project_coordinates(utm_zone=12)
```

---

## Filtering

### `start_filtering()`

**Line:** 1233-1237

Initialize filtered data layer by copying working data.

**Parameters:** None  
**Returns:** self

**Creates:** `self.data_filtered` (copy of `self.data`)

---

### `find_primary_secondary_headings(tolerance=10, filename=None)`

**Line:** 1239-1268

Identify primary and secondary flight heading ranges using K-Means clustering.

**Parameters:**
- `tolerance` (float): Degrees around cluster centers (default: 10)
- `filename` (str): Filter data by filename substring (default: None for all data)

**Returns:** tuple (primary_range, secondary_range)  
**Internal Calls:** None

**Example:**
```python
# All data
primary, secondary = wd.find_primary_secondary_headings()
# Returns: ((85, 105), (265, 285))  # E-W flight lines

# Specific flight only
primary, secondary = wd.find_primary_secondary_headings(filename='flight_01')
```

---

### `remove_turning_data(primary_range, secondary_range)`

**Line:** 1270-1309

Remove turning maneuvers, keeping only straight flight lines.

**Parameters:**
- `primary_range` (tuple): Primary heading range (min, max)
- `secondary_range` (tuple): Secondary heading range (min, max)

**Returns:** self  
**Internal Calls:** None

**Modifies:** `self.data_filtered` (removes turning data)

---

### `segment_and_filter_data(max_gap_distance=10, min_segment_length=150, lat_col='Lat', lon_col='Long')`

**Line:** 1311-1385

Segment data into continuous flight lines and remove short segments.

**Parameters:**
- `max_gap_distance` (float): Max gap in meters (default: 10)
- `min_segment_length` (float): Min segment length in meters (default: 150)
- `lat_col` (str): Latitude column
- `lon_col` (str): Longitude column

**Returns:** self  
**Internal Calls:** None (nested helper function)

**Modifies:** `self.data_filtered` (removes short segments)

---

### `auto_normalize_heading_correction(primary_range, secondary_range, mag_col='Mag', method=0)`

**Line:** 1497-1614

Apply heading error correction by normalizing (subtracting background from) magnetic field measurements for each heading direction.

**Parameters:**
- `primary_range` (tuple): Primary heading range
- `secondary_range` (tuple): Secondary heading range
- `mag_col` (str): Magnetic field column (default: 'Mag')
- `method` (int): Background calculation method (default: 0 or `WellDetective.DEFAULT_HEADING_METHOD`)
  - 0 = Mean
  - 1 = Median
  - 2 = Gaussian fit (fits Gaussian to histogram and uses peak center)

**Returns:** self  
**Internal Calls:** 
- `scipy.optimize.curve_fit` (if method=2)

**Modifies:** `self.data_filtered` (adds 'Corrected' column with background-subtracted values)

**Notes:**
- Preserves original `mag_col` data unchanged
- Creates new 'Corrected' column by subtracting respective backgrounds from each heading's data
- Primary heading data: `Corrected = mag_col - primary_bkgd`
- Secondary heading data: `Corrected = mag_col - secondary_bkgd`
- Method 2 (Gaussian fit) is more robust to outliers and skewed distributions
- Gaussian fit creates histogram, fits Gaussian curve, and uses the center (mean parameter) as background
- If Gaussian fit fails, automatically falls back to mean with warning
- Prints diagnostic output including background values and correction applied

**Example:**
```python
# Mean (default - uses WellDetective.DEFAULT_HEADING_METHOD)
wd.auto_normalize_heading_correction(primary_range, secondary_range)

# Median
wd.auto_normalize_heading_correction(primary_range, secondary_range, method=1)

# Gaussian fit (recommended for noisy data)
wd.auto_normalize_heading_correction(primary_range, secondary_range, method=2)

# Using class constant explicitly
wd.auto_normalize_heading_correction(
    primary_range, 
    secondary_range, 
    method=WellDetective.DEFAULT_HEADING_METHOD
)
```

---

## Mapping & Gridding

### `grid_and_filter_data(inclination, declination, grid_size=50, cutoff_wavelength=30, proximity_threshold=40)`

**Line:** 1725-1797

Interpolate magnetic data to regular grid, apply reduction-to-pole, and low-pass filter.

**Parameters:**
- `inclination` (float): IGRF inclination in degrees
- `declination` (float): IGRF declination in degrees
- `grid_size` (float): Grid cell size in meters (default: 50). Smaller = finer resolution
- `cutoff_wavelength` (float): Filter wavelength in meters (default: 30)
- `proximity_threshold` (float): Max distance from flight line in meters (default: 40)

**Returns:** xarray.DataArray  
**Internal Calls:** None

**Requires:** harmonica package

**Processing:**
1. Subtract mean from magnetic data
2. Calculate number of grid points from cell size and survey extent
3. Interpolate to regular grid using linear interpolation
4. Pad edges for filtering
5. Apply reduction-to-pole transformation
6. Apply Gaussian low-pass filter
7. Mask points far from flight lines

**Note:** Grid cell size determines resolution. For a 4km × 2km survey:
- `grid_size=100` → ~40×20 grid (800 cells)
- `grid_size=50` → ~80×40 grid (3,200 cells)  
- `grid_size=25` → ~160×80 grid (12,800 cells)

---

### `create_spatial_map(grid_size=50, cutoff_wavelength=30, proximity_threshold=40, use_pyigrf=True, inclination=None, declination=None, date=None, v_plotdata=False)`

**Line:** 1804-1960

Create complete spatially interpolated magnetic map with auto-detection of date and coordinates.

**Parameters:**
- `grid_size` (float): Grid cell size in meters (default: 50). Smaller = finer resolution
- `cutoff_wavelength` (float): Filter wavelength (default: 30)
- `proximity_threshold` (float): Max distance from lines (default: 40)
- `use_pyigrf` (bool): Auto-calculate inclination/declination (default: True)
- `inclination` (float): Manual inclination (if use_pyigrf=False)
- `declination` (float): Manual declination (if use_pyigrf=False)
- `date` (datetime): Date for IGRF (default: auto-detect from data)
- `v_plotdata` (bool): Generate plot (default: False)

**Returns:** self  
**Internal Calls:**
- `get_matching_columns()` (auto-detect lat/lon/date columns)
- `to_decimal_year()` (if use_pyigrf=True)
- `grid_and_filter_data()`
- `plot_Mag_Heat()` (if v_plotdata=True)

**Creates:** `self.map` (dictionary with gridded data)

**Features:**
- Auto-detects latitude/longitude/date columns from various naming conventions
- Uses IGRF model for magnetic field parameters (if pyIGRF installed)
- Stores grid with metadata including cell size, inclination, declination

**Example:**
```python
# Auto-detect everything (recommended)
wd.create_spatial_map(grid_size=50, cutoff_wavelength=30)

# Fine resolution
wd.create_spatial_map(grid_size=25)

# Manual IGRF values
wd.create_spatial_map(use_pyigrf=False, inclination=64.5, declination=2.5)
```

---

## Detection

### `detect_hotspots(mag_grid, grid_x, grid_y, well_coords, distance_threshold=100, bandwidth=80)`

**Line:** 1702-1767

Detect magnetic hotspots and identify potential orphan wells using MeanShift clustering.

**Parameters:**
- `mag_grid` (np.array): 2D magnetic data grid
- `grid_x, grid_y` (np.array): Coordinate grids
- `well_coords` (list): Known well coordinates [(x1, y1), (x2, y2), ...]
- `distance_threshold` (float): Min distance to known wells (default: 100)
- `bandwidth` (float): Clustering bandwidth (default: 80)

**Returns:** tuple (centroids, orphan_wells)  
**Internal Calls:** None

**Example:**
```python
known_wells = [(250000, 4500000), (251000, 4501000)]
centroids, orphans = wd.detect_hotspots(
    wd.map['mag_grid'], 
    wd.map['grid_x'], 
    wd.map['grid_y'], 
    known_wells
)
```

---

## Export

### `export_map_to_geotiff(output_path, crs=None)`

**Line:** 2037-2147

Export gridded magnetic data to GeoTIFF format with auto-detection of UTM zone.

**Parameters:**
- `output_path` (str): Output .tif file path
- `crs` (str): Coordinate reference system (default: None for auto-detect)
  - If None, automatically detects UTM zone from data coordinates
  - Examples: 'EPSG:32612' (Zone 12N), 'EPSG:32614' (Zone 14N)

**Returns:** None  
**Internal Calls:**
- `get_matching_columns()` (if auto-detecting CRS)
- `get_survey_utm_zone()` (if auto-detecting CRS)
- `_get_utm_wkt()` (if PROJ database issues)

**Requires:** rasterio package

**Features:**
- Auto-detects correct UTM zone from lat/lon data
- Handles array transposition for GeoTIFF convention (y, x) indexing
- Flips array vertically (GeoTIFF expects north at top)
- LZW compression
- Metadata tags (inclination, declination, creation date, units)
- Automatic WKT fallback for PROJ database issues
- Diagnostic output showing pixel sizes and bounds

**Example:**
```python
# Auto-detect UTM zone (recommended)
wd.export_map_to_geotiff('magnetic_map.tif')

# Specify CRS manually
wd.export_map_to_geotiff('magnetic_map.tif', crs='EPSG:32614')
```

---

### `export_to_netcdf(output_path, include_raw=False, include_data=False, include_filtered=True, include_map=True)`

**Line:** 1861-1983

Export complete WellDetective object to NetCDF format.

**Parameters:**
- `output_path` (str): Output .nc file path
- `include_raw` (bool): Include data_raw (default: False)
- `include_data` (bool): Include data (default: False)
- `include_filtered` (bool): Include data_filtered (default: True)
- `include_map` (bool): Include map (default: True)

**Returns:** None  
**Internal Calls:** None

**Requires:** xarray package

**Structure:**
- Group: `/raw` - Original data
- Group: `/processed` - Working data
- Group: `/filtered` - Cleaned data
- Group: `/map` - Gridded data
- Group: `/metadata` - Processing log and parameters

**Example:**
```python
# Default: exports only filtered data and map (~0.6 GB for 2M points)
wd.export_to_netcdf('survey_map.nc')

# Include all data levels (~3 GB for 2M points)
wd.export_to_netcdf('survey_complete.nc', include_raw=True, include_data=True)

# Map only (~0.1 MB)
wd.export_to_netcdf('map_only.nc', include_filtered=False, include_map=True)
```

---

## Plotting

### `plot_flight_tracks(E_incr=500, N_incr=500, figsize=(12, 10), save_path=None)`

**Line:** 1988-2089

Create scatter plot of flight tracks color-coded by source file.

**Parameters:**
- `E_incr` (int): Easting tick increment (default: 500)
- `N_incr` (int): Northing tick increment (default: 500)
- `figsize` (tuple): Figure size (default: (12, 10))
- `save_path` (str): Save path (default: None for interactive display)

**Returns:** tuple (fig, ax)  
**Internal Calls:**
- `make_utm_easting_formatter()`
- `make_utm_northing_formatter()`

**Example:**
```python
fig, ax = wd.plot_flight_tracks(E_incr=1000, N_incr=1000)
```

---

### `plot_Mag_Heat(E_incr=500, N_incr=500, figsize=(12, 10), save_path=None)`

**Line:** 2091-2163

Create heatmap/contour plot of magnetic field data with flight tracks overlay.

**Parameters:**
- `E_incr` (int): Easting tick increment (default: 500)
- `N_incr` (int): Northing tick increment (default: 500)
- `figsize` (tuple): Figure size (default: (12, 10))
- `save_path` (str): Save path (default: None for interactive display)

**Returns:** tuple (fig, ax)  
**Internal Calls:**
- `make_utm_easting_formatter()`
- `make_utm_northing_formatter()`

**Requires:** `self.map` must be created first

**Example:**
```python
wd.create_spatial_map()
fig, ax = wd.plot_Mag_Heat(save_path='heatmap.png')
```

---

### `plot_Heading_Corr(mag_col=None, filename=None, primary_range=None, secondary_range=None, bins=50, figsize=(16, 8), save_path=None)`

**Line:** 2467-2580

Create 2x1 subplot figure showing heading analysis and magnetic field distribution for a specific flight.

**Parameters:**
- `mag_col` (str): Magnetic field column name (default: None for auto-detect)
- `filename` (str): Filename or substring to filter by specific flight (default: None)
- `primary_range` (tuple): Primary heading range (min, max) in degrees (default: None for auto-detect)
- `secondary_range` (tuple): Secondary heading range (min, max) in degrees (default: None for auto-detect)
- `bins` (int): Number of histogram bins (default: 50)
- `figsize` (tuple): Figure size (default: (16, 8))
- `save_path` (str): Save path (default: None for interactive display)

**Returns:** tuple (fig, (ax0, ax1))  
**Internal Calls:**
- `get_matching_columns()` (if mag_col=None)
- `find_primary_secondary_headings()` (if ranges not provided)

**Subplots:**
- **ax[0]:** Heading vs index with primary/secondary range highlighted
- **ax[1]:** Overlapping histograms of magnetic field for primary (green) and secondary (red) headings, with mean lines and heading correction value

**Features:**
- Auto-detects magnetic field column from MAG_COLUMNS list
- Auto-detects heading ranges if not provided
- Requires filename to analyze specific flight
- Shows statistics (mean, std) for each heading direction
- Displays calculated heading correction value

**Example:**
```python
# Auto-detect mag column and heading ranges
fig, axes = wd.plot_Heading_Corr(filename='flight_01.csv')

# Specify mag column
fig, axes = wd.plot_Heading_Corr(filename='flight_01.csv', mag_col='Corrected')

# With manual heading ranges
fig, axes = wd.plot_Heading_Corr(
    filename='flight_01.csv',
    primary_range=(85, 105),
    secondary_range=(265, 285)
)
```

---

### `plot_corrected_mag_by_heading(figsize=None, save_path=None, plot_every_nth=10)`

**Line:** 2946-3067

Create tall multi-panel figure showing corrected magnetic field data color-coded by heading direction.

**Parameters:**
- `figsize` (tuple): Figure size (default: None for auto-sizing based on number of files)
- `save_path` (str): Save path (default: None for interactive display)
- `plot_every_nth` (int): Plot every Nth point to reduce density (default: 10)

**Returns:** tuple (fig, axes)  
**Internal Calls:**
- `matplotlib.collections.LineCollection` for efficient color-coded lines

**Requires:** 
- `self.data_filtered` must exist and contain 'Corrected' and 'Heading' columns
- Run `auto_normalize_heading_correction()` and `add_heading_column()` first

**Features:**
- Creates one subplot per file in dataset
- Color-codes data by heading using HSV colormap (0-360°)
- Efficient subsampling for large datasets
- Displays full dataset statistics (n, mean, std)
- Auto-detects single vs. multi-file datasets
- Uses LineCollection for smooth color transitions

**Example:**
```python
# Default (every 10th point)
fig, axes = wd.plot_corrected_mag_by_heading()

# More dense plotting
fig, axes = wd.plot_corrected_mag_by_heading(plot_every_nth=5)

# Save to file
fig, axes = wd.plot_corrected_mag_by_heading(save_path='mag_by_heading.png')
```

---

## Summary Statistics

### Method Counts

| Category | Count |
|----------|-------|
| **Total Methods** | 36 |
| Constructor | 1 |
| Static Methods | 13 |
| Instance Methods | 23 |
| Private Methods | 1 |
| Deprecated | 1 |

### By Functional Category

| Category | Methods |
|----------|---------|
| Initialization | 1 |
| Utilities | 8 |
| Data Loading | 3 |
| Data Processing | 9 |
| Filtering | 6 |
| Mapping & Gridding | 2 |
| Detection | 1 |
| Export | 2 |
| Plotting | 6 |

### Method Chaining

Most instance methods return `self` to enable method chaining:

```python
wd = (WellDetective(data)
      .Check_4_DateCol(date)
      .Check_4_LatLon()
      .add_heading_column()
      .start_filtering()
      .remove_turning_data(primary, secondary)
      .project_coordinates()
      .create_spatial_map())
```

### Dependencies

**Required:**
- pandas, numpy, xarray
- pyproj, scipy, sklearn
- matplotlib

**Optional:**
- harmonica (reduction to pole, filtering)
- pyIGRF14 (IGRF calculations)
- rasterio (GeoTIFF export)
- mpl_toolkits (advanced plotting)

---

## Complete Workflow Example

```python
# 1. Load and process data
wd = WellDetective.Load_Process_SingleMagDataFile('data/', 'survey.csv')

# 2. Automatically detect UTM zone
zone, hemi, epsg = wd.get_survey_utm_zone()
print(f"Survey is in UTM Zone {zone}{hemi}")

# 3. Create spatial map
wd.create_spatial_map(
    grid_res=100,
    cutoff_wavelength=30,
    proximity_threshold=40,
    use_pyigrf=True
)

# 4. Plot results
fig1, ax1 = wd.plot_flight_tracks()
fig2, ax2 = wd.plot_Mag_Heat()

# 5. Export results
wd.export_map_to_geotiff(f'map_zone{zone}{hemi}.tif', crs=f'EPSG:{epsg}')
wd.export_to_netcdf('complete_survey.nc')

# 6. Detect hotspots
known_wells = [(250000, 4500000), (251000, 4501000)]
centroids, orphans = wd.detect_hotspots(
    wd.map['mag_grid'],
    wd.map['grid_x'],
    wd.map['grid_y'],
    known_wells
)
print(f"Found {len(orphans)} potential orphan wells")
```

---

**Document Generated:** 2026-06-09  
**WellDetective Version:** 2.0  
**Authors:** Eric Guiltinan, Nash Taylor, James E. Lee

**Major Updates in v2.0:**
- Changed `grid_res` to `grid_size` (now in meters, not number of points)
- Auto-detection of UTM zones in GeoTIFF export
- Auto-detection of magnetic field columns in plotting
- Auto-detection of lat/lon/date columns throughout
- Forward-filling of sparse date values
- Fixed GeoTIFF orientation issues (transpose + flip)
- Improved heading correction with Gaussian fitting option
- Updated default grid size to 50m for better resolution
