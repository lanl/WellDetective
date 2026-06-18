# WellDetective Changelog

All notable changes to the WellDetective package are documented in this file.

---

## Version 2.0.1 - 2026-06-09

**Type:** Patch Release (Bug Fixes + Performance)

### Bug Fixes

#### Fixed Tick Formatter Axes Context Issue
- **Issue:** Tick formatters could grab wrong axes in multi-axes plots (e.g., colorbar vs main plot)
- **Fix:** Added `target_ax` parameter to `make_utm_easting_formatter()` and `make_utm_northing_formatter()`
- **Impact:** Tick labels now display correctly in `plot_Mag_Heat()` and `plot_flight_tracks()`
- **Files:** `WellDetective.py` lines ~401, ~468, ~2370, ~2453

### Performance Improvements

#### Optimized Grid Interpolation (3-5x Speedup)
- **Improvement:** Replaced `scipy.interpolate.griddata()` with pre-computed `LinearNDInterpolator`
- **Speedup:** 3-5x faster for large datasets (>1M points: 5+ min → 1-2 min)
- **New Parameter:** `use_fast_interpolation=True` in `grid_and_filter_data()`
- **New Constant:** `DEFAULT_USE_FAST_INTERPOLATION = True`
- **Output:** Added timing information and grid statistics
- **Backward Compatible:** Original method available via `use_fast_interpolation=False`
- **Files:** `WellDetective.py` lines ~109, ~1733

### Technical Details
- No breaking changes
- No new dependencies
- Fully backward compatible
- Grid output numerically identical between old and new methods

---

## Version 2.0.0 - 2026-06-09

**Location:** `../WP9/WellDetective 2/`

**Authors:** Eric Guiltinan, Nash Taylor, James E. Lee  
**Maintainer:** James E. Lee
**Affiliation:** Los Alamos National Laboratory
**Contact:** jamesedlee@lanl.gov

### Major Changes

#### File Import & Sensor Compatibility
- **Multiple Sensor Support**: Improved file parsing to handle outputs from different magnetometer sensors
- **Multiple File Import**: `Load_Process_MultipleMagDataFiles()` imports and concatenates multiple survey flights seamlessly
- **Flexible Column Detection**: Auto-detects column names across different sensor formats (Geometrics MagArrow, etc.)
- **Header Detection**: Improved automatic header line identification for various CSV formats
- **Date Format Handling**: Supports multiple date formats from different data acquisition systems
- **GPS Coordinate Formats**: Handles various latitude/longitude column naming conventions
- **Magnetometry Data**: Auto-detects Mag, MagX, MagY, MagZ columns from different sensor outputs
- **Delimiter Auto-Detection**: Automatically handles comma, tab, semicolon, pipe, and space-delimited files
- **Per-Flight Processing**: Heading detection and corrections applied per-file before concatenation for multi-flight surveys

#### Architecture Improvements
- **Removed Hard-Coded Parameters**: Replaced hard-coded tuning values with class-level DEFAULT_* constants
- **Configurable Defaults**: All processing parameters now accessible as `WellDetective.DEFAULT_*` attributes
- **User Customization**: Users can modify defaults before instantiation (e.g., `WellDetective.DEFAULT_GRID_SIZE = 25`)
- **Flexible Processing**: Parameters like heading window, tolerance, grid size, filter wavelength now easily adjustable
- **Better Maintainability**: Centralized parameter definitions make code easier to understand and modify
- **Default Parameters Include**:
  - `DEFAULT_HEADING_WINDOW = 20`
  - `DEFAULT_HEADING_TOLERANCE = 20`
  - `DEFAULT_HEADING_METHOD = 0`
  - `DEFAULT_GRID_SIZE = 50`
  - `DEFAULT_CUTOFF_WAVELEN = 30`
  - `DEFAULT_MAX_GAP_DIST = 10`
  - `DEFAULT_MIN_SEG_LEN = 150`
  - `DEFAULT_PROX_THRESH = 40`

#### Data Quality Improvements
- **Sparse Date Handling**: `Check_4_DateCol()` now forward-fills and backward-fills sparse date values
- **Date Format Detection**: Tries multiple date format patterns automatically
- **Better Error Messages**: More descriptive errors when columns not found, with expected column lists

#### Heading Correction Enhancements
- **Median Fitting**: Added `method=1` option for robust background estimation
- **Gaussian Fitting**: Added `method=2` option for robust background estimation
- **Method Selection**: `DEFAULT_HEADING_METHOD = 0` (mean, median, or Gaussian fit)
- **Background Subtraction**: Now subtracts respective backgrounds from each heading direction

#### Grid System Overhaul
- **Changed `grid_res` to `grid_size`**: Parameter now specifies grid cell size in meters (not number of points)
- **New default**: `DEFAULT_GRID_SIZE = 50` meters
- **Grid calculation**: Automatically calculates number of grid points from cell size and survey extent

#### Documentation
- **New Files**:
  - `WALKTHROUGH.md`: Comprehensive user guide with examples
  - `WELLDETECTIVE_FUNCTIONS.md`: Complete function reference
  - `CHANGELOG.md`: This file
- **Updated Examples**: All examples now show auto-detection features
- **Troubleshooting Guide**: Common issues and solutions

### Technical Details

#### Modified Functions
- `Check_4_DateCol()`: Forward-fill sparse dates, modern pandas syntax (.ffill(), .bfill())
- `Load_Process_SingleMagDataFile()`: Check datetime type before using .str accessor
- `find_primary_secondary_headings()`: Added `filename` parameter for per-flight analysis
- `auto_normalize_heading_correction()`: Renamed from `auto_equalize_heading_correction`, added Gaussian fit
- `grid_and_filter_data()`: Changed parameter to `grid_size`, calculate grid points from cell size
- `create_spatial_map()`: Auto-detect columns, updated parameter to `grid_size`
- `export_map_to_geotiff()`: Auto-detect UTM zone, fix array orientation, diagnostic output
- `plot_Heading_Corr()`: Auto-detect mag column, updated default figsize to (16, 8)

#### New Features
- Gaussian fitting for heading corrections (`scipy.optimize.curve_fit`)
- Automatic column matching with flexible naming conventions
- Diagnostic output for GeoTIFF exports (pixel sizes, bounds)
- Interactive date column selection with fallback options

#### Bug Fixes
- Fixed transpose/flip order in GeoTIFF export (was upside down or transposed)
- Fixed `.str` accessor error when date column already converted to datetime
- Fixed deprecated `fillna(method=...)` syntax (now uses `.ffill()` and `.bfill()`)
- Fixed hard-coded column names ('Latitude', 'Longitude', 'Mag', 'Date') throughout

#### Performance
- Grid calculation now based on survey extent and cell size (more efficient)
- Default 50m cells provide good balance between resolution and speed

### Backward Compatibility

**Breaking Changes:**
- `grid_res` parameter renamed to `grid_size` with different meaning
  - **Old:** `grid_res=100` meant 100×100 grid
  - **New:** `grid_size=100` means 100m × 100m cells
- `auto_equalize_heading_correction()` renamed to `auto_normalize_heading_correction()`
- Default NetCDF export excludes raw/data levels (was included in v1.0)

**Migration Guide:**
```python
# Version 1.0
wd.create_spatial_map(grid_res=100)  # Creates 100×100 grid
wd.export_map_to_geotiff('map.tif', crs='EPSG:32612')  # Must specify CRS
wd.plot_Heading_Corr(mag_col='Mag', filename='file.csv')  # Must specify mag_col

# Version 2.0
wd.create_spatial_map(grid_size=50)  # Creates grid with 50m cells
wd.export_map_to_geotiff('map.tif')  # Auto-detects UTM zone
wd.plot_Heading_Corr(filename='file.csv')  # Auto-detects mag_col
```

### Known Issues
- PROJ database issues may require WKT fallback (handled automatically)

---

## Version 1.0 - 2025-03-31

**Location:** `../WP9/WellDetective-main/`

**Authors:** Eric Guiltinan, Javier Santos

### Initial Release

#### Core Features
- Load and process single or multiple magnetometry files
- Auto-detect file format (delimiter, header line)
- Calculate flight headings from GPS tracks
- Detect primary/secondary heading ranges (K-means clustering)
- Remove turning data and filter short segments
- Apply heading corrections to equalize magnetic field
- Project coordinates to UTM
- Create gridded magnetic anomaly maps
- Reduction to pole using harmonica
- Gaussian low-pass filtering
- Export to GeoTIFF and NetCDF formats
- Plotting: flight tracks, magnetic heat maps

#### Supported Data Formats
- CSV files with various delimiters (comma, tab, semicolon, pipe, space)
- Automatic header detection
- Flexible column naming (Lat/Latitude, Lon/Longitude, etc.)

#### Processing Pipeline
1. File format detection
2. Data validation and interpolation
3. Heading calculation
4. Flight line identification
5. Turning data removal
6. Segment filtering
7. Heading correction
8. UTM projection
9. Gridding and filtering
10. Export

#### Dependencies
- **Required:** pandas, numpy, xarray, matplotlib, pyproj, scipy, sklearn
- **Optional:** harmonica (filtering), pyIGRF14 (IGRF), rasterio (GeoTIFF), mpl_toolkits (plotting)

#### Default Parameters
- `DEFAULT_HEADING_WINDOW = 20`
- `DEFAULT_HEADING_TOLERANCE = 20`
- `DEFAULT_BASELINE_M = 0.5`
- `DEFAULT_MAX_GAP_DIST = 10`
- `DEFAULT_MIN_SEG_LEN = 150`
- `DEFAULT_GRID_RES = 100` (number of grid points)
- `DEFAULT_CUTOFF_WAVELEN = 30`
- `DEFAULT_PROX_THRESH = 40`


## Contact

**Maintainer:** James E. Lee  
**Repository:** WellDetective 2  
**Institution:** CAFE/Orphan Wells Project

For bug reports, feature requests, or contributions, please contact the maintainer.

---

**Version History:**
- v2.0.1: 2026-06-09 (Patch - tick formatter fix, 3-5x faster grid interpolation)
- v2.0.0: 2026-06-09 (Major update - auto-detection features, architecture improvements, multi-file import)
- v1.0: 2025-03-31 (Initial release)
