# WellDetective Package Walkthrough

**Version:** 2.0  
**Last Updated:** June 2026

A practical guide for processing magnetometry data to detect buried well casings. 

This guide incudes inputs and results shared in /WellDetective/examples for loading data using a Skyfront Perimeter 8+ UAV equipped with a Geometrics MagArrow sensor for magnetometry measurements. There are 6 files, each corresponding to a single flight, that comprise the larger survey area. Figures produced by WellDetective processing are included in the examples directory.

---

## Quick Start

### 1. Import and Setup

```python
from WellDetective import WellDetective
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check optional dependencies
WellDetective.check_optional_dependencies()
```

### 2. Configure Default Parameters (Optional)

The package uses sensible defaults, but you can customize processing parameters:

```python
# View current defaults
for attr in dir(WellDetective):
    if attr.startswith('DEFAULT_'):
        print(f"{attr}: {getattr(WellDetective, attr)}")

# Modify if needed
WellDetective.DEFAULT_HEADING_TOLERANCE = 15  # degrees
WellDetective.DEFAULT_GRID_SIZE = 5         # meters
WellDetective.DEFAULT_CUTOFF_WAVELEN = 30    # meters
```

**Key Parameters:**
- `DEFAULT_HEADING_WINDOW = 20` - Window size for heading calculation
- `DEFAULT_HEADING_TOLERANCE = 20` - Tolerance for flight heading ranges (degrees)
- `DEFAULT_HEADING_METHOD = 0` - Background correction method (0=mean, 1=median, 2=Gaussian)
- `DEFAULT_MAX_GAP_DIST = 10` - Max gap between points for segmentation (meters)
- `DEFAULT_MIN_SEG_LEN = 150` - Minimum segment length to keep (meters)
- `DEFAULT_GRID_SIZE = 50` - Grid cell size in meters (smaller = finer resolution)
- `DEFAULT_CUTOFF_WAVELEN = 30` - Low-pass filter wavelength (meters)
- `DEFAULT_PROX_THRESH = 40` - Maximum distance from flight lines for masking (meters)

---

## Basic Workflow

### 3. Load and Process Data

**Single File:**
```python
# Process a single magnetometry file
wd = WellDetective.Load_Process_SingleMagDataFile(
    s_folderpath='./data/',
    Mag_File='survey_flight.csv',
    v_plotdata=False,
    v_runchecks=False
)
```

**Multiple Files (Recommended):**
```python
# Define data directory and file list
s_folderpath = './LBL/'
Mag_file_list = [
    'SRVY0-ACQU5-100Hz.csv',
    'SRVY0-ACQU6-100Hz.csv',
    'SRVY0-ACQU7-100Hz.csv',
    'SRVY0-ACQU8-100Hz.csv',
    'SRVY0-ACQU9-100Hz.csv',
    'SRVY0-ACQU10-100Hz.csv'
]

# Load and process all files
# This automatically:
# - Detects file format and headers
# - Calculates heading directions
# - Removes turning data
# - Filters short segments
# - Applies heading corrections
# - Creates UTM projections
# - Generates gridded magnetic map
wd = WellDetective.Load_Process_MultipleMagDataFiles(
    s_folderpath=s_folderpath,
    Mag_File_List=Mag_file_list,
    v_plotdata=True,      # Show plots during processing
    v_runchecks=False,    # Disable verbose debugging output
    skip_errors=True      # Continue if a file fails
)
```

**What happens during loading:**
1. Auto-detects file format (delimiter, header line)
2. Validates/adds date information (forward-fills sparse date values)
3. Interpolates sparse GPS coordinates
4. Calculates total magnetic field (if needed)
5. Computes flight headings
6. Detects primary/secondary heading ranges (K-means clustering)
7. Removes turning maneuvers
8. Filters short flight segments
9. Applies heading corrections (subtracts background from each heading direction)
10. Projects to UTM coordinates (auto-detects zone)
11. Creates gridded magnetic anomaly map (reduction to pole + filtering)

**Processing Summary:**
```
SUMMARY:
  Files processed: 6/6
  Files failed: 0

  Combined data:
    data_raw:      2,377,107 rows  (original data)
    data:          2,377,107 rows  (with headings/corrections)
    data_filtered: 1,508,201 rows  (turns removed, segments filtered)
```

---

## Data Structure

The WellDetective object contains multiple data levels:

```python
# Check what exists
print(wd.__dict__.keys())
# Output: dict_keys(['data_raw', 'data', 'data_filtered', 'processing_log', 'map'])

# Quick diagnostic
print(f"raw: {len(wd.data_raw)}, data: {len(wd.data)}, filtered: {len(wd.data_filtered)}, map: {wd.map is not None}")
# Output: raw: 2377107, data: 2377107, filtered: 1508201, map: True
```

**Data Levels:**
- `data_raw` - Original unprocessed data (preserved)
- `data` - Working data with calculated fields (heading, corrections)
- `data_filtered` - Cleaned data (turns removed, segments filtered, UTM projected)
- `map` - Gridded magnetic anomaly map (100×100 grid, ~0.08 MB)
- `processing_log` - List of processing steps and parameters

**Map Structure:**
```python
print(wd.map.keys())
# Output: dict_keys(['data_array', 'grid_x', 'grid_y', 'mag_grid', 
#                    'inclination', 'declination', 'grid_size', 
#                    'cutoff_wavelength', 'proximity_threshold', 'created_at'])
```

---

## Visualization

### Flight Tracks
```python
# Plot GPS tracks color-coded by file
fig, ax = wd.plot_flight_tracks(E_incr=500, N_incr=500)
```

### Magnetic Heat Map
```python
# Plot gridded magnetic field with flight tracks
fig, ax = wd.plot_Mag_Heat(E_incr=500, N_incr=500, save_path='heatmap.png')
```

### Heading Correction Analysis
```python
# Analyze heading corrections for a specific flight
# Shows: (1) heading vs time, (2) magnetic field histograms by heading
fig, axes = wd.plot_Heading_Corr(
    filename='SRVY0-ACQU5-100Hz.csv',
    mag_col='Mag',
    bins=50,
    save_path='heading_analysis.png'
)
```

---

## Export Results

### GeoTIFF (for QGIS/GIS software)
```python
# Export gridded map as GeoTIFF with auto-detected UTM zone
wd.export_map_to_geotiff('magnetic_map.tif')

# Or specify CRS manually
wd.export_map_to_geotiff('magnetic_map.tif', crs='EPSG:32614')  # UTM Zone 14N
```

### NetCDF (for data archiving)
```python
# Export all data levels (WARNING: Large file ~2.9 GB for 2M points)
wd.export_to_netcdf(
    'survey_complete.nc',
    include_raw=True,
    include_data=True,
    include_filtered=True,
    include_map=True
)

# Recommended: Export only filtered data + map (~0.6 GB)
wd.export_to_netcdf(
    'survey_filtered.nc',
    include_raw=False,       # Skip raw data (default)
    include_data=False,      # Skip intermediate data (default)
    include_filtered=True,   # Include cleaned data
    include_map=True         # Include gridded map
)

# Minimal: Export only the map (~0.1 MB)
wd.export_to_netcdf(
    'map_only.nc',
    include_raw=False,
    include_data=False,
    include_filtered=False,
    include_map=True
)
```

---

## Advanced: Custom Processing

### Modify Heading Detection
```python
# Adjust heading tolerance for specific flight
primary_range, secondary_range = wd.find_primary_secondary_headings(
    tolerance=15,
    filename='flight_01.csv'  # Process specific flight only
)
print(f"Primary: {primary_range[0]:.1f}° - {primary_range[1]:.1f}°")
print(f"Secondary: {secondary_range[0]:.1f}° - {secondary_range[1]:.1f}°")
```

### Change Heading Correction Method
```python
# Apply different background correction methods
# Method 0: Mean (default, fast)
wd.auto_normalize_heading_correction(primary_range, secondary_range, method=0)

# Method 1: Median (robust to outliers)
wd.auto_normalize_heading_correction(primary_range, secondary_range, method=1)

# Method 2: Gaussian fit (best for noisy data)
wd.auto_normalize_heading_correction(primary_range, secondary_range, method=2)
```

### Adjust Segmentation
```python
# Remove short segments with custom thresholds
wd.segment_and_filter_data(
    max_gap_distance=20,      # Allow larger gaps
    min_segment_length=200,   # Keep only longer segments
    lat_col='Latitude',
    lon_col='Longitude'
)
```

### Regenerate Map at Different Resolution
```python
# Create finer/coarser grid (grid_size is cell size in meters)
wd.create_spatial_map(
    grid_size=25,             # 25m × 25m cells (finer detail)
    cutoff_wavelength=20,     # Sharper features (was 30)
    proximity_threshold=30,   # Stricter masking (was 40)
    use_pyigrf=True,          # Use IGRF for inclination/declination
    v_plotdata=True           # Show plot when done
)
```

**Grid size guide:**
- Smaller `grid_size` = finer resolution, more detail, slower processing
- Typical values: 10m (very fine), 25m (fine), 50m (default), 100m (coarse)

---

## Troubleshooting

### Check Processing Status
```python
# View processing log
for entry in wd.processing_log:
    print(entry)

# Check data quality
print(f"Points removed by filtering: {len(wd.data) - len(wd.data_filtered)}")
print(f"Percentage kept: {len(wd.data_filtered)/len(wd.data)*100:.1f}%")
```

### Common Issues

**Issue: Date column not found**
- The package will prompt you to enter a date manually
- Or it will use default date (1901-01-01) if not interactive

**Issue: Too much data removed**
- Check heading tolerance: `WellDetective.DEFAULT_HEADING_TOLERANCE`
- Adjust segment filtering: `DEFAULT_MIN_SEG_LEN`, `DEFAULT_MAX_GAP_DIST`

**Issue: Map looks noisy**
- Increase `cutoff_wavelength` for smoother results
- Try different heading correction method (Gaussian fit = method 2)

**Issue: GeoTIFF in wrong location in QGIS**
- Don't specify `crs` parameter - let it auto-detect the UTM zone
- Or verify your data's UTM zone: `zone, hemi, epsg = wd.get_survey_utm_zone()`
- Then use correct EPSG: `wd.export_map_to_geotiff('map.tif', crs=f'EPSG:{epsg}')`

**Issue: GeoTIFF appears stretched or distorted in QGIS**
- This is usually resolved by using auto-detect CRS
- Check the diagnostic output when exporting (pixel sizes should be approximately equal)

---

## File Size Reference

For 2.4 million data points:
- **Full export** (raw + data + filtered + map): ~2.9 GB
- **Filtered + map** (recommended): ~0.6-0.8 GB  
- **Map only**: ~0.1 MB
- **GeoTIFF**: ~0.1 MB

---

## Complete Example

```python
from WellDetective import WellDetective

# 1. Load data
wd = WellDetective.Load_Process_MultipleMagDataFiles(
    s_folderpath='./data/',
    Mag_File_List=['flight1.csv', 'flight2.csv'],
    v_plotdata=True
)

# 2. Check results
zone, hemi, epsg = wd.get_survey_utm_zone()
print(f"Survey: UTM Zone {zone}{hemi} (EPSG:{epsg})")
print(f"Grid shape: {wd.map['mag_grid'].shape}")
print(f"Grid cell size: {wd.map['grid_size']}m")
print(f"Valid cells: {np.sum(~np.isnan(wd.map['mag_grid']))}")

# 3. Visualize
wd.plot_flight_tracks()
wd.plot_Mag_Heat(save_path='results.png')

# 4. Analyze specific flight (optional)
wd.plot_Heading_Corr(filename='flight1.csv')  # Auto-detects mag column

# 5. Export (auto-detects CRS)
wd.export_map_to_geotiff('magnetic_map.tif')  # Auto-detects UTM zone
wd.export_to_netcdf('survey.nc')  # Defaults: filtered + map only

print("✓ Processing complete!")
```

---

## Tips

1. **Always check optional dependencies first**: Some features require harmonica, pyIGRF, rasterio
2. **Start with defaults**: The default parameters work well for most UAS magnetometry data (50m grid cells)
3. **Let it auto-detect**: Column names, date formats, UTM zones, and mag columns are all auto-detected
4. **Use heading correction plots**: `plot_Heading_Corr(filename='...')` helps tune heading tolerance per flight
5. **Export wisely**: Don't include raw/data levels in NetCDF unless you need them (default is filtered + map only)
6. **Process related flights together**: Files that survey the same area should be loaded together for proper gridding
7. **Grid size matters**: Smaller `grid_size` = finer detail but slower. Start with 50m, adjust as needed

---

## Citation

If you use WellDetective in your research, please cite:

```
WellDetective: Magnetometry Processing for Orphan Well Detection
Authors: Eric Guiltinan, Nash Taylor, James E. Lee
Version: 2.0
Year: 2025
```

---

**For more information, see:**
- `WELLDETECTIVE_FUNCTIONS.md` - Complete function reference
- `WellDetective.py` - Source code with docstrings
- Example notebook: `WD2_Example_Osage_LBL.ipynb`
