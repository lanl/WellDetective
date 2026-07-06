# WellDetective Installation Notes

## Fresh Installation

### Option 1: Using environment.yml (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate WellDetectiveEnv
```

### Option 2: Using requirements.txt

```bash
# Create a new conda environment
conda create -n WellDetectiveEnv python=3.11

# Activate environment
conda activate WellDetectiveEnv

# Install requirements
pip install -r requirements.txt
```

## Verification

After installation, verify all packages are installed:

```python
import sys
sys.path.insert(0, 'src')

from general.WellDetective import WellDetective
import general.WellDetective as wd_module

print(f"✓ Rasterio available: {wd_module.HAS_RASTERIO}")
print(f"✓ Harmonica available: {wd_module.HAS_HARMONICA}")
print(f"✓ PyIGRF available: {wd_module.HAS_PYIGRF}")
print(f"✓ MPL Toolkits available: {wd_module.HAS_MPL_TOOLKITS}")
```

Expected output:
```
✓ Rasterio available: True
✓ Harmonica available: True
✓ PyIGRF available: True
✓ MPL Toolkits available: True
```

## Key Packages

### Core Dependencies
- **rasterio** - GeoTIFF export functionality
- **xarray** - Multi-dimensional arrays
- **pyproj** - Coordinate transformations
- **pandas/numpy** - Data manipulation
- **matplotlib** - Plotting

### Optional Dependencies
- **harmonica** - Magnetic field processing (reduction to pole, filtering)
- **pyIGRF14** - International Geomagnetic Reference Field calculations

## Troubleshooting

### If rasterio is missing:
```bash
conda activate WellDetectiveEnv
pip install rasterio==1.4.4
```

### PROJ Database Warnings

You may see warnings like:
```
ERROR 1: PROJ: internal_proj_create_from_database: proj.db contains DATABASE.LAYOUT.VERSION.MINOR = X
```

**This is normal and handled automatically!** WellDetective includes a built-in workaround that:
- Automatically detects PROJ database version mismatches
- Falls back to WKT (Well-Known Text) format for UTM coordinate systems
- Works seamlessly without requiring any code changes
- Displays a warning message when the workaround is used

**What causes this?**
- PyPI wheels for rasterio/pyproj may bundle older PROJ databases
- Conda environments may have conflicting PROJ library versions
- This is a known issue when mixing pip and conda installations

**Do I need to fix it?**
No! The workaround handles it automatically. If you want to eliminate the warnings:
1. Use conda-forge for installation: `conda install -c conda-forge rasterio pyproj`
2. Or ignore the warnings - your GeoTIFF exports will work correctly

### If harmonica fails to install:
Harmonica requires LLVM/numba which can have complex build dependencies. If it fails:
1. The module will still work, but reduction-to-pole operations will be unavailable
2. You can skip harmonica and manually provide filtered data instead

## Known Issues

### EPSG Code Support
- **Issue**: Some EPSG codes may fail due to PROJ database version mismatches
- **Workaround**: Built-in automatic fallback to WKT for UTM zones (EPSG:326xx, EPSG:327xx)
- **Impact**: Users can continue using `export_map_to_geotiff()` without code changes
- **Supported**: All common UTM zones (North: 32601-32660, South: 32701-32760)

## Recent Updates

- **2025-06-07**: Added rasterio==1.4.4 to environment.yml and requirements.txt
- **2025-06-07**: Made harmonica an optional dependency with graceful fallback
- **2025-06-07**: Added automatic PROJ database workaround for GeoTIFF export
- **2025-06-07**: Fixed EPSG to WKT conversion for UTM coordinate systems
