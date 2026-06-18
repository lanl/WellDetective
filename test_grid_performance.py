#!/usr/bin/env python3
"""
Performance test for WellDetective v2.0.1 grid interpolation optimization.
Tests both old (griddata) and new (LinearNDInterpolator) methods on LBL dataset.
"""

import sys
import time
from pathlib import Path

# Add WellDetective to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'WellDetective' / 'src'))

from general.WellDetective import WellDetective

# Test configuration
DATA_DIR = '/Users/jamesedlee/Desktop/Science Projects/CAFE/Orphan Wells/FY24_Osage/UAS/LBL/Raw'
FILES = [
    'SRVY0-ACQU5-100Hz.csv',
    'SRVY0-ACQU6-100Hz.csv',
    'SRVY0-ACQU7-100Hz.csv',
    'SRVY0-ACQU8-100Hz.csv',
    'SRVY0-ACQU9-100Hz.csv',
    'SRVY0-ACQU10-100Hz.csv'
]

GRID_SIZES = [50, 25, 10]  # Test different grid resolutions

def test_grid_performance():
    """Test grid interpolation performance with old vs new method."""
    
    print("="*80)
    print("WellDetective v2.0.1 Grid Interpolation Performance Test")
    print("="*80)
    print(f"\nDataset: LBL Osage ({len(FILES)} files)")
    print(f"Location: {DATA_DIR}")
    
    # Load and process data
    print("\n" + "-"*80)
    print("STEP 1: Loading and processing data...")
    print("-"*80)
    
    start_load = time.time()
    
    # Disable auto-gridding in Load_Process to control when it happens
    original_grid_size = WellDetective.DEFAULT_GRID_SIZE
    WellDetective.DEFAULT_GRID_SIZE = None  # Temporarily disable
    
    wd = WellDetective.Load_Process_MultipleMagDataFiles(
        s_folderpath=DATA_DIR,
        Mag_File_List=FILES,
        v_plotdata=False,
        v_runchecks=False,
        skip_errors=False
    )
    
    WellDetective.DEFAULT_GRID_SIZE = original_grid_size  # Restore
    load_time = time.time() - start_load
    
    n_points = len(wd.data_filtered)
    print(f"\n✓ Loaded {n_points:,} data points in {load_time:.1f} seconds")
    
    # Get UTM zone
    zone, hemi, epsg = wd.get_survey_utm_zone()
    print(f"  UTM Zone: {zone}{hemi} (EPSG:{epsg})")
    
    # Use manual IGRF values for Osage County, Oklahoma
    # (Approximately 36.5°N, 96.5°W, circa 2024)
    inclination = 63.5  # degrees
    declination = 3.5   # degrees
    print(f"\nUsing manual IGRF values (Osage County, OK):")
    print(f"  Inclination: {inclination}°")
    print(f"  Declination: {declination}°")
    
    # Run tests for each grid size
    results = []
    
    for grid_size in GRID_SIZES:
        print("\n" + "="*80)
        print(f"TESTING: Grid size = {grid_size}m")
        print("="*80)
        
        # Calculate expected grid dimensions
        easting_range = wd.data_filtered["easting"].max() - wd.data_filtered["easting"].min()
        northing_range = wd.data_filtered["northing"].max() - wd.data_filtered["northing"].min()
        n_points_x = int((easting_range / grid_size)) + 1
        n_points_y = int((northing_range / grid_size)) + 1
        total_grid_points = n_points_x * n_points_y
        
        print(f"Expected grid: {n_points_x} x {n_points_y} = {total_grid_points:,} points")
        
        # Test 1: New method (LinearNDInterpolator)
        print("\n--- Test 1: NEW METHOD (LinearNDInterpolator) ---")
        start_new = time.time()
        
        mag_grid_da_new = wd.grid_and_filter_data(
            inclination=inclination,
            declination=declination,
            grid_size=grid_size,
            cutoff_wavelength=30,
            proximity_threshold=40,
            use_fast_interpolation=True  # New method
        )
        
        time_new = time.time() - start_new
        print(f"✓ NEW method completed in {time_new:.2f} seconds")
        
        # Test 2: Old method (griddata)
        print("\n--- Test 2: OLD METHOD (griddata) ---")
        
        start_old = time.time()
        
        # Call grid_and_filter_data with old method
        mag_grid_da_old = wd.grid_and_filter_data(
            inclination=inclination,
            declination=declination,
            grid_size=grid_size,
            cutoff_wavelength=30,
            proximity_threshold=40,
            use_fast_interpolation=False  # Use old method
        )
        
        time_old = time.time() - start_old
        print(f"✓ OLD method completed in {time_old:.2f} seconds")
        
        # Calculate speedup
        speedup = time_old / time_new if time_new > 0 else 0
        time_saved = time_old - time_new
        percent_faster = ((time_old - time_new) / time_old * 100) if time_old > 0 else 0
        
        print("\n" + "-"*80)
        print(f"RESULTS for {grid_size}m grid:")
        print(f"  Old method:  {time_old:.2f} seconds")
        print(f"  New method:  {time_new:.2f} seconds")
        print(f"  Time saved:  {time_saved:.2f} seconds")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Improvement: {percent_faster:.1f}% faster")
        print("-"*80)
        
        results.append({
            'grid_size': grid_size,
            'grid_points': total_grid_points,
            'time_old': time_old,
            'time_new': time_new,
            'speedup': speedup,
            'percent_faster': percent_faster
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Performance Test Results")
    print("="*80)
    print(f"Dataset: {n_points:,} data points")
    print()
    print(f"{'Grid Size':<12} {'Grid Points':<15} {'Old (s)':<10} {'New (s)':<10} {'Speedup':<10} {'% Faster'}")
    print("-"*80)
    
    for r in results:
        print(f"{r['grid_size']:>4}m       "
              f"{r['grid_points']:>10,}      "
              f"{r['time_old']:>7.2f}    "
              f"{r['time_new']:>7.2f}    "
              f"{r['speedup']:>6.2f}x    "
              f"{r['percent_faster']:>6.1f}%")
    
    print("="*80)
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    avg_percent = sum(r['percent_faster'] for r in results) / len(results)
    
    print(f"\nAverage speedup: {avg_speedup:.2f}x ({avg_percent:.1f}% faster)")
    
    # Calculate total time saved
    total_old = sum(r['time_old'] for r in results)
    total_new = sum(r['time_new'] for r in results)
    total_saved = total_old - total_new
    
    print(f"Total time for all tests:")
    print(f"  Old method: {total_old:.1f} seconds ({total_old/60:.1f} minutes)")
    print(f"  New method: {total_new:.1f} seconds ({total_new/60:.1f} minutes)")
    print(f"  Time saved: {total_saved:.1f} seconds ({total_saved/60:.1f} minutes)")
    
    print("\n✓ Performance test completed successfully!")
    print("="*80)

if __name__ == '__main__':
    try:
        test_grid_performance()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
