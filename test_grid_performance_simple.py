#!/usr/bin/env python3
"""
Simplified performance test for grid interpolation optimization.
Loads pre-processed data and tests gridding performance only.
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add WellDetective to path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'WellDetective' / 'src'))

from general.WellDetective import WellDetective

# Use pre-merged processed data
DATA_FILE = '/Users/jamesedlee/Desktop/Science Projects/CAFE/Orphan Wells/FY24_Osage/UAS/LBL/Merged_processed_Mag_all.csv'

# Manual IGRF values for Osage County, OK (~36.5°N, 96.5°W)
INCLINATION = 63.5
DECLINATION = 3.5

GRID_SIZES = [50, 25, 10]

def test_grid_performance():
    """Test grid interpolation performance."""
    
    print("="*80)
    print("WellDetective v2.0.1 Grid Interpolation Performance Test")
    print("="*80)
    print(f"\nDataset: LBL Osage (merged processed data)")
    print(f"Location: {DATA_FILE}")
    
    # Load pre-processed data
    print("\n" + "-"*80)
    print("Loading pre-processed data...")
    print("-"*80)
    
    df = pd.read_csv(DATA_FILE)
    n_points = len(df)
    
    print(f"✓ Loaded {n_points:,} data points")
    print(f"  Columns: {list(df.columns)}")
    
    # Check for required columns
    required = ['easting', 'northing', 'Corrected']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\n✗ Missing required columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Create WellDetective object
    wd = WellDetective(df)
    wd.data_filtered = df  # Use as filtered data
    
    print(f"\nManual IGRF values (Osage County, OK):")
    print(f"  Inclination: {INCLINATION}°")
    print(f"  Declination: {DECLINATION}°")
    
    # Run tests for each grid size
    results = []
    
    for grid_size in GRID_SIZES:
        print("\n" + "="*80)
        print(f"TESTING: Grid size = {grid_size}m")
        print("="*80)
        
        # Calculate expected grid dimensions
        easting_range = df["easting"].max() - df["easting"].min()
        northing_range = df["northing"].max() - df["northing"].min()
        n_points_x = int((easting_range / grid_size)) + 1
        n_points_y = int((northing_range / grid_size)) + 1
        total_grid_points = n_points_x * n_points_y
        
        print(f"Expected grid: {n_points_x} x {n_points_y} = {total_grid_points:,} points")
        print(f"Survey area: {easting_range:.0f}m x {northing_range:.0f}m")
        
        # Test 1: New method (LinearNDInterpolator)
        print("\n--- Test 1: NEW METHOD (LinearNDInterpolator) ---")
        start_new = time.time()
        
        try:
            mag_grid_da_new = wd.grid_and_filter_data(
                inclination=INCLINATION,
                declination=DECLINATION,
                grid_size=grid_size,
                cutoff_wavelength=30,
                proximity_threshold=40,
                use_fast_interpolation=True
            )
            time_new = time.time() - start_new
            print(f"✓ NEW method completed in {time_new:.2f} seconds")
        except Exception as e:
            print(f"✗ NEW method failed: {e}")
            time_new = None
        
        # Test 2: Old method (griddata)
        print("\n--- Test 2: OLD METHOD (griddata) ---")
        start_old = time.time()
        
        try:
            mag_grid_da_old = wd.grid_and_filter_data(
                inclination=INCLINATION,
                declination=DECLINATION,
                grid_size=grid_size,
                cutoff_wavelength=30,
                proximity_threshold=40,
                use_fast_interpolation=False
            )
            time_old = time.time() - start_old
            print(f"✓ OLD method completed in {time_old:.2f} seconds")
        except Exception as e:
            print(f"✗ OLD method failed: {e}")
            time_old = None
        
        # Calculate results
        if time_new and time_old:
            speedup = time_old / time_new
            time_saved = time_old - time_new
            percent_faster = ((time_old - time_new) / time_old * 100)
            
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
        else:
            print(f"\n⚠ Skipping results for {grid_size}m (test failed)")
    
    # Summary
    if results:
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
        
        # Calculate total time
        total_old = sum(r['time_old'] for r in results)
        total_new = sum(r['time_new'] for r in results)
        total_saved = total_old - total_new
        
        print(f"\nTotal time for all tests:")
        print(f"  Old method: {total_old:.1f} seconds ({total_old/60:.1f} minutes)")
        print(f"  New method: {total_new:.1f} seconds ({total_new/60:.1f} minutes)")
        print(f"  Time saved: {total_saved:.1f} seconds ({total_saved/60:.1f} minutes)")
        
        print("\n✓ Performance test completed successfully!")
        print("="*80)
    else:
        print("\n✗ No results to display (all tests failed)")

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
