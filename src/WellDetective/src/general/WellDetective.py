# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:16:22 2025

@author: Eric Guiltinan, Nash Taylor, James E. Lee
"""

#WellDetective.py
# Standard library imports
import os
import datetime
from pathlib import Path
from typing import Tuple

# Third-party imports (core)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Scientific/geospatial
import pyproj
import xrft
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.cluster import MeanShift
from scipy.spatial import distance

# Machine learning
from sklearn.cluster import KMeans, MeanShift

# Optional imports (handle gracefully if missing)
try:
    import harmonica as hm
    HAS_HARMONICA = True
except ImportError:
    HAS_HARMONICA = False
    print("Warning: harmonica not installed. Reduction to pole and filtering will not be available.")

try:
    import pyIGRF14 as pyIGRF
    HAS_PYIGRF = True
except ImportError:
    HAS_PYIGRF = False
    print("Warning: pyIGRF not installed. IGRF calculations will not be available.")

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Affects exporting data as GeoTIFF.")

try:
    import mpl_toolkits
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MPL_TOOLKITS = True
except ImportError:
    HAS_MPL_TOOLKITS = False
    print("Warning: mpl_toolkits not installed. May affect plotting functionality.")
    

    
class WellDetective:
    """
    A class for processing geophysical magnetic survey data.

    Provides methods to read raw magnetic data, apply heading corrections, remove turning data,
    grid and filter data, identify magnetic anomalies ("hotspots"), and export results for visualization.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing processed magnetic survey data.
    """

    
    ############################################################
    # 0. Column Name Definitions
    ############################################################
    # Column name mappings (class constants)
    LAT_COLUMNS = ['Latitude[°]', 'Latitude [Decimal Degrees]', 'Latitude', 'Lat']
    LON_COLUMNS = ['Longitude[°]', 'Longitude [Decimal Degrees]', 'Longitude', 'Lon', 'Long']
    MAG_COLUMNS = ['Totalfield[nT]', 'Mag', 'Total_Avg', 'Total_B1', 'Total_B2', 'Total_B3', 'Total_B4', 'Total_B5']
    DATE_COLUMNS = ['Date', 'GPSDate', 'DateTime', 'Timestamp', 'GPSDate [yyyy/mm/dd]']
    MAG_X_COLUMNS = ['B1x [nT]', 'B2x [nT]', 'B3x [nT]', 'B4x [nT]', 'B5x [nT]', 'Mag-X[nT]']
    MAG_Y_COLUMNS = ['B1y [nT]', 'B2y [nT]', 'B3y [nT]', 'B4y [nT]', 'B5y [nT]', 'Mag-Y[nT]']
    MAG_Z_COLUMNS = ['B1z [nT]', 'B2z [nT]', 'B3z [nT]', 'B4z [nT]', 'B5z [nT]', 'Mag-Z[nT]']
    
    # Default parameters
    ## Processing Total Magnetic Field
    DEFAULT_BASELINE_M = 0.5  # Distance between sensors in dual-magnetometer system (meters)
    
    ## Determining/Filtering primary/secondary headings
    DEFAULT_HEADING_WINDOW = 20  # Look-back window size (data points) for heading calculation - reduces GPS noise
    DEFAULT_HEADING_TOLERANCE = 20  # Tolerance (degrees) around primary/secondary headings - data outside range is removed
    
    ## Heading Correction Baseline Method
    DEFAULT_HEADING_METHOD = 0  # Background correction method: 0=mean, 1=median, 2=Gaussian fit
    
    ## Filtering valid Segments
    DEFAULT_MAX_GAP_DIST = 10  # Maximum gap (meters) between consecutive points - larger gaps split segments
    DEFAULT_MIN_SEG_LEN = 150  # Minimum segment length (meters) to keep - shorter segments are removed
    
    ## Mesh Grid parameters
    DEFAULT_GRID_SIZE = 5  # Grid cell size (meters) - smaller values give finer resolution
    DEFAULT_CUTOFF_WAVELEN = 30  # Low-pass filter wavelength (meters) - removes high-frequency noise
    DEFAULT_PROX_THRESH = 40  # Proximity threshold (meters) - masks grid points far from flight lines
    DEFAULT_USE_FAST_INTERPOLATION = True  # Use optimized LinearNDInterpolator (3-5x faster)
    
    ## Used for Plotting Maps
    DEFAULT_MAP_XRES = 500  # X-axis tick spacing (meters) for map plots
    DEFAULT_MAP_YRES = 500  # Y-axis tick spacing (meters) for map plots
    
    ############################################################
    # 1. __init__
    ############################################################
    def __init__(self, data: pd.DataFrame):
        """
        Initialize GeoMagProcessor with survey data.

        Parameters
        ----------
        data : pd.DataFrame
            Initial survey data with required columns ('Lat', 'Long', 'Mag').
        """
        self.data_raw = data.copy()  # Always preserve raw
        self.data = data.copy()      # Working data
        self.data_filtered = None    # Working data
        self.processing_log = []     # Track changes

    ############################################################
    # 2. Static methods (utilities that don't need instance)
    ############################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Self  (Level: N/A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def check_optional_dependencies():
        """Check which optional dependencies are installed."""
        print("Optional Dependencies:")
        print(f"  pyIGRF:     {'✓ Available' if HAS_PYIGRF else '✗ Not installed'}")
        print(f"  harmonica:   {'✓ Available' if HAS_HARMONICA else '✗ Not installed'}")
        print(f"  rasterio:   {'✓ Available' if HAS_RASTERIO else '✗ Not installed'}")
        print(f"  mpl_toolkits:   {'✓ Available' if HAS_MPL_TOOLKITS else '✗ Not installed'}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # General Tools  (Level: N/A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def is_numeric(s: str) -> bool:
        """Check if string is numeric."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod    
    def to_decimal_year(date):
        # Get the start of the current and next year
        start_of_year = datetime.datetime(date.year, 1, 1)
        start_of_next_year = datetime.datetime(date.year + 1, 1, 1)
        
        # Calculate elapsed time and total year duration in seconds
        year_duration = (start_of_next_year - start_of_year).total_seconds()
        year_elapsed = (date - start_of_year).total_seconds()
        
        return date.year + (year_elapsed / year_duration)
    
    @staticmethod
    def get_matching_columns(df: pd.DataFrame, column_list: list) -> list:
        """Return list of DataFrame columns that exist in the string list."""
        return [col for col in df.columns if col in column_list]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loading files  (Level: .data_raw)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def detect_header_generic(filepath: str, max_lines: int = 50) -> Tuple[int, str, int]:
        """
        Generic function to detect header line, delimiter, and number of data fields.
        Handles cases where header and data may use different delimiters.
        Returns r'\\s+' for whitespace-delimited files (tabs/spaces).
        
        Args:
            filepath: Path to file
            max_lines: Maximum lines to check. Use optional input to increase if header is expected beyond line 50
        
        Returns:
            Tuple of (header_line_number_0indexed, delimiter_for_data, num_columns)
            Note: Returns r'\\s+' for whitespace-delimited files
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [f.readline() for _ in range(max_lines)]
        
        # All common delimiters
        all_delimiters = ['\t', ';', '|', ',', ' ']
        
        best_match = None
        best_score = 0
        best_num_cols = 0
        
        for i, line in enumerate(lines):
            # Skip line if empty
            if not line.strip():
                continue
            
            # Skip if search for each delimiter in the current line (potential header) is empty
            for header_delim in all_delimiters:
                if header_delim not in line:
                    continue

                # Count only non-empty fields
                # fields = [f.strip() for f in line.strip().split(header_delim) if f.strip()]
                
                 # Keep all fields (including empty) for counting
                fields = [f.strip() for f in line.strip().split(header_delim)]
                # But filter empty for analysis
                fields_non_empty = [f for f in fields if f]

                # Check if this looks like a header (at least 3 columns)
                if len(fields_non_empty) < 3:
                    continue
                
                # Check if mostly non-numeric (header characteristic)
                header_numeric = sum(1 for f in fields if WellDetective.is_numeric(f))
                if header_numeric >= len(fields) * 0.5:
                    continue  # Too many numbers to be a header
                
                # print(f'L{i}: Possible Header')
                # Now check if ANY subsequent line (within next 3) looks like data
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                        
                    # Try each delimiter for the data line (might be different from header!)
                    for data_delim in all_delimiters:
                        if data_delim not in next_line:
                            continue

                        # Keep all fields (including empty) for counting
                        next_fields = [f.strip() for f in next_line.split(data_delim)]
                        # But filter empty for numeric analysis
                        next_fields_non_empty = [f for f in next_fields if f]

                        # Need similar number of fields
                        if abs(len(fields) - len(next_fields)) > 2:
                            # print(f'\tL{j}: Suspect not data: (n_headers - n_data)={len(fields) - len(next_fields)}')
                            continue
                        
                        # Data should be mostly numeric
                        data_numeric = sum(1 for f in next_fields_non_empty if WellDetective.is_numeric(f))
                        if data_numeric < len(next_fields_non_empty) * 0.6:
                            # print(f'\tL{j}: Suspect not data: (% numeric)={data_numeric/len(next_fields)}')
                            continue  # Not enough numbers to be data

                        # print(f'\tL{j}: Likely data!')
                        
                        # Calculate a score: prefer more fields, higher numeric ratio, and non-space delimiters
                        delimiter_bonus = 50 if data_delim != ' ' else 0
                        score = len(fields_non_empty) + (data_numeric / len(next_fields_non_empty)) * 100 + delimiter_bonus
                        
                        if score > best_score:
                            # print('New Best Score')
                            best_score = score
                            best_num_cols = len(fields)  # Store number of columns
                            # If header uses space and data uses tab/space, return \s+ pattern
                            if header_delim in [' ', '\t'] and data_delim in [' ', '\t']:
                                best_match = (i, r'\s+')
                            else:
                                best_match = (i, data_delim)
                            break  # Found good match for this header candidate
                    
                    if best_match and best_match[0] == i:
                        break  # Found match for this header, move to next potential header
        
        if best_match:
            return best_match[0], best_match[1], best_num_cols
        
        return -1, 'none', -1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handle Missing Data Fields (Level: Data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def calculate_heading(lat1, lon1, lat2, lon2):
        """
        Calculate compass heading between two geographical points.

        Parameters
        ----------
        lat1, lon1, lat2, lon2 : float
            Latitude and longitude coordinates in degrees.

        Returns
        -------
        float
            Heading in degrees (0° is North).
        """
        delta_lon = np.radians(lon2 - lon1)
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        x = np.sin(delta_lon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
        initial_heading = np.arctan2(x, y)
        heading = np.degrees(initial_heading)
        compass_heading = (heading + 360+90) % 360
        return compass_heading

    @staticmethod
    def get_utm_zone_from_lon_lat(longitude, latitude):
        """
        Determine UTM zone from longitude and latitude.
        
        Parameters
        ----------
        longitude : float
            Longitude in decimal degrees (-180 to 180)
        latitude : float
            Latitude in decimal degrees (-90 to 90)
        
        Returns
        -------
        tuple : (zone_number, hemisphere, epsg_code)
            zone_number: int (1-60)
            hemisphere: str ('N' or 'S')
            epsg_code: int (EPSG code for the UTM zone)
        
        Examples
        --------
        >>> WellDetective.get_utm_zone_from_lon_lat(-111.5, 40.7)
        (12, 'N', 32612)
        
        >>> WellDetective.get_utm_zone_from_lon_lat(151.2, -33.9)
        (56, 'S', 32756)
        """
        # Calculate zone number from longitude
        # UTM zones are 6 degrees wide, starting at -180
        zone_number = int((longitude + 180) / 6) + 1
        
        # Handle edge cases
        if zone_number < 1:
            zone_number = 1
        elif zone_number > 60:
            zone_number = 60
        
        # Determine hemisphere
        hemisphere = 'N' if latitude >= 0 else 'S'
        
        # Calculate EPSG code
        # Northern hemisphere: 32601-32660
        # Southern hemisphere: 32701-32760
        if hemisphere == 'N':
            epsg_code = 32600 + zone_number
        else:
            epsg_code = 32700 + zone_number
        
        return zone_number, hemisphere, epsg_code

            
    @staticmethod
    def _get_utm_wkt(zone, northern=True):
        """
        Generate WKT for UTM zone (workaround for PROJ database issues).
        
        Parameters
        ----------
        zone : int
            UTM zone number (1-60)
        northern : bool
            True for Northern hemisphere, False for Southern
            
        Returns
        -------
        str
            WKT string for the UTM zone
        """
        central_meridian = -177 + (zone - 1) * 6
        false_northing = 0 if northern else 10000000
        hemisphere = "N" if northern else "S"
        
        wkt = f'''PROJCS["WGS 84 / UTM zone {zone}{hemisphere}",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",{central_meridian}],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",{false_northing}],
    UNIT["metre",1]]'''
        return wkt

        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figures (Level: N/A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def make_utm_easting_formatter(target_ax=None):
        """
        Factory for Easting formatter.
        - Leftmost VISIBLE tick: full format (²49⁰⁰⁰E)
        - Others with increment ≥1000m: ²49E
        - Others with increment <1000m: ²49⁵E
        
        Parameters
        ----------
        target_ax : matplotlib.axes.Axes, optional
            The axes object to format. If None, uses plt.gca() (may fail with multiple axes)
        """
        tick_increment = [None]
        leftmost_visible = [None]
        
        def formatter(x, pos):
            x_int = int(x)
            x_str = f"{x_int}"
            
            # Get the correct axis
            ax = target_ax if target_ax is not None else plt.gca()
            xticks = ax.get_xticks()
            xlim = ax.get_xlim()
            
            # Filter ticks to only those within visible range
            visible_ticks = [t for t in xticks if xlim[0] <= t <= xlim[1]]
            
            # Determine leftmost VISIBLE tick
            if leftmost_visible[0] is None and len(visible_ticks) > 0:
                leftmost_visible[0] = min(visible_ticks)
            
            # Determine increment
            if tick_increment[0] is None and len(visible_ticks) > 1:
                tick_increment[0] = abs(visible_ticks[1] - visible_ticks[0])
            
            is_leftmost = (leftmost_visible[0] is not None and abs(x - leftmost_visible[0]) < 1)
            
            if len(x_str) >= 5:
                super_low = x_str[-3:]      # Last 3 (000 or 500)
                regular = x_str[-5:-3]      # Middle 2 (49)
                super_high = x_str[:-5]     # First N (2)
                
                if super_high:
                    if is_leftmost:
                        # Full format for leftmost visible tick
                        return f"$^{{{super_high}}}${regular}$^{{{super_low}}}$E"
                    elif tick_increment[0] and tick_increment[0] < 1000:
                        # Small increment: show hundreds digit
                        hundreds_digit = x_str[-3]
                        return f"$^{{{super_high}}}${regular}$^{{{hundreds_digit}}}$"
                    else:
                        # Large increment (≥1000m): no hundreds
                        return f"$^{{{super_high}}}${regular}"
                else:
                    if is_leftmost:
                        return f"{regular}$^{{{super_low}}}$E"
                    elif tick_increment[0] and tick_increment[0] < 1000:
                        hundreds_digit = x_str[-3]
                        return f"{regular}$^{{{hundreds_digit}}}$"
                    else:
                        return f"{regular}"
            else:
                return f"{x_int}E"
        
        return formatter
        
    @staticmethod    
    def make_utm_northing_formatter(target_ax=None):
        """
        Factory for Northing formatter.
        - Topmost VISIBLE tick: full format (⁴¹23⁰⁰⁰N)
        - Others with increment ≥1000m: ⁴¹23N
        - Others with increment <1000m: ⁴¹23⁵N
        
        Parameters
        ----------
        target_ax : matplotlib.axes.Axes, optional
            The axes object to format. If None, uses plt.gca() (may fail with multiple axes)
        """
        tick_increment = [None]
        
        def formatter(y, pos):
            y_int = int(y)
            y_str = f"{y_int}"
            
            # Get the correct axis
            ax = target_ax if target_ax is not None else plt.gca()
            yticks = ax.get_yticks()
            ylim = ax.get_ylim()
            
            # Filter ticks to only those within visible range
            visible_ticks = [t for t in yticks if ylim[0] <= t <= ylim[1]]
            
            # Determine topmost VISIBLE tick (recalculate each time)
            topmost_visible = max(visible_ticks) if len(visible_ticks) > 0 else None
            
            # Determine increment
            if tick_increment[0] is None and len(visible_ticks) > 1:
                tick_increment[0] = abs(visible_ticks[1] - visible_ticks[0])
            
            is_topmost = (topmost_visible is not None and abs(y - topmost_visible) < 1)
            
            if len(y_str) >= 5:
                super_low = y_str[-3:]
                regular = y_str[-5:-3]
                super_high = y_str[:-5]
                
                if super_high:
                    if is_topmost:
                        # Full format for topmost visible tick
                        return f"$^{{{super_high}}}${regular}$^{{{super_low}}}$N"
                    elif tick_increment[0] and tick_increment[0] < 1000:
                        # Small increment: show hundreds
                        hundreds_digit = y_str[-3]
                        return f"$^{{{super_high}}}${regular}$^{{{hundreds_digit}}}$"
                    else:
                        # Large increment
                        return f"$^{{{super_high}}}${regular}"
                else:
                    if is_topmost:
                        return f"{regular}$^{{{super_low}}}$N"
                    elif tick_increment[0] and tick_increment[0] < 1000:
                        hundreds_digit = y_str[-3]
                        return f"{regular}$^{{{hundreds_digit}}}$"
                    else:
                        return f"{regular}"
            else:
                return f"{y_int}N"
        
        return formatter

        
    ############################################################
    # Public Methods
    ############################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and process Mag Data files  (Level: Data_Raw)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def Load_Process_SingleMagDataFile(
        s_folderpath: str,
        Mag_File: str,
        v_plotdata = False,
        v_runchecks = False
    ) -> "WellDetective":
        """
        Load and process a single magnetometry file.
        
        Args:
            s_folderpath: (str) Path to folder containing the file
            Mag_File: (str) Filename to process
            v_plotdata: (logic) open up plots from data
            v_runchecks: (logic) Verbose output for debugging
        
        Returns:
            WellDetective: Object with processed data and attributes:
                - data_raw: Preserved raw data copy
                - data: Working data with calculated fields
                - data_filtered: Filtered data with invalid points removed
                - processing_log: List tracking all processing steps
        
        Raises:
            TypeError: If self is not a pandas DataFrame
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Constants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # Date/Time Fields
        # v_date = datetime.datetime(1901,1,1,0,0) # Used in case there is no date information in the file.
        # v_deltaT = 0.759134
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # File handling
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # Create a Path object
        file_path = Path(s_folderpath) / Mag_File
                
        # # Verify that file exists
        # # Check if it exists AND is a file (not a directory)
        if v_runchecks:
            if file_path.is_file():
                print("File exists!: Loading type: " + file_path.suffix)
            elif os.path.exists(s_folderpath):
                print("Well, at least there is a folder here: "+s_folderpath)
            else:
                print("File does not exist. Current Dir: "+os.getcwd())
        
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        if v_runchecks:
            print(f"File exists! Loading type: {file_path.suffix}")
        
        # # Read raw data
        # # Determine Header Line Number and Delimiter of file type
        headerline, delim, num_cols = WellDetective.detect_header_generic(str(file_path))
        if v_runchecks:
            print(f"Detected header line: {headerline} (0-indexed)")
            print(f"Detected delimiter: {repr(delim)}")
            print(f"Detected number of fields: {repr(num_cols)}")
        
        # # Read File
        data = pd.read_csv(str(file_path), 
                           sep=delim,
                           skiprows=headerline,
                           quotechar = "'",
                           usecols=range(num_cols),
                           on_bad_lines = 'warn'
                          )
        
        # # Clean column names
        data.columns = data.columns.str.lstrip() # Some header formats result in spaces at beginning of column names
        
        # # Verify that data was loaded correctly
        if v_runchecks:
            print("data: ")
            for col in data.columns: print(f"\t{col}")
            print("\nData Snapshot")
            print(data.iloc[97:103])

        # #Preserve Metadata
        data['Filepath'] = str(file_path)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Handle Missing Data Fields (Level: Data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # Activate WellDetective Object
        New_data = WellDetective(data)
        
        # # Check for Date Information
        New_data.Check_4_DateCol()
        DateCol = WellDetective.get_matching_columns(New_data.data,WellDetective.DATE_COLUMNS)

        # Check if already datetime (from Check_4_DateCol)
        if pd.api.types.is_datetime64_any_dtype(New_data.data[DateCol[0]]):
            # Already datetime - just get first valid value
            v_date = New_data.data[DateCol[0]].dropna().iloc[0]
            if v_runchecks:
                print(f"Using date from data: {v_date}")
        else:
            # Still string - need to parse
            first_value = New_data.data[New_data.data[DateCol[0]].str.len() > 1][DateCol[0]].iloc[0]
            
            # Try multiple date formats
            date_formats = ["%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"]
            v_date = None
            for fmt in date_formats:
                try:
                    v_date = datetime.datetime.strptime(first_value, fmt)
                    break
                except ValueError:
                    continue
            
            if v_date is None:
                raise ValueError(f"Unable to parse date '{first_value}'. Tried formats: {date_formats}")
        
        if v_runchecks:
            print(v_date)
        
        # # Determine Lat/Long fields in dataframe and interpolate sparse values
        latcol = WellDetective.get_matching_columns(New_data.data,WellDetective.LAT_COLUMNS)
        loncol = WellDetective.get_matching_columns(New_data.data,WellDetective.LON_COLUMNS)
        if not latcol or not loncol:
            raise ValueError("Latitude or Longitude columns not found in data")
        New_data.Check_4_LatLon(latcol[0],loncol[0])
        
        # # Calculate Total magnetic field if field doesn't exist
        New_data.Check_4_MagTotal()
        magcol = WellDetective.get_matching_columns(New_data.data,WellDetective.MAG_COLUMNS)
        if not magcol:
            raise ValueError("Magnetic Field columns not found in data")
            
        # # Add heading column
        New_data.add_heading_column(lat_col=latcol[0], lon_col=loncol[0], window = WellDetective.DEFAULT_HEADING_WINDOW)
        
        # # Verify that data still looks good.
        if v_runchecks:
            print("data (L1):\t(fields)")
            for col in New_data.data.columns: print(f"\t{col}")
            print("\nData (L1) Snapshot")
            New_data.data
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Filter-clean Data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # Start work on filtered layer
        New_data.start_filtering()
        
        # # Clear rows with 0's and NaN values in Coordinates
        New_data.data_filtered.dropna(subset=[latcol[0], loncol[0], 'Heading'], inplace=True)
        
        # # Find primary/secondary headings
        primary_range, secondary_range = New_data.find_primary_secondary_headings(tolerance=WellDetective.DEFAULT_HEADING_TOLERANCE)
        print(f'Primary Heading Range: {primary_range[0]:.2f} - {primary_range[1]:.2f}°')
        print(f'Secondary Heading Range: {secondary_range[0]: .2f} - {secondary_range[1]: .2f}°')
        
        # Remove turning data
        New_data.remove_turning_data(primary_range, secondary_range) 
        
        #Create segments and remove ones below a certain length
        New_data.segment_and_filter_data(max_gap_distance=WellDetective.DEFAULT_MAX_GAP_DIST, 
                                         min_segment_length=WellDetective.DEFAULT_MIN_SEG_LEN, 
                                         lat_col=latcol[0], 
                                         lon_col=loncol[0])
        
        # Equalize heading corrections
        New_data.auto_normalize_heading_correction(primary_range, secondary_range, mag_col=magcol[0],method=WellDetective.DEFAULT_HEADING_METHOD)
        
        # Project coordinates (UTM zone 12)
        zone, hemi, epsg = New_data.get_survey_utm_zone(lat_col=latcol[0], lon_col=loncol[0])
        New_data.project_coordinates(utm_zone=zone, lat_col=latcol[0], lon_col=loncol[0])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if v_plotdata:
            New_data.plot_flight_tracks(E_incr=WellDetective.DEFAULT_MAP_XRES, N_incr=WellDetective.DEFAULT_MAP_YRES)
            
        return New_data

    @staticmethod
    def Load_Process_MultipleMagDataFiles(
        s_folderpath: str,
        Mag_File_List: list,
        v_plotdata: False,
        v_runchecks=False,
        skip_errors=False
    ) -> "WellDetective":
        """
        Load and process multiple magnetometry files and combine them.
        
        Args:
            s_folderpath: Path to folder containing files
            Mag_File_List: List of filenames to process
            v_runchecks: Verbose output
            skip_errors: If True, skip files that fail; if False, raise error
        
        Returns:
            WellDetective object with concatenated data from all files

        Notes:
            - Filename is already added to data_raw in Load_Process_SingleMagDataFile()
            - Processing log tracks each file processed
        """
        all_data_raw = []
        all_data = []
        all_data_filtered = []
        processing_log = []
        failed_files = []
        
        # Process each file
        print('Loading files from Mag_File_List')
        for i, Mag_File in enumerate(Mag_File_List, 1):
            print(f"\n[{i}/{len(Mag_File_List)}] Processing: {Mag_File}")
            
            try:
                # Load and process single file
                wd = WellDetective.Load_Process_SingleMagDataFile(
                    s_folderpath, 
                    Mag_File, 
                    v_runchecks
                )
                
                # Validate that all required data exists before appending
                if wd.data_raw is None or wd.data_raw.empty:
                    raise ValueError(f"data_raw is empty or None for file {Mag_File}")
                if wd.data is None or wd.data.empty:
                    raise ValueError(f"data is empty or None for file {Mag_File}")
                if wd.data_filtered is None or wd.data_filtered.empty:
                    raise ValueError(f"data_filtered is empty or None for file {Mag_File}")
                
                # Collect data from each level
                all_data_raw.append(wd.data_raw)
                all_data.append(wd.data)
                all_data_filtered.append(wd.data_filtered)
                
                # Add to processing log
                processing_log.append({
                    'file': Mag_File,
                    'status': 'success',
                    'rows_raw': len(wd.data_raw),
                    'rows_data': len(wd.data),
                    'rows_filtered': len(wd.data_filtered),
                    'timestamp': pd.Timestamp.now()
                })
                
                # Extend the individual file's processing log
                processing_log.extend(wd.processing_log)
                
                print(f"  ✓ Success: {len(wd.data_filtered):,} filtered rows")
                
            except Exception as e:
                failed_files.append((Mag_File, str(e)))
                
                # Log the failure
                processing_log.append({
                    'file': Mag_File,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': pd.Timestamp.now()
                })
                
                print(f"  ✗ Failed: {e}")
                
                if not skip_errors:
                    raise  # Re-raise the exception
        
        # Check if we have any data
        if not all_data_raw:
            raise ValueError("No files were successfully processed")
        
        # Validate column consistency across files
        if len(all_data_raw) > 1:
            first_cols = set(all_data_raw[0].columns)
            for i, df in enumerate(all_data_raw[1:], 2):
                if set(df.columns) != first_cols:
                    missing = first_cols - set(df.columns)
                    extra = set(df.columns) - first_cols
                    warning_msg = f"Column mismatch in file {i}:"
                    if missing:
                        warning_msg += f" Missing: {missing}."
                    if extra:
                        warning_msg += f" Extra: {extra}."
                    print(f"  ⚠ Warning: {warning_msg}")
        
        # Concatenate all dataframes
        combined_data_raw = pd.concat(all_data_raw, ignore_index=True)
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data_filtered = pd.concat(all_data_filtered, ignore_index=True)
        
        # Create new WellDetective object with combined data
        combined_wd = WellDetective(combined_data_raw)
        combined_wd.data = combined_data
        combined_wd.data_filtered = combined_data_filtered
        combined_wd.processing_log = processing_log  # Attach combined processing log

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Files processed: {len(all_data_raw)}/{len(Mag_File_List)}")
        print(f"  Files failed: {len(failed_files)}")
        print(f"\n  Combined data:")
        print(f"    data_raw:      {len(combined_data_raw):,} rows")
        print(f"    data:          {len(combined_data):,} rows")
        print(f"    data_filtered: {len(combined_data_filtered):,} rows")
        
        if failed_files:
            print(f"\n  Failed files:")
            for filename, error in failed_files:
                print(f"    - {filename}: {error}")
        
        print(f"\n")
        print(f"{'='*60}")
        
        # Create Spatially Mapped Data
        print(f"PROCESSING FUNCTIONS:")
        print(f"  1. Find primary/secondary headings:")
        print(f"\tprimary_range, secondary_range = wd.find_primary_secondary_headings(tolerance={WellDetective.DEFAULT_HEADING_TOLERANCE},...)")
        print(f"  2. Clear rows with 0's and NaN values in Coordinates:")
        print(f"\twd.data_filtered.dropna(subset=[latcol[0], loncol[0]], inplace=True)")
        print(f"  3. Remove turning data:")
        print(f"\twd.remove_turning_data(primary_range, secondary_range)")
        print(f"  4. Create segments and remove ones below a certain length")
        print(f"\tNew_data.segment_and_filter_data(max_gap_distance={WellDetective.DEFAULT_MAX_GAP_DIST}, min_segment_length={WellDetective.DEFAULT_MIN_SEG_LEN}, lat_col=latcol[0], lon_col=loncol[0])")
        
        print(f"\n")
        print(f"{'='*60}")
        
        # Create Spatially Mapped Data
        print(f"MAPPING GRIDDED DATA:")
        print(f"  1. Mesh grid:")
        print(f"  WD.create_spatial_map(grid_size={WellDetective.DEFAULT_GRID_SIZE}, "
              f"cutoff_wavelength={WellDetective.DEFAULT_CUTOFF_WAVELEN}, "
              f"proximity_threshold={WellDetective.DEFAULT_PROX_THRESH}, "
              f"use_pyigrf=True, inclination=None, declination=None, date=None, "
              f"v_plotdata=False)")
        
        print(f"  3. Generate Spatial Map:")
        combined_wd.create_spatial_map(grid_size=WellDetective.DEFAULT_GRID_SIZE, 
                                       cutoff_wavelength=WellDetective.DEFAULT_CUTOFF_WAVELEN, 
                                       proximity_threshold=WellDetective.DEFAULT_PROX_THRESH,
                                       use_pyigrf=True, 
                                       inclination=None, 
                                       declination=None, 
                                       date=None,
                                       v_plotdata=False)
        
        
        print(f"\n")
        print(f"{'='*60}")
        
        if v_plotdata:
            combined_wd.plot_flight_tracks(E_incr=WellDetective.DEFAULT_MAP_XRES, N_incr=WellDetective.DEFAULT_MAP_YRES)
            combined_wd.plot_Mag_Heat(E_incr=WellDetective.DEFAULT_MAP_XRES, N_incr=WellDetective.DEFAULT_MAP_YRES, figsize=(12, 10), save_path=None)
            
        # 
        return combined_wd



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Handling Missing Columns (Level: Data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    def Check_4_DateCol(
        self,
        v_date: datetime.date = None,
        v_deltaT: float = .001,
        interactive: bool = True
    ):
        """
        Determine if there is date information imported from delimited file.
        If no date column is found, prompt user for date information (if interactive=True).
        
        Args:
            v_date: Default date to use if no date column found (optional)
            v_deltaT: Time step in days (default: 0.001)
            interactive: If True, prompt user when date column missing (default: True)
        
        Returns:
            self (WellDetective): For method chaining
        
        Raises:
            ValueError: If no date column found and interactive=False and v_date=None
        """
        # Check for existing date columns
        DateCol = self.get_matching_columns(self.data, self.DATE_COLUMNS)
        
        if DateCol:
            # Date column found
            print(f"✓ Found date column: {DateCol[0]}")
            
            # Convert to datetime first (handles NaN properly)
            self.data[DateCol[0]] = pd.to_datetime(self.data[DateCol[0]], errors='coerce')
            
            # Count missing values before filling
            n_missing = self.data[DateCol[0]].isna().sum()
            if n_missing > 0:
                print(f"  \tForward-filling {n_missing:,} sparse date values...")
                self.data[DateCol[0]] = self.data[DateCol[0]].ffill()
                
                # If first value is still NaN, backward fill
                if pd.isna(self.data[DateCol[0]].iloc[0]):
                    print(f"  \tBackward-filling dates from first valid entry...")
                    self.data[DateCol[0]] = self.data[DateCol[0]].bfill()
            
            return self
        
        # No date column found - need to handle this
        print("\n⚠ No date column found in data")
        print(f"   Available columns: {list(self.data.columns)}")
        print(f"   Expected column names: {self.DATE_COLUMNS}")
        
        if interactive:
            # Interactive mode - prompt user
            print("\nOptions:")
            print("  1. Enter a date manually")
            print("  2. Select an existing column as the date column")
            print("  3. Use default date (1901-01-01)")
            
            while True:
                choice = input("\nSelect option (1, 2, or 3): ").strip()
                
                if choice == "1":
                    # Manual date entry
                    while True:
                        date_str = input("Enter date (YYYY-MM-DD): ").strip()
                        try:
                            v_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                            print(f"✓ Using date: {v_date.strftime('%Y-%m-%d')}")
                            break  # Exit the date entry loop
                        except ValueError:
                            print(f"  ✗ Invalid format. Please use YYYY-MM-DD")
                    break  # Exit the main choice loop
                    
                elif choice == "2":
                    print("\nAvailable columns:")
                    for i, col in enumerate(self.data.columns, 1):
                        sample = self.data[col].iloc[0] if len(self.data) > 0 else "N/A"
                        print(f"  {i}. {col:20s} (sample: {sample})")
                    
                    while True:
                        col_input = input("\nSelect column number (or 'c' to cancel): ").strip()
                        
                        if col_input.lower() == 'c':
                            print("  Cancelled column selection")
                            break  # Go back to main choice menu
                        
                        try:
                            col_idx = int(col_input) - 1
                            if 0 <= col_idx < len(self.data.columns):
                                selected_col = self.data.columns[col_idx]
                                
                                # Try to convert to datetime
                                try:
                                    self.data[selected_col] = pd.to_datetime(self.data[selected_col], errors='coerce')
                                    
                                    # Forward-fill sparse date values
                                    n_missing = self.data[selected_col].isna().sum()
                                    if n_missing > 0:
                                        print(f"  \t✓ Forward-filling {n_missing:,} sparse date values...")
                                        self.data[selected_col] = self.data[selected_col].ffill()
                                        
                                        # Backward fill if first value is still NaN
                                        if pd.isna(self.data[selected_col].iloc[0]):
                                            print(f"  \t✓ Backward-filling from first valid entry...")
                                            self.data[selected_col] = self.data[selected_col].bfill()
                                        
                                        # Check if all filled
                                        remaining_missing = self.data[selected_col].isna().sum()
                                        if remaining_missing > 0:
                                            print(f"  ⚠ Warning: {remaining_missing:,} dates still missing (no valid dates found)")
                                    
                                    print(f"✓ Using '{selected_col}' as date column")
                                    return self  # Success - exit entire function
                                    
                                except Exception as e:
                                    print(f"  ✗ Cannot convert '{selected_col}' to datetime: {e}")
                                    retry = input("  Try another column? (y/n): ").strip().lower()
                                    if retry != 'y':
                                        break  # Go back to main choice menu
                            else:
                                print(f"  ✗ Invalid choice. Enter 1-{len(self.data.columns)}")
                        except ValueError:
                            print("  ✗ Invalid input. Enter a number or 'c' to cancel.")
                    
                    # If we get here from column selection, continue the main loop
                    continue
                    
                elif choice == "3":
                    # Use default
                    v_date = datetime.datetime(1901, 1, 1, 0, 0)
                    print(f"✓ Using default date: {v_date}")
                    break  # Exit the main choice loop
                    
                else:
                    print("  ✗ Invalid choice. Enter 1, 2, or 3")
            
            # If choice 2 returned successfully, we won't reach here
            # Otherwise, we have v_date set from choice 1 or 3
        
        else:
            # Non-interactive mode
            if v_date is None:
                raise ValueError(
                    "No date column found and no default date provided. "
                    "Either provide v_date parameter or set interactive=True"
                )
        
        # Add date and time columns with fixed values
        self.data["Date"] = pd.to_datetime(v_date)
        self.data["Time"] = pd.to_timedelta(v_deltaT, unit="D")
        print(f"✓ Added 'Date' column with value: {v_date}")
        print(f"✓ Added 'Time' column with Δt: {v_deltaT} days")
        
        return self

    def Check_4_LatLon(
        self,
        latcolname: str = "Lat",
        loncolname: str = "Lon",
    ):
        """
        Interpolate lat/lon coordinates for sparsely recorded values
        
        Args:
            self: DataFrame containing magnetometer data
            latcolname: Name of self field containing Latitude data
            loncolname: Name of self field containing Longitude data
        Returns:
            DataFrame with added 0,0 values replaced by linearly interpolated locations:
                - : 
        Raises:
            TypeError: If self is not a pandas DataFrame
        """
        # Type checking
        # if not isinstance(self.data, pd.DataFrame):
        #     raise TypeError(f"df must be a pandas DataFrame, got {type(self).__name__}")
        
        for col in [latcolname, loncolname]:
            self.data[col] = self.data[col].replace(0, np.nan).interpolate(method='linear')
        
        return self
    
    def Check_4_MagTotal(
        self
    ):
        """        
        Determine if there is Mag information imported from delimited file
        
        Args:
            self: DataFrame containing magnetometer data
            v_date: Manually entered information measurement date
            v_deltaT: Manually entered information about time steps
        Returns:
            DataFrame with added assigned fields:
                - Date: Total field magnitude from sensor 1
                - Time: Total field magnitude from sensor 2
        
        Raises:
            TypeError: If self is not a pandas DataFrame
        """
        # Type checking
        # if not isinstance(self.data, pd.DataFrame):
        #     raise TypeError(f"df must be a pandas DataFrame, got {type(self).__name__}")
        
        # Are any of the Columns in the dataframe matching a date format
    
        # Option 1:
        # Likely labels for a Date Column Name
        magcol = self.get_matching_columns(self.data, self.MAG_COLUMNS)
        
        # Operate if No Columns Found
        if not magcol:
            # Check for vectorized magnetometry data
            magXcol = self.get_matching_columns(self.data,self.MAG_X_COLUMNS)
            magYcol = self.get_matching_columns(self.data,self.MAG_Y_COLUMNS)
            magZcol = self.get_matching_columns(self.data,self.MAG_Z_COLUMNS)
            
            if len(magXcol)==1:
                self.data[magcol] = self.process_magnetometry_data(magXcol[0],magYcol[0],magZcol[0])
                print("Creating Total Magnetic Field field from single-sensor vector data")
            elif len(magXcol)==2:
                self.data[magcol] = self.process_dual_magnetometry_data(magXcol[0],magYcol[0],magZcol[0],
                                                                        magXcol[-1],magYcol[-1],magZcol[-1],
                                                                        baseline_m=WellDetective.DEFAULT_BASELINE_M)
                print("Creating Total Magnetic Field field from dual-sensor vector data")
            elif not magXcol:
                print('No Mag Data found')    
            else:
                print('More than 2 set of sensors found?')
    
        return self
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Interpreted Variables (Level: Data)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_heading_column_old(self, lat_col="Lat", lon_col="Long", window=20):
        """
        Compute and add a 'Heading' column based on coordinate differences.

        Parameters
        ----------
        lat_col, lon_col : str
            Column names for latitude and longitude.
        window : int
            Look-back window size for computing headings.

        Returns
        -------
        WellDetective
            Self for method chaining
        """
        
        # Check if required columns exist
        if lat_col not in self.data.columns:
            raise ValueError(f"Column '{lat_col}' not found in data. Available columns: {list(self.data.columns)}")
        if lon_col not in self.data.columns:
            raise ValueError(f"Column '{lon_col}' not found in data. Available columns: {list(self.data.columns)}")

        headings = [0] * window
        for i in range(window, len(self.data)):
            lat1, lon1 = self.data.iloc[i - window][[lat_col, lon_col]]
            lat2, lon2 = self.data.iloc[i][[lat_col, lon_col]]
            heading = self.calculate_heading(lat1, lon1, lat2, lon2)
            headings.append(heading)
        self.data["Heading"] = headings
        
        return self

    def add_heading_column(self, lat_col="Lat", lon_col="Long", window=20):
        """
        Compute and add a 'Heading' column based on coordinate differences.

        Parameters
        ----------
        lat_col, lon_col : str
            Column names for latitude and longitude.
        window : int
            Look-back window size for computing headings.

        Returns
        -------
        WellDetective
            Self for method chaining
        """
        
        # Check if required columns exist
        if lat_col not in self.data.columns:
            raise ValueError(f"Column '{lat_col}' not found in data. Available columns: {list(self.data.columns)}")
        if lon_col not in self.data.columns:
            raise ValueError(f"Column '{lon_col}' not found in data. Available columns: {list(self.data.columns)}")

        # Vectorized implementation for speed
        lat = self.data[lat_col].values
        lon = self.data[lon_col].values
        
        # Shift arrays by window to get previous positions
        lat1 = np.concatenate([np.full(window, lat[0]), lat[:-window]])
        lon1 = np.concatenate([np.full(window, lon[0]), lon[:-window]])
        lat2 = lat
        lon2 = lon
        
        # Vectorized heading calculation
        delta_lon = np.radians(lon2 - lon1)
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        
        x = np.sin(delta_lon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
        
        initial_heading = np.arctan2(x, y)
        heading = np.degrees(initial_heading)
        compass_heading = (heading + 360 + 90) % 360
        
        # Set first 'window' headings to 0 (or NaN if preferred)
        compass_heading[:window] = 0
        
        self.data["Heading"] = compass_heading

        #
        return self

    def process_dual_magnetometry_data(
        self,
        b1x_col: str,
        b1y_col: str,
        b1z_col: str,
        b2x_col: str,
        b2y_col: str,
        b2z_col: str,
        baseline_m: float = 0.5,
        add_gradient: bool = True,
        add_vector_diff: bool = False
    ):
        """
        Process dual magnetometer data to calculate total field, gradients, and differences.
        
        Args:
            self: DataFrame containing magnetometer data
            b1x_col: Column name for sensor 1 X component (e.g., 'B1x [nT]')
            b1y_col: Column name for sensor 1 Y component (e.g., 'B1y [nT]')
            b1z_col: Column name for sensor 1 Z component (e.g., 'B1z [nT]')
            b2x_col: Column name for sensor 2 X component (e.g., 'B2x [nT]')
            b2y_col: Column name for sensor 2 Y component (e.g., 'B2y [nT]')
            b2z_col: Column name for sensor 2 Z component (e.g., 'B2z [nT]')
            baseline_m: Distance between sensors in meters (default: 0.5)
            add_gradient: Whether to calculate gradient (default: True)
            add_vector_diff: Whether to calculate vector differences (default: True)
        
        Returns:
            WellDetective: Self for method chaining. Modifies self.data in-place by adding:
                - Total_B1: Total field magnitude from sensor 1
                - Total_B2: Total field magnitude from sensor 2
                - Total_Avg: Average total field from both sensors
                - Total_Diff: Difference between sensor 2 and sensor 1
                - Gradient_nT_m: Magnetic gradient (if add_gradient=True)
                - dBx, dBy, dBz: Vector differences (if add_vector_diff=True)
                - Vector_Diff_Mag: Magnitude of vector difference (if add_vector_diff=True)
        
        Raises:
            ValueError: If any specified column doesn't exist in the DataFrame
            TypeError: If self is not a pandas DataFrame
        
        Example:
            >>> df = process_magnetometry_data(
            ...     self,
            ...     b1x_col='B1x [nT]',
            ...     b1y_col='B1y [nT]',
            ...     b1z_col='B1z [nT]',
            ...     b2x_col='B2x [nT]',
            ...     b2y_col='B2y [nT]',
            ...     b2z_col='B2z [nT]',
            ...     baseline_m=0.5
            ... )
        """
        # Type checking
        # if not isinstance(self, pd.DataFrame):
        #     raise TypeError(f"df must be a pandas DataFrame, got {type(self).__name__}")
        
        # Validate that all required columns exist
        required_cols = [b1x_col, b1y_col, b1z_col, b2x_col, b2y_col, b2z_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(self.data.columns)}"
            )

        # Calculate total field for sensor 1
        self.data['Total_B1'] = np.sqrt(
            self.data[b1x_col]**2 + 
            self.data[b1y_col]**2 + 
            self.data[b1z_col]**2
        )
        
        # Calculate total field for sensor 2
        self.data['Total_B2'] = np.sqrt(
            self.data[b2x_col]**2 + 
            self.data[b2y_col]**2 + 
            self.data[b2z_col]**2
        )
        
        # Average total field (most common method)
        self.data['Total_Avg'] = (self.data['Total_B1'] + self.data['Total_B2']) / 2
        
        # Difference between sensors
        self.data['Total_Diff'] = self.data['Total_B2'] - self.data['Total_B1']
        
        # Calculate gradient if requested
        if add_gradient:
            self.data['Gradient_nT_m'] = self.data['Total_Diff'] / baseline_m
        
        # Calculate vector differences if requested
        if add_vector_diff:
            self.data['dBx'] = self.data[b2x_col] - self.data[b1x_col]
            self.data['dBy'] = self.data[b2y_col] - self.data[b1y_col]
            self.data['dBz'] = self.data[b2z_col] - self.data[b1z_col]
            
            # Vector difference magnitude
            self.data['Vector_Diff_Mag'] = np.sqrt(
                self.data['dBx']**2 + 
                self.data['dBy']**2 + 
                self.data['dBz']**2
            )
        
        return self
    
    def process_magnetometry_data(
        self,
        bx_col: str,
        by_col: str,
        bz_col: str
    ):
        """
        Process single magnetometer data to calculate total field.
        
        Args:
            self: DataFrame containing magnetometer data
            bx_col: Column name for X component
            by_col: Column name for Y component
            bz_col: Column name for Z component
        
        Returns:
            WellDetective: Self for method chaining. Modifies self.data in-place by adding 'Total_Field' column
        
        Raises:
            ValueError: If any specified column doesn't exist
            TypeError: If self is not a pandas DataFrame
        """
        # Type checking
        # if not isinstance(self, pd.DataFrame):
        #     raise TypeError(f"df must be a pandas DataFrame, got {type(self).__name__}")
        
        # Validate columns
        required_cols = [bx_col, by_col, bz_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(self.data.columns)}"
            )
        
        # Calculate total field
        self.data['Total_Field'] = np.sqrt(
            self.data[bx_col]**2 + 
            self.data[by_col]**2 + 
            self.data[bz_col]**2
        )
        
        return self    

        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Filtering dataset (Level: Data_Filtered)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def start_filtering(self):
        """Initialize filtered data level from current data."""
        self.data_filtered = self.data.copy()
        print(f"Initialized data_filtered with {len(self.data_filtered)} rows")
        return self     

    def find_primary_secondary_headings(self, tolerance=10, filename=None):
        """
        Identify primary and secondary heading ranges using K-Means clustering.
    
        Parameters
        ----------
        tolerance : float
            Degrees around the cluster centers for heading ranges.
        filename : str, optional
            Filename or substring to filter data by specific flight.
            If None, uses all data.
    
        Returns
        -------
        tuple
            Primary and secondary heading ranges as tuples (min, max).
        """
        
        # Check if Heading column exists
        if "Heading" not in self.data_filtered.columns:
            raise ValueError(
                "Heading column not found in data. "
                "Please run add_heading_column() before finding heading ranges."
            )
        
        # Filter by filename if provided
        if filename is not None:
            if 'Filepath' not in self.data_filtered.columns:
                raise ValueError("'Filepath' column not found in data.")
            file_mask = self.data_filtered['Filepath'].str.contains(filename, na=False)
            if file_mask.sum() == 0:
                raise ValueError(f"No data found matching filename: '{filename}'")
            headings = self.data_filtered.loc[file_mask, "Heading"].values.reshape(-1, 1)
            print(f"Finding headings for {file_mask.sum():,} points matching '{filename}'")
        else:
            headings = self.data_filtered["Heading"].values.reshape(-1, 1)
            print(f"Finding headings for all {len(headings):,} points")
    
        kmeans = KMeans(n_clusters=2, random_state=0).fit(headings)
        centers = sorted(kmeans.cluster_centers_.flatten())
    
        primary_range = (centers[0] - tolerance, centers[0] + tolerance)
        secondary_range = (centers[1] - tolerance, centers[1] + tolerance)
    
        return primary_range, secondary_range

    def remove_turning_data(self, primary_range, secondary_range):
        """
        Remove turning data based on heading ranges.

        Parameters
        ----------
        primary_range, secondary_range : tuple
            Tuples representing primary and secondary heading ranges.

        Returns
        -------
        WellDetective
            Self for method chaining
        """
        # Check if data_filtered is created
        if self.data_filtered is None:
            raise ValueError("Call start_filtering() first")
            
         # Check if Heading column exists
        if "Heading" not in self.data_filtered.columns:
            raise ValueError(
              "Heading column not found in data. "
              "Please run add_heading_column() before removing turning data."
            )
        

        mask = ((self.data_filtered["Heading"].between(*primary_range)) |
                (self.data_filtered["Heading"].between(*secondary_range)))

        if mask.sum() == 0:
            raise ValueError(
                f"No data points fall within the specified heading ranges. "
                f"Primary range: {primary_range}, Secondary range: {secondary_range}. "
                f"All {len(self.data_filtered)} data points would be removed."
            )

        self.data_filtered = self.data_filtered[mask].reset_index(drop=True)

        #
        return self
        
    def segment_and_filter_data(self, max_gap_distance=10, min_segment_length=150, lat_col="Lat", lon_col="Long"):
        """
        Segment data into continuous lines and remove segments shorter than a specified minimum length.
    
        Parameters
        ----------
        max_gap_distance : float, optional
            Maximum allowed gap distance between consecutive points (in meters).
            A gap larger than this starts a new segment.
        min_segment_length : float, optional
            Minimum length of segments to retain (in meters). Segments shorter
            than this will be discarded.
        lat_col : str, optional
            Name of the latitude column.
        lon_col : str, optional
            Name of the longitude column.
    
        Returns
        -------
        WellDetective
            Self for method chaining. Updates self.data_filtered in-place.
        """
        def compute_distances_vectorized(lat, lon):
            R = 6371000.0  # Earth radius in meters

            # Get arrays for current and next points
            lat1, lon1 = lat[:-1].values, lon[:-1].values
            lat2, lon2 = lat[1:].values, lon[1:].values

            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)

            a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c
        
        # Check if data_filtered is created
        if self.data_filtered is None:
            raise ValueError("Call start_filtering() first")
            
        # Calculate ALL distances at once
        distances = compute_distances_vectorized(self.data_filtered[lat_col], self.data_filtered[lon_col])

        segmentBreaks = np.where(distances > max_gap_distance)[0]

        # Group data into segments
        segmentRanges = []
        startIdx = 0
    
        for breakIdx in segmentBreaks:
            segmentRanges.append((startIdx, breakIdx + 1))  # +1 because break is between [breakIdx] and [breakIdx+1]
            startIdx = breakIdx + 1
    
        if startIdx < len(self.data_filtered):
            segmentRanges.append((startIdx, len(self.data_filtered)))
    
        # Calculate and filter by segment lengths
        filtered_segments = []
        for start,end in segmentRanges:
            if end - start > 1:
                segmentDistances = distances[start:end-1]
                totalDistance = segmentDistances.sum()

                if totalDistance >= min_segment_length:
                    filtered_segments.append(self.data_filtered.iloc[start:end])
    
        # Update self.data with filtered segments
        if filtered_segments:
            self.data_filtered = pd.concat(filtered_segments).reset_index(drop=True)
        else:
            self.data_filtered = pd.DataFrame(columns=self.data_filtered.columns)

        #
        return self


    def auto_normalize_heading_correction(self, primary_range, secondary_range, mag_col="Mag",method=0):
        """
        Apply heading correction to equalize magnetic field measurements.

        Parameters
        ----------
        primary_range, secondary_range : tuple
            Heading ranges for primary and secondary directions.
        mag_col : str
            Column name of magnetic field measurements.
        method : int
            Determine whether filtering by mean, median, or other method
            0 = mean (default)
            1 = median
            2 = gauss fit
            3 = ...

        Returns
        -------
        WellDetective
            Self for method chaining

        Notes
        -----
        Method 2 (Gaussian fit) is more robust to outliers and skewed distributions
        compared to mean or median. It finds the peak of the distribution by fitting
        a Gaussian curve to the histogram of magnetic field values.
        """
        def gaussian(x, amplitude, mean, stddev):
                """Gaussian function for curve fitting."""
                return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    
        def fit_gaussian_peak(data, bins=50):
            """
            Fit Gaussian to histogram and return the peak center.
            
            Parameters
            ----------
            data : pd.Series
                Magnetic field data
            bins : int
                Number of histogram bins
            
            Returns
            -------
            float
                Center of Gaussian peak (mean parameter)
            """
            # Create histogram
            hist, bin_edges = np.histogram(data.dropna(), bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Initial guesses for Gaussian parameters
            amplitude_guess = hist.max()
            mean_guess = bin_centers[np.argmax(hist)]
            stddev_guess = data.std()
            
            try:
                # Fit Gaussian
                params, _ = curve_fit(
                    gaussian, 
                    bin_centers, 
                    hist,
                    p0=[amplitude_guess, mean_guess, stddev_guess],
                    maxfev=5000
                )
                
                # Return the mean (center) of the fitted Gaussian
                return params[1]
                
            except Exception as e:
                print(f"Warning: Gaussian fit failed ({e}). Falling back to mean.")
                return data.mean()

        # Extract data for each heading range
        primary_data = self.data_filtered.loc[
            self.data_filtered["Heading"].between(*primary_range), mag_col
        ]
        secondary_data = self.data_filtered.loc[
            self.data_filtered["Heading"].between(*secondary_range), mag_col
        ]
        
         # Calculate background/central value based on method
        if method == 0:
            # Mean
            primary_bkgd = primary_data.mean()
            secondary_bkgd = secondary_data.mean()
            print(f"Method: Mean")
            
        elif method == 1:
            # Median
            primary_bkgd = primary_data.median()
            secondary_bkgd = secondary_data.median()
            print(f"Method: Median")
            
        elif method == 2:
            # Gaussian fit
            print(f"Method: Gaussian Fit")
            primary_bkgd = fit_gaussian_peak(primary_data, bins=50)
            secondary_bkgd = fit_gaussian_peak(secondary_data, bins=50)
            print(f"  Primary Gaussian center: {primary_bkgd:.2f} nT")
            print(f"  Secondary Gaussian center: {secondary_bkgd:.2f} nT")
            
        else:
            raise ValueError(f"Invalid method: {method}. Must be 0 (mean), 1 (median), or 2 (Gaussian fit)") 

        # Apply correction - subtract background from each heading's data
        idx_primary = self.data_filtered["Heading"].between(*primary_range)
        idx_secondary = self.data_filtered["Heading"].between(*secondary_range)
        
        # Create Corrected column (copy first to preserve original mag_col)
        self.data_filtered["Corrected"] = self.data_filtered[mag_col].copy()
        
        # Apply corrections to the Corrected column only
        self.data_filtered.loc[idx_primary, "Corrected"] -= primary_bkgd
        self.data_filtered.loc[idx_secondary, "Corrected"] -= secondary_bkgd

        return self

    def project_coordinates(self, utm_zone=12, lat_col="Lat", lon_col="Long"): 
        """
        Project geographic coordinates to UTM.

        Parameters
        ----------
        utm_zone : int
            UTM zone for projection.

        Returns
        -------
        WellDetective
            Self for method chaining

        Update: JEL
        """
        projection = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84", preserve_units=False)
        easting, northing = projection(self.data_filtered[lon_col].values, self.data_filtered[lat_col].values)
        self.data_filtered["easting"] = easting
        self.data_filtered["northing"] = northing
        
        #
        return self

        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Interpolating 2-D maps (Level: .map)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_survey_utm_zone(self, lat_col='Lat', lon_col='Long'):
        """
        Determine the UTM zone for the survey based on data coordinates.
        Uses the centroid of all coordinates.
        
        Parameters
        ----------
        lat_col : str
            Column name for latitude (default: 'Lat')
        lon_col : str
            Column name for longitude (default: 'Long')
        
        Returns
        -------
        tuple : (zone_number, hemisphere, epsg_code)
            zone_number: int (1-60)
            hemisphere: str ('N' or 'S')
            epsg_code: int (EPSG code for the UTM zone)
        
        Example
        -------
        >>> wd = WellDetective(data)
        >>> zone, hemi, epsg = wd.get_survey_utm_zone()
        >>> print(f"Survey is in UTM Zone {zone}{hemi} (EPSG:{epsg})")
        """
        # Use data_filtered if available, otherwise use data
        df = self.data_filtered if self.data_filtered is not None else self.data
        
        # Check if columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in data")
        
        # Calculate centroid
        center_lon = df[lon_col].mean()
        center_lat = df[lat_col].mean()
        
        return self.get_utm_zone_from_lon_lat(center_lon, center_lat)

        
    def grid_and_filter_data(self, inclination, declination, 
                             grid_size=50, cutoff_wavelength=30, proximity_threshold=40,
                             use_fast_interpolation=True
                            ):
        """
        Interpolate, reduce-to-pole, and low-pass filter magnetic data.
        Subtracts mean magnetic field before processing and masks areas
        too far from the flight path.
    
        Parameters
        ----------
        inclination, declination : float
            IGRF inclination and declination values.
        grid_size : float
            Grid cell size in meters (default: 50). Smaller values = finer resolution.
        cutoff_wavelength : float
            Wavelength for low-pass filter (meters).
        proximity_threshold : float
            Maximum distance (in meters) from flight line to keep grid points.
        use_fast_interpolation : bool, optional
            Use optimized LinearNDInterpolator (default: True).
            Much faster for large datasets (>100k points). 3-5x speedup.
    
        Returns
        -------
        xarray.DataArray
            Processed and masked magnetic data grid.
        """
        import time
        
        # Subtract mean
        mag_mean = self.data_filtered['Corrected'].mean()
        corrected = self.data_filtered['Corrected'] - mag_mean

        # Calculate number of grid points from grid size
        easting_range = self.data_filtered["easting"].max() - self.data_filtered["easting"].min()
        northing_range = self.data_filtered["northing"].max() - self.data_filtered["northing"].min()
        
        n_points_x = int(np.ceil(easting_range / grid_size))
        n_points_y = int(np.ceil(northing_range / grid_size))
        
        print(f"Creating grid: {n_points_x} x {n_points_y} = {n_points_x*n_points_y:,} points")
        
        # Create grid
        xi, yi = np.mgrid[
            self.data_filtered["easting"].min():self.data_filtered["easting"].max():n_points_x*1j,
            self.data_filtered["northing"].min():self.data_filtered["northing"].max():n_points_y*1j
        ]
        
        # Interpolate to grid (OPTIMIZED)
        start = time.time()
        print(f"Interpolating {len(corrected):,} data points to grid...")
        
        if use_fast_interpolation:
            from scipy.interpolate import LinearNDInterpolator
            
            # Pre-compute triangulation (done once, reused for interpolation)
            interp = LinearNDInterpolator(
                np.c_[self.data_filtered["easting"].values, 
                      self.data_filtered["northing"].values],
                corrected.values,
                fill_value=np.nan
            )
            
            # Interpolate (reuses triangulation - much faster)
            zi = interp(xi, yi)
            
        else:
            # Original method (slower but proven)
            zi = griddata(
                (self.data_filtered["easting"], self.data_filtered["northing"]),
                corrected,
                (xi, yi),
                method="linear"
            )
        
        elapsed = time.time() - start
        print(f"✓ Interpolation completed in {elapsed:.1f} seconds")
    
        # Create xarray
        da = xr.DataArray(zi, dims=["x", "y"], coords={"x": xi[:, 0], "y": yi[0, :]})
    
        # Pad and reduce to pole (requires harmonica)
        if not HAS_HARMONICA:
            raise ImportError(
                "harmonica is required for reduction to pole and filtering. "
                "Install it with: pip install harmonica (requires LLVM/cmake)"
            )
        
        pad_width = {"x": da.x.size // 3, "y": da.y.size // 3}
        padded = xrft.pad(da, pad_width=pad_width).fillna(0.0)
        rtp = hm.reduction_to_pole(padded, inclination, declination)
        rtp_unpadded = xrft.unpad(rtp, pad_width)
    
        # Apply low-pass filter
        filtered = hm.gaussian_lowpass(rtp_unpadded, wavelength=cutoff_wavelength)
    
        # Build KDTree from original easting/northing
        flight_tree = cKDTree(np.c_[self.data_filtered["easting"], self.data_filtered["northing"]])
        grid_points = np.c_[xi.flatten(), yi.flatten()]
        distances, _ = flight_tree.query(grid_points)
        distance_grid = distances.reshape(xi.shape)
    
        # Mask out points beyond the threshold
        mask = distance_grid <= proximity_threshold
        low_passed = filtered.where(mask)
    
        return low_passed

    def create_spatial_map(self, grid_size=50, cutoff_wavelength=30, proximity_threshold=40,
                          use_pyigrf=True, inclination=None, declination=None, date=None, v_plotdata=False):
        """
        Create spatially interpolated map of filtered magnetometry data.
        
        Parameters
        ----------
        grid_size : float, optional
            Grid cell size in meters (default: 50). Smaller values = finer resolution.
        cutoff_wavelength : float, optional
            Wavelength for low-pass filter in meters (default: 30)
        proximity_threshold : float, optional
            Maximum distance from flight line in meters (default: 40)
        use_pyigrf : bool, optional
            Use pyIGRF to calculate inclination/declination (default: True)
        inclination : float, optional
            Manual inclination value in degrees (used if use_pyigrf=False)
        declination : float, optional
            Manual declination value in degrees (used if use_pyigrf=False)
        date : datetime, optional
            Date for IGRF calculation (uses first data point's date if None)
        v_plotdata : bool, optional
            Generate heatmap plot (default: False)
        
        Returns
        -------
        WellDetective
            Self for method chaining. Creates self.map attribute with grid data.
        
        Raises
        ------
        ValueError
            If data_filtered is empty or coordinates not projected
        ImportError
            If pyIGRF not installed and use_pyigrf=True

        Example Usage
        -------------
        combined.create_spatial_map(grid_size=50, cutoff_wavelength=30, proximity_threshold=40,
                          use_pyigrf=True, inclination=None, declination=None, date=None)
        """
        # Check prerequisites
        if self.data_filtered is None or len(self.data_filtered) == 0:
            raise ValueError("data_filtered is empty. Process data first.")
        
        if 'easting' not in self.data_filtered.columns or 'northing' not in self.data_filtered.columns:
            raise ValueError("Coordinates not projected. Run project_coordinates() first.")
        
        if 'Corrected' not in self.data_filtered.columns:
            raise ValueError("'Corrected' column missing. Run auto_equalize_heading_correction() first.")
        
        # Get inclination and declination
        if use_pyigrf:
            if not HAS_PYIGRF:
                raise ImportError("pyIGRF not installed. Install with: pip install pyIGRF")
            
            # Get representative point from first row
            field_info = self.data_filtered.iloc[0]
            
            # Find lat/lon columns using column matching
            LatCol = self.get_matching_columns(self.data_filtered, self.LAT_COLUMNS)
            LonCol = self.get_matching_columns(self.data_filtered, self.LON_COLUMNS)
            
            if not LatCol or not LonCol:
                raise ValueError(f"Latitude/Longitude columns not found. Expected lat: {self.LAT_COLUMNS}, lon: {self.LON_COLUMNS}")
            
            # Determine date for IGRF
            if date is None:
                # Use get_matching_columns to find date column (handles various names)
                DateCol = self.get_matching_columns(self.data_filtered, self.DATE_COLUMNS)
                if DateCol:
                    date = pd.to_datetime(self.data_filtered.iloc[0][DateCol[0]])
                else:
                    raise ValueError("No date provided and no date column found in data. Expected columns: " + str(self.DATE_COLUMNS))
            
            # Calculate IGRF values
            decimal_year = WellDetective.to_decimal_year(date)
            igrf_results = pyIGRF.igrf_value(
                field_info[LatCol[0]], 
                field_info[LonCol[0]], 
                alt=0, 
                year=decimal_year
            )
            inclination, declination = igrf_results[1], igrf_results[0]
            
            print(f"IGRF values for {date.date()}:")
            print(f"  Inclination: {inclination:.2f}°")
            print(f"  Declination: {declination:.2f}°")
        
        else:
            # Use manual values
            if inclination is None or declination is None:
                raise ValueError("Must provide inclination and declination if use_pyigrf=False")
            
            print(f"Using manual IGRF values:")
            print(f"  Inclination: {inclination:.2f}°")
            print(f"  Declination: {declination:.2f}°")
        
        # Grid and filter data
        print(f"\nCreating spatial map with parameters:")
        print(f"  Grid cell size: {grid_size} m")
        print(f"  Cutoff wavelength: {cutoff_wavelength} m")
        print(f"  Proximity threshold: {proximity_threshold} m")
        
        mag_grid_da = self.grid_and_filter_data(
            inclination, 
            declination, 
            grid_size=grid_size, 
            cutoff_wavelength=cutoff_wavelength, 
            proximity_threshold=proximity_threshold
        )
        
        # Extract numpy arrays for further processing
        grid_x, grid_y = np.meshgrid(mag_grid_da.x.values, mag_grid_da.y.values, indexing='ij')
        mag_grid = mag_grid_da.values
        
        # Store as attribute
        self.map = {
            'data_array': mag_grid_da,      # xarray DataArray (original)
            'grid_x': grid_x,                # 2D numpy array of x coordinates
            'grid_y': grid_y,                # 2D numpy array of y coordinates
            'mag_grid': mag_grid,            # 2D numpy array of magnetic values
            'inclination': inclination,      # IGRF inclination used
            'declination': declination,      # IGRF declination used
            'grid_size': grid_size,          # Grid cell size in meters
            'cutoff_wavelength': cutoff_wavelength,
            'proximity_threshold': proximity_threshold,
            'created_at': pd.Timestamp.now()
        }
        
        print(f"\n✓ Spatial map created:")
        print(f"  Grid shape: {mag_grid.shape}")
        print(f"  Grid cell size: {grid_size} m")
        print(f"  X range: [{grid_x.min():.1f}, {grid_x.max():.1f}] m")
        print(f"  Y range: [{grid_y.min():.1f}, {grid_y.max():.1f}] m")
        print(f"  Mag range: [{np.nanmin(mag_grid):.2f}, {np.nanmax(mag_grid):.2f}] nT")
        print(f"  Valid cells: {np.sum(~np.isnan(mag_grid))}/{mag_grid.size}")
        
        # Add to processing log
        self.processing_log.append({
            'operation': 'create_spatial_map',
            'timestamp': pd.Timestamp.now(),
            'parameters': {
                'grid_size': grid_size,
                'cutoff_wavelength': cutoff_wavelength,
                'proximity_threshold': proximity_threshold,
                'inclination': inclination,
                'declination': declination
            }
        })

        #
        
        if v_plotdata:
            self.plot_Mag_Heat(E_incr=WellDetective.DEFAULT_MAP_XRES, N_incr=WellDetective.DEFAULT_MAP_YRES, figsize=(12, 10), save_path=None)
            
        return self
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Detection
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def detect_hotspots(self, mag_grid, grid_x, grid_y, well_coords, distance_threshold=100, bandwidth=80):
        """
        Detect hotspots and flag potential orphan wells.
    
        Parameters
        ----------
        mag_grid : np.array
            2D magnetic data grid.
        grid_x, grid_y : np.array
            Coordinate grids.
        well_coords : list of tuple
            Known well coordinates in projected units.
        distance_threshold : float
            Minimum distance from a hotspot to known wells to be flagged as orphan.
        bandwidth : float
            MeanShift clustering bandwidth.
    
        Returns
        -------
        tuple of pd.DataFrame
            - cluster centroids
            - orphan wells
        """
    
        # Compute threshold for hotspot detection
        if np.isnan(mag_grid).all():
            print("Magnetic grid is fully NaN. No hotspots detected.")
            return pd.DataFrame(), pd.DataFrame()
    
        threshold = np.nanmean(mag_grid) + np.nanstd(mag_grid)
        peak_indices = np.argwhere(mag_grid > threshold)
    
        if peak_indices.size == 0:
            print("No high points detected in the magnetic grid.")
            return pd.DataFrame(), pd.DataFrame()
    
        # Extract coordinates and values for peaks
        points = pd.DataFrame({
            "easting": grid_x[peak_indices[:, 0], peak_indices[:, 1]],
            "northing": grid_y[peak_indices[:, 0], peak_indices[:, 1]],
            "value": mag_grid[peak_indices[:, 0], peak_indices[:, 1]],
        })
    
        # Remove rows with NaNs (this is critical)
        points = points.dropna(subset=["easting", "northing"])
    
        if points.empty:
            print("All hotspot candidates were NaN after masking. No hotspots detected.")
            return pd.DataFrame(), pd.DataFrame()
    
        # Perform MeanShift clustering
        clustering = MeanShift(bandwidth=bandwidth).fit(points[["easting", "northing"]])
        points["cluster"] = clustering.labels_
    
        # Compute centroids
        centroids = points.groupby("cluster").mean().reset_index()
    
        # Find orphan wells: clusters far from all known wells
        orphan_wells = centroids[
            centroids.apply(
                lambda row: min(distance.cdist([(row["easting"], row["northing"])], well_coords)[0]) > distance_threshold,
                axis=1
            )
        ]
    
        return centroids, orphan_wells


   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Exporting Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def export_map_to_geotiff(self, output_path, crs=None):
        """
        Export gridded magnetic data to GeoTIFF for QGIS.
        
        Parameters
        ----------
        output_path : str
            Path to output .tif file
        crs : str, optional
            Coordinate reference system (e.g., 'EPSG:32612' for UTM Zone 12N).
            If None, auto-detects from data coordinates (default: None)

        Example Usage
        -------
        # Auto-detect UTM zone
        wd.export_map_to_geotiff('magnetic_grid.tif')
        
        # Or specify manually
        wd.export_map_to_geotiff('magnetic_grid.tif', crs='EPSG:32614')
        
        """
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio not installed. Install with:\n"
                "  conda install -c conda-forge rasterio"
            )
        
        if self.map is None:
            raise ValueError("Map not created. Run create_spatial_map() first.")
        
        # Auto-detect UTM zone if CRS not provided
        if crs is None:
            # Get lat/lon columns from filtered data
            LatCol = self.get_matching_columns(self.data_filtered, self.LAT_COLUMNS)
            LonCol = self.get_matching_columns(self.data_filtered, self.LON_COLUMNS)
            
            if not LatCol or not LonCol:
                raise ValueError("Cannot auto-detect UTM zone: Lat/Lon columns not found. Please specify 'crs' parameter.")
            
            # Get UTM zone from data
            zone, hemi, epsg = self.get_survey_utm_zone(lat_col=LatCol[0], lon_col=LonCol[0])
            crs = f'EPSG:{epsg}'
            print(f"Auto-detected CRS: {crs} (UTM Zone {zone}{hemi})")
        
        if self.map is None:
            raise ValueError("Map not created. Run create_spatial_map() first.")
        
        mag_grid = self.map['mag_grid']
        grid_x = self.map['grid_x']
        grid_y = self.map['grid_y']
        
        # Convert EPSG to WKT if needed (workaround for PROJ database issues)
        crs_to_use = crs
        if isinstance(crs, str) and crs.upper().startswith('EPSG:'):
            try:
                # Try to use EPSG code directly
                from rasterio.crs import CRS
                test_crs = CRS.from_string(crs)
                # If we get here without error, EPSG works
                crs_to_use = crs
            except Exception as e:
                # EPSG failed, try WKT fallback for common UTM zones
                if 'PROJ' in str(e) or 'database' in str(e).lower():
                    epsg_code = int(crs.split(':')[1])
                    # UTM zones: 326xx for Northern, 327xx for Southern
                    if 32601 <= epsg_code <= 32660:  # UTM North
                        zone = epsg_code - 32600
                        crs_to_use = self._get_utm_wkt(zone, northern=True)
                        print(f"⚠ PROJ database issue detected, using WKT for UTM Zone {zone}N")
                    elif 32701 <= epsg_code <= 32760:  # UTM South
                        zone = epsg_code - 32700
                        crs_to_use = self._get_utm_wkt(zone, northern=False)
                        print(f"⚠ PROJ database issue detected, using WKT for UTM Zone {zone}S")
                    else:
                        # For other EPSG codes, re-raise the error
                        raise
        
        # Prepare array for GeoTIFF:
        # 1. Transpose: matplotlib/numpy use (x, y) indexing but GeoTIFF uses (y, x)
        # 2. Flip vertically: GeoTIFF expects first row = northernmost (top)
        mag_grid_transposed = mag_grid.T
        mag_grid_geotiff = np.flipud(mag_grid_transposed)
        
        # Calculate actual pixel sizes
        pixel_size_x = (grid_x.max() - grid_x.min()) / mag_grid_geotiff.shape[1]
        pixel_size_y = (grid_y.max() - grid_y.min()) / mag_grid_geotiff.shape[0]
        
        # Define transform AFTER transpose (maps pixel coords to real-world coords)
        # from_bounds(west, south, east, north, width, height)
        transform = from_bounds(
            grid_x.min(), grid_y.min(),
            grid_x.max(), grid_y.max(),
            mag_grid_geotiff.shape[1],  # width (number of columns in final array)
            mag_grid_geotiff.shape[0]   # height (number of rows in final array)
        )
        
        print(f"GeoTIFF export info:")
        print(f"  Array shape: {mag_grid_geotiff.shape} (rows, cols)")
        print(f"  Pixel size: {pixel_size_x:.2f}m × {pixel_size_y:.2f}m (x, y)")
        print(f"  Bounds: X=[{grid_x.min():.1f}, {grid_x.max():.1f}], Y=[{grid_y.min():.1f}, {grid_y.max():.1f}]")
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=mag_grid_geotiff.shape[0],  # rows (y)
            width=mag_grid_geotiff.shape[1],   # cols (x)
            count=1,
            dtype=mag_grid_geotiff.dtype,
            crs=crs_to_use,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(mag_grid_geotiff, 1)
            
            # Add metadata
            dst.update_tags(
                inclination=self.map['inclination'],
                declination=self.map['declination'],
                created_at=str(self.map['created_at']),
                units='nT'
            )
        
        print(f"✓ Exported to GeoTIFF: {output_path}")

    def export_to_netcdf(self, output_path, include_raw=False, include_data=False, 
                         include_filtered=True, include_map=True):
        """
        Export WellDetective object to NetCDF file.
        
        Parameters
        ----------
        output_path : str
            Path to output .nc file
        include_raw : bool, optional
            Include data_raw (default: True)
        include_data : bool, optional
            Include data (default: True)
        include_filtered : bool, optional
            Include data_filtered (default: True)
        include_map : bool, optional
            Include spatial map if available (default: True)
        
        Returns
        -------
        None
            Saves to NetCDF file
        
        Notes
        -----
        NetCDF file will contain multiple groups:
        - /raw : data_raw
        - /processed : data
        - /filtered : data_filtered
        - /map : spatial grid (if available)
        - /metadata : processing log and parameters

        Example Usage
        -------------
        # Example 1: Single file with everything
        combined.export_to_netcdf(
            'magnetometry_complete.nc',
            include_raw=True,
            include_data=True,
            include_filtered=True,
            include_map=True
        )

        """
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create main dataset with metadata
        ds_dict = {}
        
        # Add data_raw
        if include_raw and self.data_raw is not None:
            ds_dict['raw'] = xr.Dataset.from_dataframe(
                self.data_raw.reset_index(drop=True)
            )
            ds_dict['raw'].attrs['description'] = 'Raw unprocessed magnetometry data'
            ds_dict['raw'].attrs['n_rows'] = len(self.data_raw)
        
        # Add data (processed)
        if include_data and self.data is not None:
            ds_dict['processed'] = xr.Dataset.from_dataframe(
                self.data.reset_index(drop=True)
            )
            ds_dict['processed'].attrs['description'] = 'Processed data with heading and corrections'
            ds_dict['processed'].attrs['n_rows'] = len(self.data)
        
        # Add data_filtered
        if include_filtered and self.data_filtered is not None:
            ds_dict['filtered'] = xr.Dataset.from_dataframe(
                self.data_filtered.reset_index(drop=True)
            )
            ds_dict['filtered'].attrs['description'] = 'Filtered data with turns removed'
            ds_dict['filtered'].attrs['n_rows'] = len(self.data_filtered)
        
        # Add spatial map if available
        if include_map and hasattr(self, 'map') and self.map is not None:
            # Use the xarray DataArray directly
            ds_dict['map'] = self.map['data_array'].to_dataset(name='magnetic_anomaly')
            
            # Add map metadata
            ds_dict['map'].attrs['description'] = 'Gridded magnetic anomaly map'
            ds_dict['map'].attrs['inclination'] = self.map['inclination']
            ds_dict['map'].attrs['declination'] = self.map['declination']
            ds_dict['map'].attrs['grid_cell_size_meters'] = self.map['grid_size']
            ds_dict['map'].attrs['cutoff_wavelength'] = self.map['cutoff_wavelength']
            ds_dict['map'].attrs['proximity_threshold'] = self.map['proximity_threshold']
        
        # Save each group separately (NetCDF groups)
        # Note: xarray doesn't directly support hierarchical groups in to_netcdf()
        # So we'll save as separate variables with prefixes
        
        combined_ds = xr.Dataset()
        
        for group_name, ds in ds_dict.items():
            # Prefix variable names with group name
            for var_name in ds.data_vars:
                new_var_name = f"{group_name}__{var_name}"
                combined_ds[new_var_name] = ds[var_name]
            
            # Store group attributes
            for attr_name, attr_value in ds.attrs.items():
                combined_ds.attrs[f"{group_name}_{attr_name}"] = attr_value
        
        # Add global metadata
        combined_ds.attrs['created_by'] = 'WellDetective'
        combined_ds.attrs['created_at'] = str(pd.Timestamp.now())
        combined_ds.attrs['version'] = '1.0'
        
        # Add processing log as global attribute (as JSON string)
        if hasattr(self, 'processing_log') and self.processing_log:
            import json
            combined_ds.attrs['processing_log'] = json.dumps(
                self.processing_log, default=str, indent=2
            )
        
        # Save to NetCDF
        combined_ds.to_netcdf(output_path)
        
        print(f"✓ Exported WellDetective object to NetCDF: {output_path}")
        print(f"  Groups included:")
        for group_name in ds_dict.keys():
            print(f"    - {group_name}")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figures (Level: N/A)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_flight_tracks(self, E_incr=500, N_incr=500, figsize=(12, 10), save_path=None, plot_every_nth=100):
        """
        Create scatter plot of flight tracks color-coded by filename.
        
        Parameters
        ----------
        E_incr : int
            Increments for x_ticks (meters)
        N_incr : int
            Increments for y_ticks (meters)
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (12, 10))
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        plot_every_nth : int, optional
            Plot every Nth data point for faster rendering (default: 100).
            Set to 1 to plot all points (slowest), or higher for faster plotting.
            For 1.5M points: nth=100 plots 15k points, nth=10 plots 150k points.
        
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        
        if self.data_filtered is None or len(self.data_filtered) == 0:
            raise ValueError("No filtered data available. Process data first.")
        
        if 'easting' not in self.data_filtered.columns or 'northing' not in self.data_filtered.columns:
            raise ValueError("Coordinates not projected. Run project_coordinates() first.")
        
        if 'Filepath' not in self.data_filtered.columns:
            raise ValueError("'Filepath' column not found in data.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique files
        unique_files = self.data_filtered['Filepath'].unique()
        n_files = len(unique_files)
        
        print(f"Plotting {n_files} flight tracks...")
        
        # Get colors from colormap
        cmap = plt.cm.get_cmap('tab10' if n_files <= 10 else 'tab20')
        colors = [cmap(i / n_files) for i in range(n_files)]
        
        # Plot each flight track
        for i, (filepath, color) in enumerate(zip(unique_files, colors)):
            # Get data for this file
            mask = self.data_filtered['Filepath'] == filepath
            data_subset = self.data_filtered[mask]
            
            # Decimate data for faster plotting (plot every Nth point)
            data_subset = data_subset.iloc[::plot_every_nth]
            
            # Extract filename from path
            filename = Path(filepath).name
            
            # Plot using plot() instead of scatter() for much better performance
            ax.plot(
                data_subset['easting'],
                data_subset['northing'],
                ',',  # Pixel marker (fastest)
                color=color,
                label=filename,
                alpha=0.6,
                markersize=1
            )
            
            print(f"  {i+1}/{n_files}: {filename} ({len(data_subset):,} points plotted, every {plot_every_nth}th point)")

        # Set Axis Properties
        E_buffer = E_incr/5
        N_buffer = N_incr/5
        ax.set_xlim(self.data_filtered['easting'].min()-E_buffer,self.data_filtered['easting'].max()+E_buffer)
        ax.set_ylim(self.data_filtered['northing'].min()-N_buffer,self.data_filtered['northing'].max()+N_buffer)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        
        # Apply formatters (pass ax explicitly)
        ax.xaxis.set_major_formatter(FuncFormatter(WellDetective.make_utm_easting_formatter(target_ax=ax)))
        ax.yaxis.set_major_formatter(FuncFormatter(WellDetective.make_utm_northing_formatter(target_ax=ax)))
        
        # Set tick spacing (try 500m or 1000m)
        ax.xaxis.set_major_locator(MultipleLocator(E_incr))  # 500m increments
        ax.yaxis.set_major_locator(MultipleLocator(N_incr))
        
        # Labels and formatting
        ax.set_xlabel('Easting (m)', fontsize=12)
        ax.set_ylabel('Northing (m)', fontsize=12)
        ax.set_title('Flight Tracks Color-Coded by File', fontsize=14, fontweight='bold')
        
        # Legend
        if n_files <= 10:
            ax.legend(loc='best', fontsize=8, markerscale=5)
        else:
            # Too many files - put legend outside
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=8, markerscale=5)
            plt.tight_layout()

        # Annotations
        LatCol = self.get_matching_columns(self.data_filtered, self.LAT_COLUMNS)
        LonCol = self.get_matching_columns(self.data_filtered, self.LON_COLUMNS)
        zone, hemisphere, epsg = self.get_survey_utm_zone(lat_col=LatCol[0],lon_col=LonCol[0])
        fig.text(0.02, 0.00, f'Mercator Projection\nWGS84\nUTM Zone: {zone}{hemisphere}', 
                    horizontalalignment='left', fontsize=6)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to: {save_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        return fig, ax
        
    def plot_Mag_Heat(self, E_incr=500, N_incr=500, figsize=(12, 10), save_path=None):
        """
        Create scatter plot of flight tracks color-coded by filename.
        
        Parameters
        ----------
        E_incr : (int) increments for x_ticks
        N_incr : (int) increments for y_ticks
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (12, 10))
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        
        if self.data_filtered is None or len(self.data_filtered) == 0:
            raise ValueError("No filtered data available. Process data first.")
        
        if 'easting' not in self.data_filtered.columns or 'northing' not in self.data_filtered.columns:
            raise ValueError("Coordinates not projected. Run project_coordinates() first.")
        
        if 'Filepath' not in self.data_filtered.columns:
            raise ValueError("'Filepath' column not found in data.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        # Plot final results
        cf = ax.contourf(self.map['grid_x'], self.map['grid_y'], self.map['mag_grid'], 100, cmap="turbo")
        l_trax = ax.plot(self.data_filtered['easting'], self.data_filtered['northing'], color='k', lw=0.5, label="Flight Track")
        
        # Define Axis properties
        E_buffer = E_incr/5
        N_buffer = N_incr/5
        ax.set_xlim(self.map['grid_x'].min()-E_buffer,self.map['grid_x'].max()+E_buffer)
        ax.set_ylim(self.map['grid_y'].min()-N_buffer,self.map['grid_y'].max()+N_buffer)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        # add Colorbar
        # Create colorbar with proper sizing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cf, cax=cax, fraction=0.046, pad=0.04, aspect=30)
        cbar.set_label("Magnetic Field (nT)", fontsize=12)
        
        # Apply formatters (pass ax explicitly to avoid confusion with colorbar axes)
        # print(f'Set lat ticks at: {N_incr}\tSet long ticks at: {E_incr}')
        ax.xaxis.set_major_formatter(FuncFormatter(WellDetective.make_utm_easting_formatter(target_ax=ax)))
        ax.yaxis.set_major_formatter(FuncFormatter(WellDetective.make_utm_northing_formatter(target_ax=ax)))
        
        # # Set tick spacing (try 500m or 1000m)
        ax.xaxis.set_major_locator(MultipleLocator(E_incr))  # 500m increments
        ax.yaxis.set_major_locator(MultipleLocator(N_incr))
        
        # # Annotations
        ax.set_title("Processed Magnetic Field Survey")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.legend()

        LatCol = self.get_matching_columns(self.data_filtered, self.LAT_COLUMNS)
        LonCol = self.get_matching_columns(self.data_filtered, self.LON_COLUMNS)
        zone, hemisphere, epsg = self.get_survey_utm_zone(lat_col=LatCol[0],lon_col=LonCol[0])
        fig.text(0.02, 0.00, f'Mercator Projection\nWGS84\nUTM Zone: {zone}{hemisphere}', 
                    horizontalalignment='left', fontsize=6)
        
        # # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to: {save_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        return fig, ax

    def plot_Heading_Corr(self, 
                          mag_col=None, 
                          filename=None, 
                          primary_range=None, 
                          secondary_range=None, 
                          bins=50, 
                          figsize=(16, 8), 
                          save_path=None):
        """
        Create figure showing heading and distribution of magnetic signals as a function of heading for a specific flight.
        
        Parameters
        ----------
        mag_col : str, optional
            Name of magnetic field column. If None, auto-detects from MAG_COLUMNS (default: None)
        filename : str
            Filename or substring to filter data by specific flight
        primary_range : tuple, optional
            Primary heading range (min, max) in degrees. 
            If None, will auto-detect using find_primary_secondary_headings()
        secondary_range : tuple, optional
            Secondary heading range (min, max) in degrees.
            If None, will auto-detect using find_primary_secondary_headings()
        bins : int
            Number of histogram bins (default: 50)
        figsize : tuple, optional
            Figure size (width, height) in inches (default: (16, 8))
        save_path : str, optional
            Path to save figure. If None, displays interactively.
        
        Returns
        -------
        fig, (ax0, ax1) : tuple
            Matplotlib figure and axes objects
        
        Example
        -------
        >>> wd = WellDetective(data)
        >>> wd.add_heading_column()
        >>> fig, axes = wd.plot_Heading_Corr(filename='flight_01')
        >>> plt.show()
        """
        # Check if heading column exists
        if 'Heading' not in self.data.columns:
            raise ValueError("Heading not calculated. Run add_heading_column() first.")
        
        # Auto-detect mag column if not provided
        if mag_col is None:
            MagCol = self.get_matching_columns(self.data, self.MAG_COLUMNS)
            if not MagCol:
                raise ValueError(f"No magnetic field column found. Expected columns: {self.MAG_COLUMNS}")
            mag_col = MagCol[0]
            print(f"Auto-detected magnetic field column: '{mag_col}'")
        
        # Check if mag column exists
        if mag_col not in self.data.columns:
            raise ValueError(f"'{mag_col}' column not found in data")
        
        # Filter by filename if provided
        if filename is not None:
            if 'Filepath' not in self.data.columns:
                raise ValueError("'Filepath' column not found in data.")
            file_mask = self.data['Filepath'].str.contains(filename, na=False)
            if file_mask.sum() == 0:
                raise ValueError(f"No data found matching filename: '{filename}'")
            plot_data = self.data[file_mask].copy()
            print(f"Filtered to {len(plot_data):,} points matching '{filename}'")
        else:
            raise ValueError(f"No filename provided. Heading Corrections should be determined by flight")
        
        # Auto-detect heading ranges if not provided
        if primary_range is None or secondary_range is None:
            print("Auto-detecting heading ranges...")
            print("Auto-detecting heading ranges from filtered data...")
            primary_range, secondary_range = self.find_primary_secondary_headings(
                tolerance=WellDetective.DEFAULT_HEADING_TOLERANCE,
                filename=filename
            )
            print(f"Primary range: {primary_range[0]:.2f}° - {primary_range[1]:.2f}°")
            print(f"Secondary range: {secondary_range[0]:.2f}° - {secondary_range[1]:.2f}°")
        
        # Create masks for primary and secondary headings
        primary_mask = plot_data['Heading'].between(*primary_range)
        secondary_mask = plot_data['Heading'].between(*secondary_range)
        
        # Extract data for histograms
        primary_mag = plot_data.loc[primary_mask, mag_col].dropna()
        secondary_mag = plot_data.loc[secondary_mask, mag_col].dropna()
        
        print(f"Primary heading points: {len(primary_mag):,}")
        print(f"Secondary heading points: {len(secondary_mag):,}")
        
        # Create figure with 1x2 subplots
        fig, ax = plt.subplots(1,2, figsize=figsize)
        
        # ========================================
        # ax[0] - Heading vs Index
        # ========================================
        ax[0].plot(plot_data.index, plot_data["Heading"], color="k", linewidth=0.5, label="Heading",zorder=2)

        ax[0].axhspan(primary_range[0], primary_range[1], color='lightgray', alpha=0.3,zorder=1, 
                      label="Primary Heading ({primary_range[0]:.0f}°-{primary_range[1]:.0f}°)")
        ax[0].axhspan(secondary_range[0], secondary_range[1], color='lightblue', alpha=0.3,zorder=1, 
                      label="Secondary Heading ({secondary_range[0]:.0f}°-{secondary_range[1]:.0f}°)")
        
        ax[0].set_xlabel("Index", fontsize=11)
        ax[0].set_ylabel("Heading (degrees)", fontsize=11)
        s_title = (f"Flight Heading Over Time\n"
                   f"Primary Heading ({primary_range[0]:.0f}°-{primary_range[1]:.0f}°)\n"
                   f"Secondary Heading ({secondary_range[0]:.0f}°-{secondary_range[1]:.0f}°)"
                  )
        ax[0].set_title(s_title, fontsize=12, fontweight='bold')
        ax[0].grid(True, alpha=0.3)
        # ax[0].legend(loc='best', fontsize=10)
        ax[0].set_ylim(0, 360)
        
        # ========================================
        # ax[1] - Histograms of Mag by Heading
        # ========================================
        # Calculate statistics
        primary_mean = primary_mag.mean()
        secondary_mean = secondary_mag.mean()
        primary_median = primary_mag.median()
        secondary_median = secondary_mag.median()
        primary_std = primary_mag.std()
        secondary_std = secondary_mag.std()
        
        # Calculate and display correction
        correction_mean = primary_mean - secondary_mean
        correction_median = primary_median - secondary_median
        
        # Plot histograms
        ax[1].hist(primary_mag, bins=bins, alpha=0.6, color='gray', 
                   label=f'Primary (μ={primary_mean:.2f}, σ={primary_std:.2f})', 
                   edgecolor='black', linewidth=0.5)
        ax[1].hist(secondary_mag, bins=bins, alpha=0.6, color='blue', 
                   label=f'Secondary (μ={secondary_mean:.2f}, σ={secondary_std:.2f})', 
                   edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for means
        ax[1].axvline(primary_mean, color='black', linestyle='--', linewidth=2, 
                      label=f'Primary Mean')
        ax[1].axvline(secondary_mean, color='darkblue', linestyle='--', linewidth=2, 
                      label=f'Secondary Mean')
        ax[1].axvline(primary_median, color='black', linestyle=':', linewidth=2, 
                      label=f'Primary Mean')
        ax[1].axvline(secondary_median, color='darkblue', linestyle=':', linewidth=2, 
                      label=f'Secondary Mean')
        
        ax[1].set_xlabel(f"{mag_col} (nT)", fontsize=11)
        ax[1].set_ylabel("Frequency", fontsize=11)
        ax[1].set_title(f"Magnetic Field Distribution by Heading Direction", fontsize=12, fontweight='bold')
        ax[1].grid(True, alpha=0.3, axis='y')
        ax[1].legend(loc='best', fontsize=10,
                    title = f'Heading Correction: \n ∆mean = {correction_mean:.2f} nT \n ∆median = {correction_median:.2f} nT')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to: {save_path}")
        else:
            plt.show()
        
        return fig, ax