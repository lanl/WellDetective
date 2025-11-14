# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:16:22 2025

@author: Eric Guiltinan
"""

#WellDetective.py
import numpy as np
import pandas as pd
import pyproj
from sklearn.cluster import KMeans 
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import xrft
import harmonica as hm
import xarray as xr

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

    def __init__(self, data: pd.DataFrame):
        """
        Initialize GeoMagProcessor with survey data.

        Parameters
        ----------
        data : pd.DataFrame
            Initial survey data with required columns ('Lat', 'Long', 'Mag').
        """
        self.data = data.copy()

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
        None
        """
        headings = [0] * window
        for i in range(window, len(self.data)):
            lat1, lon1 = self.data.iloc[i - window][[lat_col, lon_col]]
            lat2, lon2 = self.data.iloc[i][[lat_col, lon_col]]
            heading = self.calculate_heading(lat1, lon1, lat2, lon2)
            headings.append(heading)
        self.data["Heading"] = headings

    def find_primary_secondary_headings(self, tolerance=10):
        """
        Identify primary and secondary heading ranges using K-Means clustering.

        Parameters
        ----------
        tolerance : float
            Degrees around the cluster centers for heading ranges.

        Returns
        -------
        tuple
            Primary and secondary heading ranges as tuples (min, max).
        """
        headings = self.data["Heading"].values.reshape(-1, 1)
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
        None
        """
        mask = ((self.data["Heading"].between(*primary_range)) |
                (self.data["Heading"].between(*secondary_range)))
        self.data = self.data[mask].reset_index(drop=True)
        
        
    def segment_and_filter_data(self, max_gap_distance=20, min_segment_length=100, lat_col="Lat", lon_col="Long"):
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
        None
            Updates self.data to include only segments meeting length criteria.
        """
        def compute_distance(lat1, lon1, lat2, lon2):
            R = 6371000.0  # Earth radius in meters
            phi1, phi2 = np.radians([lat1, lat2])
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)
            a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            return R * c
    
        # Group data into segments
        segments = []
        current_segment = [0]
    
        for i in range(1, len(self.data)):
            lat1, lon1 = self.data.iloc[i - 1][[lat_col, lon_col]]
            lat2, lon2 = self.data.iloc[i][[lat_col, lon_col]]
            dist = compute_distance(lat1, lon1, lat2, lon2)
    
            if dist > max_gap_distance:
                segments.append(self.data.iloc[current_segment])
                current_segment = [i]
            else:
                current_segment.append(i)
    
        if current_segment:
            segments.append(self.data.iloc[current_segment])
    
        # Calculate and filter by segment lengths
        filtered_segments = []
        for segment in segments:
            total_distance = 0
            for j in range(1, len(segment)):
                lat1, lon1 = segment.iloc[j - 1][[lat_col, lon_col]]
                lat2, lon2 = segment.iloc[j][[lat_col, lon_col]]
                total_distance += compute_distance(lat1, lon1, lat2, lon2)
    
            if total_distance >= min_segment_length:
                filtered_segments.append(segment)
    
        # Update self.data with filtered segments
        if filtered_segments:
            self.data = pd.concat(filtered_segments).reset_index(drop=True)
        else:
            self.data = pd.DataFrame(columns=self.data.columns)


    def auto_equalize_heading_correction(self, primary_range, secondary_range, mag_col="Mag"):
        """
        Apply heading correction to equalize magnetic field measurements.

        Parameters
        ----------
        primary_range, secondary_range : tuple
            Heading ranges for primary and secondary directions.
        mag_col : str
            Column name of magnetic field measurements.

        Returns
        -------
        None
        """
        primary_mean = self.data.loc[self.data["Heading"].between(*primary_range), mag_col].mean()
        secondary_mean = self.data.loc[self.data["Heading"].between(*secondary_range), mag_col].mean()
        correction = primary_mean - secondary_mean
        idx_secondary = self.data["Heading"].between(*secondary_range)
        self.data.loc[idx_secondary, mag_col] += correction
        self.data["Corrected"] = self.data[mag_col]

    def project_coordinates(self, utm_zone=12):
        """
        Project geographic coordinates to UTM.

        Parameters
        ----------
        utm_zone : int
            UTM zone for projection.

        Returns
        -------
        None
        """
        projection = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84", preserve_units=False)
        easting, northing = projection(self.data["Long"].values, self.data["Lat"].values)
        self.data["easting"] = easting
        self.data["northing"] = northing
 
    
    def grid_and_filter_data(self, inclination, declination, grid_res=500, cutoff_wavelength=30, proximity_threshold=40):
        """
        Interpolate, reduce-to-pole, and low-pass filter magnetic data.
        Subtracts mean magnetic field before processing and masks areas
        too far from the flight path.
    
        Parameters
        ----------
        inclination, declination : float
            IGRF inclination and declination values.
        grid_res : int
            Grid resolution for interpolation.
        cutoff_wavelength : float
            Wavelength for low-pass filter (meters).
        proximity_threshold : float
            Maximum distance (in meters) from flight line to keep grid points.
    
        Returns
        -------
        xarray.DataArray
            Processed and masked magnetic data grid.
        """
        # Subtract mean
        mag_mean = self.data["Corrected"].mean()
        corrected = self.data["Corrected"] - mag_mean
    
        # Create grid
        xi, yi = np.mgrid[
            self.data["easting"].min():self.data["easting"].max():grid_res*1j,
            self.data["northing"].min():self.data["northing"].max():grid_res*1j
        ]
        
        # Interpolate to grid
        zi = griddata(
            (self.data["easting"], self.data["northing"]),
            corrected,
            (xi, yi),
            method="linear"
        )
    
        # Create xarray
        da = xr.DataArray(zi, dims=["x", "y"], coords={"x": xi[:, 0], "y": yi[0, :]})
    
        # Pad and reduce to pole
        pad_width = {"x": da.x.size // 3, "y": da.y.size // 3}
        padded = xrft.pad(da, pad_width=pad_width).fillna(0.0)
        rtp = hm.reduction_to_pole(padded, inclination, declination)
        rtp_unpadded = xrft.unpad(rtp, pad_width)
    
        # Apply low-pass filter
        filtered = hm.gaussian_lowpass(rtp_unpadded, wavelength=cutoff_wavelength)
    
        # Build KDTree from original easting/northing
        flight_tree = cKDTree(np.c_[self.data["easting"], self.data["northing"]])
        grid_points = np.c_[xi.flatten(), yi.flatten()]
        distances, _ = flight_tree.query(grid_points)
        distance_grid = distances.reshape(xi.shape)
    
        # Mask out points beyond the threshold
        mask = distance_grid <= proximity_threshold
        low_passed = filtered.where(mask)
    
        return low_passed


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
        import pandas as pd
        import numpy as np
        from sklearn.cluster import MeanShift
        from scipy.spatial import distance
    
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

