# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:24:36 2025

@author: 336191
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyproj
import zipfile
import xml.etree.ElementTree as ET
import simplekml

from WellDetective.src.general.WellDetective import WellDetective
import pyIGRF
import os
import matplotlib.ticker as ticker

# Define file paths and parameters
asc_file = "400_F1_NN_rtr.asc"
kmz_well_file = "NM oil wells on NN.kmz"

# Read raw data
data = pd.read_csv(asc_file, sep=r"\s+", skiprows=1).iloc[::20]
data.rename(columns={"Longitude[°]": "Long", "Latitude[°]": "Lat", "Totalfield[nT]": "Mag"}, inplace=True)
data.dropna(subset=["Lat", "Long"], inplace=True)

# Add a fixed date/time if needed
data["Date"] = pd.to_datetime("2024-10-25")
data["Time"] = pd.to_timedelta(0.759134, unit="D")

# Plot raw flight line
plt.scatter(data["Long"], data["Lat"], c="k", s=12)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Raw Flight Line")
plt.show()
# Initialize processor
processor = WellDetective(data)

# Add heading column
processor.add_heading_column()

# Find primary/secondary headings
primary_range, secondary_range = processor.find_primary_secondary_headings(tolerance=20)
print("Primary Heading Range:", primary_range)
print("Secondary Heading Range:", secondary_range)

plt.figure(figsize=(10, 6))
plt.plot(processor.data.index, processor.data["Heading"], label="Heading")
plt.axhline(y=primary_range[0], color="g", linestyle="--", label=f"Primary Lower ({primary_range[0]:.2f}°)")
plt.axhline(y=primary_range[1], color="g", linestyle="-", label=f"Primary Upper ({primary_range[1]:.2f}°)")
plt.axhline(y=secondary_range[0], color="r", linestyle="--", label=f"Secondary Lower ({secondary_range[0]:.2f}°)")
plt.axhline(y=secondary_range[1], color="r", linestyle="-", label=f"Secondary Upper ({secondary_range[1]:.2f}°)")
plt.xlabel("Index")
plt.ylabel("Heading (degrees)")
plt.title("Flight - Heading Ranges")
plt.legend()
plt.grid(True)
plt.show()

# Remove turning data
processor.remove_turning_data(primary_range, secondary_range)

print("this is the mean", processor.data["Mag"].mean())


processor.data["Mag"]=processor.data["Mag"] - processor.data["Mag"].mean()
#processor.data = processor.data[processor.data["Mag"] > -45].copy()

#Create segments and remove ones below a certain length
processor.segment_and_filter_data(max_gap_distance=10, min_segment_length=150)

# Equalize heading corrections
processor.auto_equalize_heading_correction(primary_range, secondary_range)


# Project coordinates (UTM zone 12)
processor.project_coordinates(utm_zone=12)

plt.figure(figsize=(10, 6))
plt.plot(processor.data.index, processor.data["Heading"], label="Heading")
plt.axhline(y=primary_range[0], color="g", linestyle="--", label=f"Primary Lower ({primary_range[0]:.2f}°)")
plt.axhline(y=primary_range[1], color="g", linestyle="-", label=f"Primary Upper ({primary_range[1]:.2f}°)")
plt.axhline(y=secondary_range[0], color="r", linestyle="--", label=f"Secondary Lower ({secondary_range[0]:.2f}°)")
plt.axhline(y=secondary_range[1], color="r", linestyle="-", label=f"Secondary Upper ({secondary_range[1]:.2f}°)")
plt.xlabel("Index")
plt.ylabel("Heading (degrees)")
plt.title("Flight - Heading Ranges")
plt.legend()
plt.grid(True)
plt.show()


# Quick flight line plot
plt.figure(figsize=(10, 6))
plt.scatter(
    processor.data["Long"],
    processor.data["Lat"],
    c=processor.data["Corrected"],
    s=4,
    cmap="rainbow",
    edgecolors="none"
)
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.title("Flight - Corrected Magnetic Field")
cbar = plt.colorbar()
cbar.set_label("Corrected Magnetic Field (nT)", fontsize=14)
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis="both", which="major", labelsize=12)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
plt.gca().yaxis.get_major_formatter().set_scientific(False)
plt.show()


# Estimate IGRF parameters
field_info = processor.data.iloc[0]
igrf_results = pyIGRF.igrf_value(
    field_info['Lat'], field_info['Long'], alt=0, year=2024.81
)
inclination, declination = igrf_results[1], igrf_results[0]
print("IGRF Inclination:", inclination, "Declination:", declination)

# Grid and filter data
mag_grid_da = processor.grid_and_filter_data(inclination, declination, grid_res=100, cutoff_wavelength=30, proximity_threshold=40)

# Extract numpy arrays for further processing
grid_x, grid_y = np.meshgrid(mag_grid_da.x.values, mag_grid_da.y.values, indexing='ij')
mag_grid = mag_grid_da.values

# Load known well coordinates from KMZ
with zipfile.ZipFile(kmz_well_file, "r") as kmz:
    kml_file = kmz.open(kmz.namelist()[0])
    tree = ET.parse(kml_file)
    root = tree.getroot()
namespace = {"kml": "http://www.opengis.net/kml/2.2"}
coords = []
for placemark in root.findall(".//kml:Placemark", namespace):
    coord_text = placemark.find(".//kml:coordinates", namespace).text.strip()
    lon_str, lat_str = coord_text.split(",")[:2]
    coords.append((float(lat_str), float(lon_str)))

projection = pyproj.Proj(proj="utm", zone=12, ellps="WGS84")
lat_arr, lon_arr = zip(*coords)
wells_easting, wells_northing = projection(lon_arr, lat_arr)
well_coords = list(zip(wells_easting, wells_northing))

# Detect hotspots and potential orphan wells
hotspots, orphan_wells = processor.detect_hotspots(
    mag_grid, grid_x, grid_y, well_coords, bandwidth=82, distance_threshold=100
)

# Plot final results
plt.figure(figsize=(10, 8), dpi=300)
plt.contourf(grid_x, grid_y, mag_grid, 100, cmap="rainbow")
cbar = plt.colorbar()
cbar.set_label("Magnetic Field (nT)", fontsize=12)

# Plot known wells
plt.scatter(wells_easting, wells_northing, c="k", s=12, label="Known Wells")

# Plot hotspots
#if not hotspots.empty:
#    plt.scatter(hotspots["easting"], hotspots["northing"], c="r", s=30, label="Hotspots")

# Plot orphan wells
#if not orphan_wells.empty:
#    plt.scatter(orphan_wells["easting"], orphan_wells["northing"], c="lime", s=60, marker="^", edgecolors="black", label="Potential Orphan Wells")

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.legend()
plt.title("Magnetic Survey Hotspots & Orphan Wells")
plt.xlim(grid_x.min(),grid_x.max())
plt.ylim(grid_y.min(),grid_y.max())

plt.show()

# Save results as KMZ Overlay
overlay_name = "Magnetic_Field_Overlay"
plt.figure(figsize=(10, 8), dpi=300)
plt.contourf(grid_x, grid_y, mag_grid, 100, cmap="rainbow")
plt.axis("off")
out_image_file = f"{overlay_name}.png"
plt.savefig(out_image_file, bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

# Create KML overlay
kml = simplekml.Kml()
ground = kml.newgroundoverlay(name="Magnetic Overlay")
ground.icon.href = out_image_file
ground.latlonbox.north = processor.data["Lat"].max()
ground.latlonbox.south = processor.data["Lat"].min()
ground.latlonbox.east = processor.data["Long"].max()
ground.latlonbox.west = processor.data["Long"].min()
ground.visibility = 1
kmz_filename = f"{overlay_name}.kmz"
kml.savekmz(kmz_filename)

print(f"KMZ file saved: {kmz_filename}")

# Clean up temporary image
if os.path.exists(out_image_file):
    os.remove(out_image_file)
    print(f"Temporary image {out_image_file} removed.")



