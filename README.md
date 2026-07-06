# Well Detective 🕵🏼
Python module for processing and analyzing UAV-based magnetometry data to identify buried oil and gas well casings

## Version & Authorship

**Version:** 2.0.2 (In Development)  
**Original Authors (v1.0):** Eric Guiltinan, Javier Santos  
**Current Authors (v2.0):** Eric Guiltinan, Nash Taylor, James E. Lee  
**Maintainer:** James E. Lee (Los Alamos National Laboratory)  
**Repository:** https://github.com/lanl/WellDetective  
**Last Updated:** June 2026

# Setting up WellDetective

    # clone into the git repo 
    $ git clone git@github.com:lanl/WellDetective.git
    $ cd WellDetective/src
    # install dependencies
    $ python -m pip install --user -r requirements.txt 
    # make the packages available from any directory
    $ python -m pip install -e . 

# Documentation 

## Quick Start

```python
from WellDetective import WellDetective

# Load and process magnetometry data
wd = WellDetective.Load_Process_MultipleMagDataFiles(
    s_folderpath='./data/',
    Mag_File_List=['flight1.csv', 'flight2.csv']
)

# Visualize and export
wd.plot_Mag_Heat()
wd.export_map_to_geotiff('magnetic_map.tif')
wd.export_to_netcdf('survey.nc')

# Fast reload on subsequent runs
wd = WellDetective.Load_from_NetCDF('survey.nc')
```

## Full Documentation

WellDetective documentation can be found [here](https://github.com/lanl/WellDetective/tree/main)

    https://github.com/lanl/WellDetective/tree/main

Key documentation files:
- `WALKTHROUGH.md` - User guide with examples
- `WELLDETECTIVE_FUNCTIONS.md` - Complete function reference
- `INSTALLATION_NOTES.md` - Setup instructions

Or see WellDetective.pdf document in the main directory

# Open-Source License

This LANL repository reference number is O4932

This program is distributed under an OSS license.
 
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.