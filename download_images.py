#!/usr/bin/env python3

import ee
service_account = 'monitoring-bot@coastal-monitoring-system.iam.gserviceaccount.com' # init gee with project 
credentials = ee.ServiceAccountCredentials(service_account, '.ee_service_account_key.json')
ee.Initialize(credentials)

import ccdutils.monitoringutils as monitoringutils
from datetime import datetime
import time
import glob
import os
import geopandas as gpd 
import pandas as pd
import multiprocessing

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"

folder = "data/HR6" # define top-level folder 

cells = gpd.read_file('global-inputs/HR6-cells-beach-slope.gpkg')
valid_cells = pd.read_csv("global-inputs/valid_cells.csv")
cells_to_process = set(valid_cells.hex_id)  # Using a set for faster lookup

# Filter the cells based on the valid cell IDs
filtered_cells = cells[cells['cell_id'].isin(cells_to_process)]

# Convert each row to a GeoDataFrame and store in a list
gdf_cell_list = [filtered_cells.iloc[[i]] for i in range(len(filtered_cells))]

### Download images ###
for cell in gdf_cell_list:
    monitoringutils.download_images(cell, folder)