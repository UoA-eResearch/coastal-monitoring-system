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

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"

start_time = time.time() # Start time tracking

folder = "data/HR6" # define top-level folder 
try:
    os.mkdir(folder) 
except:
    os.path.exists(folder) == True

cells = gpd.read_file('global-inputs/HR6-cells-beach-slope.gpkg') # return gdf of monitored sites

gdf_cell_list = [gpd.GeoDataFrame([row]) for idx, row in cells.iterrows()] # Create a list of GeoSeries

### Download images ###
for cell in gdf_cell_list:
    monitoringutils.download_images_in_collection(cell, folder)

### perform change detection ###
cell_directories = glob.glob(f"{folder}/*") # iterate over cell directories
for cell_dir in cell_directories:
    monitoringutils.run_change_detection(cell_dir)

# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")