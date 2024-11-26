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

hr5_folder = "data/HR5" # define top-level folder 
try:
    os.mkdir(hr5_folder) 
except:
    os.path.exists(hr5_folder) == True

hr5_cells = gpd.read_file('global-inputs/HR5-cells-beach-slope.gpkg') # return gdf of monitored sites
sites = pd.read_csv('global-inputs/sites.csv') # return sites as list
sites_list = sites.cell_id.tolist()
monitored_sites = hr5_cells[hr5_cells['cell_id'].isin(sites_list)]

gdf_cell_list = [gpd.GeoDataFrame([row]) for idx, row in monitored_sites.iterrows()] # Create a list of GeoSeries

### Download images ###
for cell in gdf_cell_list:
    monitoringutils.download_images_in_collection(cell, hr5_folder)

### perform change detection ###
cell_directories = glob.glob(f"{hr5_folder}/*") # iterate over cell directories
for cell_dir in cell_directories:
    monitoringutils.run_change_detection(cell_dir)

# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")