import ccdutils.monitoringutils as monitoringutils
import ee
from datetime import datetime
import time
import glob
import os
import geopandas as gpd 
import pandas as pd
import os


# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"

# init gee with project 
ee.Initialize()


# Start time tracking
start_time = time.time()



#monitoringutils.download_images_in_collection("data/regions/85bb58c7fffffff.gpkg", "data/85bb58c7fffffff") # Download images

# monitoringutils.run_change_detection("data/85bb58c7fffffff") # run change detection

# define top-level folder 
hr5_folder = "data/HR5"
try:
    os.mkdir(hr5_folder) 
except:
    os.path.exists(hr5_folder) == True


# return gdf of sites
hr5_cells = gpd.read_file('global-inputs/HR5-cells-beach-slope.gpkg')

sites = pd.read_csv('global-inputs/sites.csv')
sites_list = sites.cell_id.tolist()
#print(sites_list)

monitor_sites = hr5_cells[hr5_cells['cell_id'].isin(sites_list)]

monitor_sites

# Create a list of GeoSeries
gdf_cell_list = [gpd.GeoDataFrame([row]) for idx, row in monitor_sites.iterrows()]

# Iterate over the list of GeoSeries
for cell in gdf_cell_list:
    monitoringutils.download_historical_imagery(cell, hr5_folder)



# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")

