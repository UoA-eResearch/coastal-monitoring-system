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

start_time = time.time() # Start time tracking

folder = "data/HR6" # define top-level folder 
try:
    os.mkdir(folder) 
except:
    os.path.exists(folder) == True

cells = gpd.read_file('global-inputs/HR6-cells-beach-slope.gpkg')
valid_cells = pd.read_csv("global-inputs/valid_cells.csv")
cells_to_process = set(valid_cells.hex_id)  # Using a set for faster lookup

# Filter the cells based on the valid cell IDs
filtered_cells = cells[cells['cell_id'].isin(cells_to_process)]

# Reset the index of the filtered cells
filtered_cells = filtered_cells.reset_index(drop=True)

# Find the index of the specific cell
specific_cell_id = "86da92137ffffff"
specific_cell_index = filtered_cells.index[filtered_cells['cell_id'] == specific_cell_id].tolist()

# Print the index if the specific cell is found
if specific_cell_index:
    print(specific_cell_index[0])

filtered_cells = filtered_cells.iloc[specific_cell_index[0]:]

# Convert each row to a GeoDataFrame and store in a list
gdf_cell_list = [filtered_cells.iloc[[i]] for i in range(len(filtered_cells))]

### Download images ###
for cell in gdf_cell_list:
    monitoringutils.download_images(cell, folder)


# ### Process change detection ###
# def convert_images(folder):
#     img_dir = glob.glob(f"{folder}/images/*.tif")
#     for img in img_dir:
#         monitoringutils.convert_image(img)

# def check_images(folder):
#     print(folder)
#     img_dir = glob.glob(f"{folder}/images/*.kea")
#     aoi = f"{folder}/aoi_mask.kea"
#     for img in img_dir:
#         monitoringutils.check_input_image(img, aoi) # remove images

# def process_cell(folder):
#     convert_images(folder)
#     check_images(folder)
#     monitoringutils.run_change_detection(folder) # run change detection
#     monitoringutils.return_tide_levels(f"{folder}/image_metadata.json") # return tide levels

# ### perform change detection ###
# if __name__ == "__main__":
#     cell_directories = glob.glob(f"{folder}/*") # iterate over cell directories
#     cell_directories = [i for i in cell_directories if i.split('/')[-1] in cells_to_process]
#     for dir in cell_directories:
#         p = multiprocessing.Process(target=process_cell, args=(dir,)) # use multiprocessing to avoid memory issues
#         p.start()
#         p.join()  # Wait for the process to finish


# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")