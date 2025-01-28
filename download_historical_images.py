from ccdutils import monitoringutils
import geopandas as gpd 
import pandas as pd
import os
import glob
import ee
import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import time

service_account = 'monitoring-bot@coastal-monitoring-system.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '.ee_service_account_key.json')
ee.Initialize(credentials)
# read processed items as list
def read_processed_cells(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []

def remove_from_fail(fail_list, cell_id):
    if fail_list == []:
        return
    # Remove the item from the list
    removed = [i for i in fail_list if i != cell_id]
    
    # Write the remaining items to the file
    with open(".cells_failed.log", 'w') as file:
        file.write("\n".join(removed))


### Download images ###
folder = "data/HR6" # define top-level folder 
try:
    os.mkdir(folder) 
except:
    os.path.exists(folder) == True

# return gdf of sites
cells = gpd.read_file('global-inputs/HR6-cells-beach-slope.gpkg')

# Create a list of GeoSeries
gdf_cell_list = [gpd.GeoDataFrame([row]) for idx, row in cells.iterrows()]

to_process = read_processed_cells(".cells_todo.log")

cells = [i for i in gdf_cell_list if i.cell_id.to_string(index=False) in to_process]

#Iterate over the list of GeoSeries
# for cell in tqdm.tqdm(cells):  
#     try:
#         monitoringutils.download_historical_imagery(cell, folder)
#     except  ee.ee_exception.EEException: 
#         print(f"Computation timed out. Retrying...")
#         time.sleep(300)
#         continue

# retry failed cells
# failed = read_processed_cells(".cells_failed.log")

# cells = [i for i in gdf_cell_list if i.cell_id.to_string(index=False) in failed]

# def process_cells(cell, folder):
#     monitoringutils.download_historical_imagery(cell, folder)
#     #remove_from_fail(failed, cell.cell_id.to_string(index=False))

# process_map(process_cells, cells, repeat(folder))


for cell in tqdm.tqdm(cells):  
    monitoringutils.download_historical_imagery(cell, folder)
        # remove_from_fail(failed, cell.cell_id.to_string(index=False))
        # failed = read_processed_cells(".cells_failed.log")
    