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

unmatched = read_processed_cells(".metadata_record_unmatched.log")

cell_directories = glob.glob("data/HR6/*")

cell_directories = [i for i in cell_directories if i[9:] in unmatched]

print(len(cell_directories))

for i in tqdm.tqdm(cell_directories):
    print(i)
    monitoringutils.return_metadata(i)