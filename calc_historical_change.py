from ccdutils import monitoringutils
import geopandas as gpd 
import pandas as pd
import os
import glob
import ee
import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import json
from datetime import datetime
import gc
import time
import sys
import multiprocessing
import signal

# Redirect stdout and stderr
#sys.stdout = open('.change_detection_processing.log', 'w')
#sys.stderr = open('.change_detection_errors.log', 'w')

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"

##### FUNCTIONS #####
def read_processed_cells(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []

def log_cell(cell_id, log_file):
    try: 
        with open(log_file, 'a') as file:
            file.write(f"{cell_id}\n")
    except FileNotFoundError:
        with open(log_file, 'w') as file:
            file.write(f"{cell_id}")

def batch_iterator(iterable, batch_size):
            for i in range(0, len(iterable), batch_size):
                yield iterable[i:i + batch_size]

def convert_images(folder):
    img_dir = glob.glob(f"{folder}/images/*.tif")
    for img in img_dir:
        monitoringutils.convert_image(img)

def check_images(folder):
    print(folder)
    img_dir = glob.glob(f"{folder}/images/*.kea")
    aoi = f"{folder}/aoi_mask.kea"
    for img in img_dir:
        monitoringutils.check_input_image(img, aoi) # remove images
    #monitoringutils.remove_images_from_metadata(folder)

def process_change(folder):
    cell_id = folder[9:]
    print(f" processing cell: {cell_id}")
    
    monitoringutils.run_change_detection(folder)
    log_cell(cell_id, ".CD_completed_cells.log")

def return_tide_levels(folder):
    cell_id = folder[9:]
    metadata_file = f"{folder}/image_metadata.json"
    monitoringutils.return_tide_levels(metadata_file)
    log_cell(cell_id, ".tide_levels_complete.log")

######## PROCESSING #######

cell_directories = glob.glob("data/HR6/*")

completed = read_processed_cells(".tide_levels_complete.log")

cell_directories = [i for i in cell_directories if i.split('/')[-1] not in completed]

start = time.time()

# for dir in tqdm.tqdm(batch_iterator(cell_directories, 50)): # convert images
#     process_map(convert_images, dir)

# for dir in tqdm.tqdm(batch_iterator(cell_directories, 50)): # check images
#     process_map(check_images, dir)

for dir in tqdm.tqdm(cell_directories):
    
        return_tide_levels(dir)
       

end = time.time()
print(f"Time: {(end-start)/60} mins")