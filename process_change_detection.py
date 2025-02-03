#!/usr/bin/env python3

import ccdutils.monitoringutils as monitoringutils
import glob
import os
import pandas as pd
import multiprocessing

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"



### Process change detection ###
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

def process_cell(folder):
    convert_images(folder)
    check_images(folder)
    monitoringutils.run_change_detection(folder) # run change detection
    monitoringutils.return_tide_levels(f"{folder}/image_metadata.json") # return tide levels

### perform change detection ###
if __name__ == "__main__":
    folder = "data/HR6" # define top-level folder 
    # return valid_cells 
    valid_cells = pd.read_csv("global-inputs/valid_cells.csv")
    cells_to_process = valid_cells.hex_id.to_list()
    cell_directories = glob.glob(f"{folder}/*") # iterate over cell directories
    cell_directories = [i for i in cell_directories if i.split('/')[-1] in cells_to_process]
    for dir in cell_directories:
        p = multiprocessing.Process(target=process_cell, args=(dir,)) # use multiprocessing to avoid memory issues
        p.start()
        p.join()  # Wait for the process to finish
