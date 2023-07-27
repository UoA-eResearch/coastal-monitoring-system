# import modules
import geopandas as gpd
import os
import time
from tqdm.contrib.concurrent import process_map
from functools import partial
import glob

# import processing functions
import processing_functions

### DOWNLOAD IMAGES ###
if __name__ == '__main__':
    # define start time 
    start = time.time()

    # read cells to download as gpd
    h3_grids = gpd.read_file('global-inputs/h3-coast-all-res.gpkg')
    h3_cells = gpd.read_file('global-inputs/HR5-change-cells.gpkg')

    # return list of cell indexes
    index_list = list(h3_cells[:1]['index'])

    print(index_list)

    # iterate over index_list and download composites
    for i in index_list:
        # create folder 
        cell_folder_path = f'/cd-data/HR6/{i}' 
        cell_folder = os.path.abspath(cell_folder_path)
        if not os.path.exists(cell_folder):
            os.makedirs(cell_folder)
        # create download folder path
        down_folder_path = f'{cell_folder}/inputs' 
        down_folder = os.path.abspath(down_folder_path)
        if not os.path.exists(down_folder):
            os.makedirs(down_folder)

        # clip classification to h3 cell
        # create cls folder path
        cls_folder_path = f'{cell_folder}/inputs/classification' 
        cls_folder = os.path.abspath(cls_folder_path)
        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder)
        # define cls-img
        cls_img = 'global-inputs/classification/2019-national-5cls-nztm-MMU1ha.kea'
        feature = h3_cells[h3_cells['index'] == i]
        msk_img_by_gpd(feature, cls_img, cls_folder)

        # return all child cells at res 5 for test cell at index 4
        down_cells = h3_grids[h3_grids['index'] == i] 
        # reproject to 4326 for GEE
        down_cells.to_crs(4326, inplace=True)

        # create dict of cells to be downloaded
        features = down_cells.iterfeatures()

        # download images
        process_map(partial(processing_functions.download_images_mp, down_folder=down_folder), features, max_workers=7)
        
        # mosaic annual images
        processing_functions.mosaic_annual_composites(down_folder)

        # process cell
        process_map(partial(processing_functions.process_change_for_cell, output_folder='/cd-data/HR6-results'),
        cell_folder, max_workers=7)
