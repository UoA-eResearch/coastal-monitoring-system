# import modules
import geopandas as gpd
import os
import time
from tqdm.contrib.concurrent import process_map
from functools import partial
import glob

# import processing functions
import processing_functions

def generate_cls_imgs(cell_id):
    # create folder 
    cell_folder_path = f'/cd-data/HR6/{cell_id}' 
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
    feature = h3_cells[h3_cells['index'] == cell_id]
    processing_functions.msk_img_by_gpd(feature, cls_img, cls_folder)

### DOWNLOAD IMAGES ###
if __name__ == '__main__':
    # define start time 
    start = time.time()

    # read cells to download as gpd
    h3_grids = gpd.read_file('global-inputs/h3-coast-all-res.gpkg')
    h3_cells = gpd.read_file('global-inputs/HR6-change-cells-incl-islands.gpkg')

    # return list of cell indexes
    index_list = list(h3_cells[:1]['index'])

    print(index_list)

    # iterate over index_list and download composites
    for i in index_list:

        # generate cls img
        generate_cls_imgs(i)

        # test folder return
        test = processing_functions.return_input_imgs_folder(i)

        print(test)


        # # make cell folder iterable
        # cell_iterable = [cell_folder]

        # print(cell_iterable)

        # # process cell
        # process_map(partial(processing_functions.process_change_for_cell, output_folder='/cd-data/HR6-results'),
        # cell_iterable, max_workers=7)

    end = time.time()
    elapsed = end - start
    print('run time: ', elapsed/60, 'minutes')