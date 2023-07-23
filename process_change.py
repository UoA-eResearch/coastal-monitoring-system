# import modules
from geeutils import feature_utils
from geeutils import sentinel2_utils
from geeutils import image_utils
from geeutils import normalised_difference
from geeutils import h3_utils 
from geeutils import imagecollection_utils

import change_detection.change_workflows as workflow
from boundary_analysis import boundary_functions

import geopandas as gpd
import pandas as pd
import ee
import time
import rsgislib
from rsgislib import imageutils
import rasterio
import rasterio.mask

from tqdm.contrib.concurrent import process_map
from functools import partial
import glob
import os
import json

ee.Initialize()

# iterate over HR5 directory to get cell folders
hr5_dir = '../national-scale-change/HR5'

for cell_folder in glob.glob(f'{hr5_dir}/**'):
    
    print(cell_folder)

    # get input folder
    down_folder = f'{cell_folder}/inputs'

    print(down_folder)

    # get cell_id for subdir
    cell_id = cell_folder.split('/')[-1]

    print(cell_id)

### RUN CHANGE ANALYSIS ###
    # define empty list for args
    args_list = []
    # define outputs folder path
    out_folder_path = f'{cell_folder}/change-detection' 
    out_folder = os.path.abspath(out_folder_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Generate list containing dict of arguments
    for in_img in glob.glob(down_folder  + '/*.kea'):
        # split path 
        path_list = in_img.split('/')

        # define img_id
        img_id = path_list[-1][:-4]
        # define output folder for each year
        folder_id_path = f'{out_folder}/{img_id}'
        folder_id = os.path.abspath(folder_id_path)

        # define class_img, ndvi and mndwi bands and class_vals
        cls_img = f'{down_folder}/classification/cls-img.kea'
        print(cls_img)
        ndvi = 1
        mndwi = 2
        class_values = {'water': 2, 'sand': 1, 'vegetation': 3}

        # define args dict
        args = {'input_img' : in_img, 
                'class_img': cls_img, 
                'out_folder': folder_id, 
                'cell_id': cell_id,
                'ndvi_band': ndvi,
                'ndwi_band': mndwi,
                'class_vals': {'water': 2, 'sand': 1, 'vegetation': 3}
                }
        args_list.append(args)
    
    # iterate over args_list and generate change outputs
    for i in args_list:
        print(i)
        workflow.return_change_output_otsu_merged_classes(**i)

    ### PERFORM BOUNDARY ANALYSIS
    # create boundary analysis output folder
    boundary_folder_path = f'{cell_folder}/boundary-analysis'
    boundary_folder = os.path.abspath(boundary_folder_path)
    if not os.path.exists(boundary_folder):
        os.makedirs(boundary_folder)
    
    # create tmp folder for boundary analysis outputs
    tmp = os.path.join(boundary_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # return list of folders that change analysis is to be run on
    chg_output_subdirs = glob.glob(f'{out_folder}/*', recursive=True)

    # define output list to contain dicts for each time instance
    outputs = []

    # iterate over subdirs to build dictionary containing all attributes required in csv
    for dir in chg_output_subdirs:
        print(dir)
        # define change outputs
        ndwi_chg = glob.glob(dir + '/ndwi-chg.kea')[0]
        ndvi_chg = glob.glob(dir + '/ndvi-chg.kea')[0]
        print(ndwi_chg)
        print(ndvi_chg)

        # define a dict to store variables
        chg_variables  = {}
        # get the date from the subdir path
        date = dir.split('/')[-1]
        chg_variables['year'] = date

        # calc cdi and estimated change and add to chg_variables
        # calc cdi_iw
        cdi_iw = boundary_functions.calc_cdi(ndwi_chg, cls_img, tmp=tmp, mask_vals=[3,4,5], eov_boundary=False)
        chg_variables['cdi_iw'] = cdi_iw['cdi_iw']
        chg_variables['est_iw_chg'] = cdi_iw['est_iw_chg']

        # calc cdi_eov
        cdi_eov = boundary_functions.calc_cdi(ndvi_chg, cls_img, tmp=tmp, mask_vals=[2,4,5], eov_boundary=True)
        chg_variables['cdi_eov'] = cdi_eov['cdi_eov']
        chg_variables['est_eov_chg'] = cdi_eov['est_eov_chg']

        # return change pixels and calc area change
        output_chg_pxls_img = f'{tmp}/{date}-chg-pixels.kea'
        boundary_functions.return_chg_pixels(dir, cls_img, output_chg_pxls_img)
        # calc area change
        area_chg = boundary_functions.calc_area_change(output_chg_pxls_img)

        # add area change values to chg_variables dict
        chg_variables['area_change_iw (m\u00b2)'] = area_chg['area_iw']
        chg_variables['area_change_eov (m\u00b2)'] = area_chg['area_eov']

        # return new class images for each time instance
        new_class_img = f'{dir}/new_classification.kea'
        boundary_functions.return_new_class_image(dir, new_class_img)

        # append chg_variables to outputs_list
        outputs.append(chg_variables)
        
    # create pandas dataframe of change results 
    df = pd.DataFrame.from_dict(outputs)

    df.sort_values(by=['year'])

    df.to_csv(f'{boundary_folder}/boundary-analysis-results.csv')