# import modules
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import time
from tqdm.contrib.concurrent import process_map
from functools import partial
import glob

# import boundary function
import boundary_analysis.boundary_functions as boundary_functions

def run_boundary_analysis_for_cell(cell_folder, output_folder):
    # define boundary boundary analysis output folder
    boundary_folder_path = f'{cell_folder}/boundary-analysis'
    boundary_folder = os.path.abspath(boundary_folder_path)
    if not os.path.exists(boundary_folder):
        os.makedirs(boundary_folder)

    # create outputs folder for boundary analysis outputs
    outputs = os.path.join(boundary_folder, 'outputs')
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    # blank list to store change outputs
    outputs_list = []

    # define cell_id from folder path
    cell_id = cell_folder.split('/')[-1]

    # iterate over subdirs to build dictionary containing all attributes required in csv
    for dir in glob.glob(cell_folder + '/change-detection/*[0-9]'):
        print(dir)
        
        # define a dict to store variables
        chg_variables  = {}
        # get the date from the subdir path
        date = dir.split('/')[-1]
        chg_variables['year'] = date

        # add cell_index id to chg variables
        chg_variables['cell_id'] = cell_id

        # define change outputs and initial cls image
        ndwi_chg = f'{dir}/ndwi-chg.kea'
        ndvi_chg = f'{dir}/ndvi-chg.kea'
        cls_img = f'{cell_folder}/inputs/classification/cls-img.kea'

        print(cls_img)
        print(ndwi_chg)
        print(ndvi_chg)

        # calc cdi and estimated change and add to chg_variables
        # calc cdi_iw
        if os.path.exists(ndwi_chg) == True:
            cdi_iw = boundary_functions.calc_cdi(ndwi_chg, cls_img, tmp=outputs, mask_vals=[3,4,5], eov_boundary=False)
            chg_variables['cdi_iw'] = cdi_iw['cdi_iw']
            chg_variables['est_iw_chg'] = cdi_iw['est_iw_chg']
        else:
            chg_variables['cdi_iw'] = np.nan
            chg_variables['est_iw_chg'] = np.nan

        # calc cdi_eov
        if os.path.exists(ndvi_chg) == True:
            cdi_eov = boundary_functions.calc_cdi(ndvi_chg, cls_img, tmp=outputs, mask_vals=[2,4,5], eov_boundary=True)
            chg_variables['cdi_eov'] = cdi_eov['cdi_eov']
            chg_variables['est_eov_chg'] = cdi_eov['est_eov_chg']
        else:
            chg_variables['cdi_eov'] = np.nan
            chg_variables['est_eov_chg'] = np.nan
        
        try:
            # return change pixels and calc area change
            output_chg_pxls_img = f'{outputs}/{date}-chg-pixels.kea'
            boundary_functions.return_chg_pixels(dir, cls_img, output_chg_pxls_img)
            # calc area change
            area_chg = boundary_functions.calc_area_change(output_chg_pxls_img)

            # add area change values to chg_variables dict
            chg_variables['area_change_iw (Ha)'] = area_chg['area_iw']
            chg_variables['area_change_eov (Ha)'] = area_chg['area_eov']

            # return new class images for each time instance
            new_class_img = f'{dir}/new_classification.kea'
            boundary_functions.return_new_class_image(dir, new_class_img)

        except:
            chg_variables['area_change_iw (Ha)'] = np.nan
            chg_variables['area_change_eov (Ha)'] = np.nan
        
        # add class areas from new class image
        try:
            cls_areas = boundary_functions.return_new_class_area(new_class_img)

            # add cls areas to variables dict
            chg_variables['sand_area'] = cls_areas['sand_area']
            chg_variables['water_area'] = cls_areas['water_area']
            chg_variables['vegetation_area'] = cls_areas['vegetation_area']
        
        except:
            chg_variables['sand_area'] = np.nan
            chg_variables['water_area'] = np.nan
            chg_variables['vegetation_area'] = np.nan

        # append chg_variables to outputs_list
        outputs_list.append(chg_variables)
        
       
    # create pandas dataframe of change results 
    df = pd.DataFrame.from_dict(outputs_list)
    # save results to csv with folder cell_id as fn
    df.to_csv(f'{output_folder}/{cell_id}-results.csv')

def run_chg_pixels_for_cell(cell_folder, output_folder):
    # define boundary boundary analysis output folder
    boundary_folder_path = f'{cell_folder}/boundary-analysis'
    boundary_folder = os.path.abspath(boundary_folder_path)
    if not os.path.exists(boundary_folder):
        os.makedirs(boundary_folder)

    # create outputs folder for boundary analysis outputs
    outputs = os.path.join(boundary_folder, 'outputs')
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    # iterate over subdirs to build dictionary containing all attributes required in csv
    for dir in glob.glob(cell_folder + '/change-detection/*[0-9]'):
        print(dir)
        
        # define a dict to store variables
        chg_variables  = {}
        # get the date from the subdir path
        date = dir.split('/')[-1]

        # define change outputs and initial cls image
        ndwi_chg = f'{dir}/ndwi-chg.kea'
        ndvi_chg = f'{dir}/ndvi-chg.kea'
        cls_img = f'{cell_folder}/inputs/classification/cls-img.kea'

        print(cls_img)
        print(ndwi_chg)
        print(ndvi_chg)
        
        
        # return change pixels and calc area change
        output_chg_pxls_img = f'{outputs}/{date}-chg-pixels.kea'
        boundary_functions.return_chg_pixels(dir, cls_img, output_chg_pxls_img)

        
        

def chunk_iterable(iterable, chunk_size):
    return [iterable[x:x+chunk_size] for x in range(0, len(iterable), chunk_size)]


def process_mp(cell_folder, output_folder):

    run_chg_pixels_for_cell(cell_folder, 
    output_folder=output_folder)

    run_boundary_analysis_for_cell(cell_folder, 
    output_folder=output_folder)

    



    
if __name__ == '__main__':
    # define start time 
    start = time.time()

    # create output folder
    out_dir_fp = '/Users/ben/Desktop/national-scale-change/HR5-run-2/results'
    if not os.path.exists(out_dir_fp):
        os.makedirs(out_dir_fp)

    # return list of cell indexes
    index_list = glob.glob('/Users/ben/Desktop/national-scale-change/HR5-run-2/data/*') 

    print(index_list)

    # chunk list
    index_chunks = chunk_iterable(index_list, 10)

    # print(index_chunks)
    print(len(index_chunks))

    # process
    for chunk in index_chunks:
        process_map(partial(process_mp, output_folder=out_dir_fp),
        chunk, max_workers=7)

    end = time.time()
    elapsed = end - start
    print('run time: ', elapsed/60, 'minutes')