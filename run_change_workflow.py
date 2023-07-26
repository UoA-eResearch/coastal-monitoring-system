## IMPORT MODULES ##
# import base modules
import time
import os
import glob
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map


# import change modules
import change_detection.change_workflows as workflow
from boundary_analysis import boundary_functions

# create function to process change
def process_change_for_cell(cell_folder):
    ## CREATE OUTPUT FOLDERS ##
    # define outputs folder path
    out_folder_path = f'{cell_folder}/change-detection' 
    out_folder = os.path.abspath(out_folder_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # define boundary boundary analysis output folder
    boundary_folder_path = f'{cell_folder}/boundary-analysis'
    boundary_folder = os.path.abspath(boundary_folder_path)
    if not os.path.exists(boundary_folder):
        os.makedirs(boundary_folder)
    
    # create tmp folder for boundary analysis outputs
    tmp = os.path.join(boundary_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    ## DEFINE INPUTS ##
    # get input folder
    img_folder = f'{cell_folder}/inputs'
    print(img_folder)

    # get cell_id for subdir
    cell_id = cell_folder.split('/')[-1]
    print(cell_id)

    # define class_img, ndvi and mndwi bands and class_vals
    cls_img = f'{cell_folder}/inputs/classification/cls-img.kea'
    ndvi = 1
    mndwi = 2
    class_values = {'water': 2, 'sand': 1, 'vegetation': 3}

    ### RUN CHANGE ANALYSIS ###
    # iterate over images and return change outputs
    for in_img in glob.glob(img_folder  + '/*.kea'):
        # define img_id
        img_id = in_img.split('/')[-1][:-4]
        
        # define output folder for each year
        folder_id_path = f'{out_folder}/{img_id}'
        folder_id = os.path.abspath(folder_id_path)

        # run change detection for img
        workflow.return_change_output_otsu_merged_classes(
            in_img,
            cls_img,
            folder_id,
            cell_id,
            ndvi,
            mndwi,
            class_values
        )
    
    ### RUN BOUNDARY ANALYSIS ###
    # return list of folders that change analysis is to be run on
    chg_output_subdirs = glob.glob(f'{out_folder}/*[0-9]', recursive=True)
    # define output list to contain dicts for each time instance
    outputs = []

    # iterate over subdirs to build dictionary containing all attributes required in csv
    for dir in chg_output_subdirs:
        print(dir)
        
        # define a dict to store variables
        chg_variables  = {}
        # get the date from the subdir path
        date = dir.split('/')[-1]
        chg_variables['year'] = date

        # add cell_index id to chg variables
        chg_variables['cell_id'] = cell_id

        # define change outputs
        ndwi_chg = glob.glob(dir + '/ndwi*')
        ndvi_chg = glob.glob(dir + '/ndvi*')
        print(ndwi_chg)
        print(ndvi_chg)

        # calc cdi and estimated change and add to chg_variables
        # calc cdi_iw
        if ndwi_chg != []:
            cdi_iw = boundary_functions.calc_cdi(ndwi_chg[0], cls_img, tmp=tmp, mask_vals=[3,4,5], eov_boundary=False)
            chg_variables['cdi_iw'] = cdi_iw['cdi_iw']
            chg_variables['est_iw_chg'] = cdi_iw['est_iw_chg']
        else:
            chg_variables['cdi_iw'] = np.nan
            chg_variables['est_iw_chg'] = np.nan

        # calc cdi_eov
        if ndvi_chg != []:
            cdi_eov = boundary_functions.calc_cdi(ndvi_chg[0], cls_img, tmp=tmp, mask_vals=[2,4,5], eov_boundary=True)
            chg_variables['cdi_eov'] = cdi_eov['cdi_eov']
            chg_variables['est_eov_chg'] = cdi_eov['est_eov_chg']
        else:
            chg_variables['cdi_eov'] = np.nan
            chg_variables['est_eov_chg'] = np.nan
        
        try:
            # return change pixels and calc area change
            output_chg_pxls_img = f'{tmp}/{date}-chg-pixels.kea'
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


        # append chg_variables to outputs_list
        outputs.append(chg_variables)
        
        
        
    # create pandas dataframe of change results 
    df = pd.DataFrame.from_dict(outputs)

    

    df.to_csv(f'{boundary_folder}/boundary-analysis-results.csv')


### run processing ###
if __name__ == '__main__':
    # define start time
    start = time.time()

    # define main directtory
    hr5_dir = '/cd-data/HR5'

    # generate list of folders
    cell_list = glob.glob(f'{hr5_dir}/*')
    print(len(cell_list))

    # chunk list into portions of 10
    cell_lst_chunks = [cell_list[x:x+10] for x in range(0, len(cell_list), 10)]

    print(cell_lst_chunks)

    # iterate over chunks processing in parallel
    for chunk in cell_lst_chunks:
        process_map(process_change_for_cell, chunk, max_workers=7)

    end = time.time()
    elapsed = end - start
    print('run time: ', elapsed/60, 'minutes')


