## IMPORT MODULES ##
# import base modules
import time
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from tqdm.contrib.concurrent import process_map
from functools import partial


# import change modules
import change_detection.change_workflows as workflow
from boundary_analysis import boundary_functions

def msk_img_by_gpd(row, img, folder):
    out_img_path = '{}/cls-img.kea'.format(folder)
    # make sure feature and img crs match
    #row.to_crs(2193, inplace=True)
    geom = row['geometry']
    with rasterio.open(img) as src:
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "KEA",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(out_img_path, "w", **out_meta) as dest:
        dest.write(out_image)

def generate_cls_imgs(cell_id, h3_shp):
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
    cls_img = 'global-inputs/classification/2019-national-5cls-nztm-aoi.kea'
    feature = h3_shp[h3_shp['index'] == cell_id]
    msk_img_by_gpd(feature, cls_img, cls_folder)

def return_input_imgs_folder(h3_id, hr5_dir):
    # define all H3 shp
    h3_grids = gpd.read_file('global-inputs/h3_coast_HR_all_incl_islands.gpkg')
    # return cell
    parent_id = h3_grids.loc[h3_grids['hex_id'] == h3_id, 'parent_id'].item()
    print(parent_id)
    # find folder that matches parent_id
    folder = [i for i in glob.glob(f'{hr5_dir}/*') if i.split('/')[-1] == parent_id]
    return f'{folder[0]}/inputs'

# create function to process change
def process_change_for_cell(cell_folder, h3_shp, inputs_folder=None):
    ## GENERATE CLASS IMAGE FOR EACH CELL
    h3_cells = gpd.read_file(h3_shp)

    # create cls folder path
    cls_folder_path = f'{cell_folder}/inputs/classification' 
    cls_folder = os.path.abspath(cls_folder_path)
    if not os.path.exists(cls_folder):
        os.makedirs(cls_folder)

    # get cell_id from folder path
    cell_id = cell_folder.split('/')[-1]
    print(cell_id)
    
    # define cls-img
    cls_img = 'global-inputs/classification/2019-national-5cls-nztm-aoi.kea'
    feature = h3_cells[h3_cells['index'] == cell_id]
    msk_img_by_gpd(feature, cls_img, cls_folder)


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
    tmp = os.path.join(boundary_folder, 'outputs')
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    ## DEFINE INPUTS ##

    # get input folder
    # if input_folder is none inputs will be in cell folder
    if inputs_folder == None:
        img_folder = f'{cell_folder}/inputs'
    else:
        img_folder = return_input_imgs_folder(cell_id, inputs_folder) ## img_folder comes from HR5 directory using return_input_imgs_folder() 
    print(img_folder)

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
        outputs.append(chg_variables)
        
        
        
    # create pandas dataframe of change results 
    df = pd.DataFrame.from_dict(outputs)
    # save results to csv with folder cell_id as fn
    df.to_csv(f'/Users/ben/Desktop/national-scale-change/HR6-run-2/results/{cell_id}-results.csv')

def process_mp(cell_folder, h3_shp, inputs_folder):
    try:
        # generate cls img
        process_change_for_cell(cell_folder, h3_shp=h3_shp, inputs_folder=inputs_folder)
    except:
        pass

### run processing ###
if __name__ == '__main__':
    # define start time
    start = time.time()

    hr6_dir_path = '/Users/ben/Desktop/national-scale-change/HR6-run-2'
    hr6_dir = os.path.abspath(hr6_dir_path)
    if not os.path.exists(hr6_dir):
        os.makedirs(hr6_dir)
    
    # make data folder
    data_dir_path = f'{hr6_dir}/data'
    data_dir = os.path.abspath(data_dir_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ### Create folder structure for HR6-run-2
    Hr6_gdf = gpd.read_file('global-inputs/HR6-change-cells-aoi.gpkg')

    cell_list = list(Hr6_gdf['index'])

    # for cell_id in cell_list:
    #     # read hr6 shape and generate list of cell id and create folder structure 
    # # create cls folder path
    #     cls_folder_path = f'{data_dir}/{cell_id}'
    #     cls_folder = os.path.abspath(cls_folder_path)
    #     if not os.path.exists(cls_folder):
    #         os.makedirs(cls_folder)

    # define main directtory
    hr5_dir = '/Users/ben/Desktop/national-scale-change/HR5-run-2/data'

    # generate list of folders
    cell_list = glob.glob(f'{data_dir}/*')
    print(len(cell_list))

    # chunk list into portions of 10
    cell_lst_chunks = [cell_list[x:x+10] for x in range(0, len(cell_list), 10)]

    print(cell_lst_chunks)

    #iterate over chunks processing in parallel
    for chunk in cell_lst_chunks:
        process_map(partial(process_mp, h3_shp='global-inputs/HR6-change-cells-aoi.gpkg', inputs_folder=hr5_dir), chunk, max_workers=7)

    end = time.time()
    elapsed = end - start
    print('run time: ', elapsed/60, 'minutes')

