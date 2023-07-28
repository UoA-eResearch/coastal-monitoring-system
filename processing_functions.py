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
import numpy as np
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

def return_metadata(img_collection, sensor):
    # define metadata dict and img list
    metadata = {}
    img_list = []

    # add year and sensor to dict
    #metadata['year'] = year
    metadata['sensor'] = sensor

    # add number of images to dict
    # return number of image in collection
    metadata['number of images'] = img_collection.size().getInfo()

    # return image ids
    for i in img_collection.getInfo()['features']:
        # return ee.Image id
        id_str = i['id']
        # add to img_list
        img_list.append(id_str)

    # add image ids to dict
    metadata['images'] = img_list
    print(metadata)

    # convert metadata dict to df and write to csv
    #metadata_df = pd.DataFrame.from_dict(metadata_dict)
    return metadata

# function to mask national classification to h3 cell
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

# define function to download composites for h3 cell in parallel
def download_images_mp(feature, down_folder):
    
    # return cell as ee.featureCollection
    aoi = feature_utils.item_to_featureCollection(feature)

   # create dict of 'year: sensor' for annual composites
    composites = {} 
    # add '1998: LS4'
    composites[1988] = 'LS4'
    # add other sensors
    for i in range(1999, 2023, 1):
        if i < 2013:
            composites[i] = 'LS7'
        elif 2012 < i < 2018:
            composites[i] = 'LS8'
        else:
            composites[i] = 'S2'

    # define a metadata dict to store metadata for each sensor and year
    metadata_dict = {}
    
    # iterate over composites dict and download 
    for year, sensor in composites.items():
        # generate image collection
        img_collection = imagecollection_utils.gen_imageCollection(year, aoi, sensor)

        if img_collection.size().getInfo() != 0:
            # calculate ndvi and mndwi
            img_collection = (img_collection.map(normalised_difference.apply_ndvi)
                            .map(normalised_difference.apply_mndwi))
            
            # select indice bands
            indices_col = img_collection.select(['ndvi', 'mndwi'])

            # generate geometric composite
            indices_composite = indices_col.reduce(ee.Reducer.geometricMedian(2))

            # rename bands
            band_names = ['ndvi', 'mndwi']
            indices_composite = indices_composite.select([0,1]).rename(band_names)

            # generate pixel count and add to indices_composite
            pixel_freq = indices_col.reduce(ee.Reducer.count()).rename(['observation_freq_ndvi', 'observation_freq_mndwi'])
            # add pixel_freq to composite
            indices_composite = indices_composite.addBands(pixel_freq)
            
            # clip composite_image for export
            clip = indices_composite.clip(aoi).unmask(-99)

            # define fn and metadata fn
            fn = f"{feature['properties']['index']}-{sensor}-{year}.tif"
            
            # return metadata and update metadata_dict
            comp_metadata = return_metadata(img_collection, sensor)
            metadata_dict[year] = comp_metadata

            # download image
            image_utils.download_img_local(clip.toFloat(), down_folder, fn, aoi.geometry(), 'EPSG:2193', 20)
        
        else:
            continue

    # return metadata_dict as json file
    fn_meta = f"{down_folder}/{feature['properties']['index']}-metadata.json"
    with open(fn_meta, 'w') as file:
          json.dump(metadata_dict, file)

def mosaic_annual_composites(folder):
# generate list of composite years
        year_list = []
        for i in range(1999, 2023, 1):
            year_list.append(i)
        # add 1988
        year_list.append(1988)

        # iterate over composite years list to generate mosaics for HR5
        for year in year_list:
            try:
                # create list of images for mosaic
                img_list = []
                for img in glob.glob(f'{folder}/*{year}.tif'):
                    img_list.append(img)
                # define out_img fn
                out_img_fn = f'{folder}/{year}.kea'
                # mosaic images
                image_utils.mosaic(img_list, out_img_fn)
                # remove tif files
                [os.remove(tif) for tif in glob.glob(f'{down_folder}/*{year}.tif')]
            except:
                continue

def return_input_imgs_folder(h3_id):
    # define all H3 shp
    h3_grids = gpd.read_file('global-inputs/h3-coast-all-res.gpkg')
    # define HR5 directory
    HR5_dir = '/cd-data/HR5'
    # return cell
    h3_cell = h3_grids[h3_grids['cell_id'] == h3_id]
    # find folder that matches parent_id
    folder = [i for i in glob.glob(f'{HR5_dir}/*') if i.split['/'][-1] == h3_cell['parent_id']]
    return folder


# create function to process change
def process_change_for_cell(cell_folder, output_folder):
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
    # save results to csv with folder cell_id as fn
    df.to_csv(f'{output_folder}/{cell_id}-results.csv')