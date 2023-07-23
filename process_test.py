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

    # return metadata_dict as json file
    fn_meta = f"{down_folder}/{feature['properties']['index']}-metadata.json"
    with open(fn_meta, 'w') as file:
        json.dump(metadata_dict, file)

### DOWNLOAD IMAGES ###
if __name__ == '__main__':
    # define start time 
    start = time.time()

    # read cells to download as gpd
    h3_grids = gpd.read_file('H3-coast-all-res.gpkg')
    h3_cells = gpd.read_file('HR5-change-cells.gpkg')

    # return list of cell indexes
    index_list = list(h3_cells[2:3]['index'])

    print(index_list)

    # iterate over index_list and download composites
    for i in index_list:
        # create folder 
        cell_folder_path = f'../national-scale-change/test_folder/{i}' 
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
        cls_img = 'classification/2019-national-5cls-nztm-MMU1ha.kea'
        feature = h3_cells[h3_cells['index'] == i]
        msk_img_by_gpd(feature, cls_img, cls_folder)
        
        # return all child cells at res 5 for test cell at index 4
        down_cells = h3_utils.get_child_cells(h3_grids, i, 6)
        # reproject to 4326 for GEE
        down_cells.to_crs(4326, inplace=True)

        # create dict of cells to be downloaded
        features = down_cells.iterfeatures()

        # download images
        process_map(partial(download_images_mp, down_folder=down_folder), features, max_workers=6)
        
        # generate list of composite years
        year_list = []
        for i in range(1999, 2023, 1):
            year_list.append(i)
        # add 1988
        year_list.append(1988)

        # iterate over composite years list to generate mosaics for HR5
        for year in year_list:
            # create list of images for mosaic
            img_list = []
            for img in glob.glob(f'{down_folder}/*{year}.tif'):
                img_list.append(img)
            # define out_img fn
            out_img_fn = f'{down_folder}/{year}.kea'
            # mosaic images
            image_utils.mosaic(img_list, out_img_fn)
            # remove tif files
            [os.remove(tif) for tif in glob.glob(f'{down_folder}/*{year}.tif')]

        ### RUN CHANGE ANALYSIS ###
        # define empty list for args
        args_list = []
        # define outputs folder path
        out_folder_path = f'{cell_folder}/change-detection' 
        out_folder = os.path.abspath(out_folder_path)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # Generate list containing dict of arguments
        for in_img in glob.glob(f'{down_folder}/*.kea'):
            # split path 
            path_list = in_img.split('/')

            # define img_id
            img_id = path_list[-1][:-4]
            # define output folder for each year
            folder_id_path = f'{out_folder}/{img_id}'
            folder_id = os.path.abspath(folder_id_path)

            # define class_img, ndvi and mndwi bands and class_vals
            cls_img = f'{down_folder}/classification/cls-img.kea'
            ndvi = 1
            mndwi = 2
            class_values = {'water': 2, 'sand': 1, 'vegetation': 3}

            # define args dict
            args = {'input_img' : in_img, 
                    'class_img': cls_img, 
                    'out_folder': folder_id, 
                    'cell_id': i,
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

    end = time.time()
    elapsed = end - start
    print('run time: ', elapsed/60, 'minutes')


    

