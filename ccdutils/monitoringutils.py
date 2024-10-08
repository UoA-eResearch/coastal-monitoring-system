#!/usr/bin/env python

#########################################################
# monitoringutils.py

# Purpose: A series of functions to download and process 
#          images for automatic monitoring 
#          of coastal landcover change. 
#########################################################

## import modules 
from ccdutils.geetools import featureutils
from ccdutils.geetools import imagecollectionutils
from ccdutils.geetools import ndutils
from ccdutils.geetools import imageutils
from ccdutils.changedetection import workflows
from ccdutils.boundaryanalysis import tools
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import ee
import tqdm
import json
import os
import rsgislib
import rsgislib.imageutils
import rsgislib.vectorutils.createvectors
import rasterio
from rasterio.mask import mask
import shutil
import glob
from tqdm.contrib.concurrent import thread_map
from tqdm.contrib.concurrent import process_map
from itertools import repeat
import requests

def gen_aoi_mask(cls_img, folder):
    aoi_msk_img = f"{folder}/aoi_mask.kea"

    rsgislib.imageutils.gen_valid_mask(cls_img, aoi_msk_img, 'KEA', 0.0)

    out_vec_file = f"{folder}/valid_data_mask.gpkg"
    out_vec_lyr ='valid_data_mask'
    out_format = 'GPKG'
    pxl_val_fieldname = 'pxl_val'

    rsgislib.vectorutils.createvectors.polygonise_raster_to_vec_lyr(out_vec_file, out_vec_lyr, out_format, aoi_msk_img, pxl_val_fieldname=pxl_val_fieldname)

    aoi_mask = gpd.read_file(out_vec_file) # return geometries indicating aoi
    aoi_mask = aoi_mask[aoi_mask.pxl_val == 1]
    #aoi_mask = aoi_mask.dissolve()
    aoi_mask.to_file(out_vec_file)

    return out_vec_file

def msk_img_by_gpd(geometry, img, folder):
    out_img_path = f'{folder}/2019.kea'
    # make sure feature and img crs match
    #row.to_crs(2193, inplace=True)
    with rasterio.open(img) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "KEA",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(out_img_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return out_img_path

def return_initial_cls_img(gdf, cell_directory):
    """
    populate cell directory with initial classification 
    """
    classification_dir_path = f"{cell_directory}/classification"
    if not os.path.exists(classification_dir_path):
        os.makedirs(classification_dir_path)
    geom = gdf.geometry
    cls_img_path = "global-inputs/classification/2019-national-5cls-nztm-MMU1ha.kea"

    msk_img_by_gpd(geom, cls_img_path, classification_dir_path)

def replace_inf_with_nodata(file_path, nodata_value):
    # Open the input TIFF file
    with rasterio.open(file_path, 'r+') as src:
        # Loop through each band
        for band in range(1, src.count + 1):
            # Read the data for the current band
            data = src.read(band)
            
            # Replace -inf values with the nodata value
            data[np.isneginf(data)] = nodata_value
            
            # Write the modified data back to the same band
            src.write(data, band)
        
        # Update the metadata to include the new nodata value
        src.nodata = nodata_value

def convert_image(image, nodataVal, outFormat='KEA'):
    """
    function to convert all gdal supported image files in folder to specficed gdal_format format using gdal_translate

    Args
    folder - folder containing images to be converted
    inFormat - input file format
    outFormat - output format default = 'KEA'
    
    """
    replace_inf_with_nodata(image, nodataVal) # replace any -inf to nodata value
    out_img_path = f"{image[:-4]}.{outFormat.lower()}"
    # get band names
    bandNames = rsgislib.imageutils.get_band_names(image)
    gdal_translate = f"gdal_translate -of {outFormat} -a_nodata {nodataVal} {image} {out_img_path}"
    print(gdal_translate)
    os.system(gdal_translate)
    rsgislib.imageutils.set_band_names(out_img_path, bandNames)
    rsgislib.imageutils.pop_img_stats(out_img_path, True,nodataVal,True)
    os.remove(image) # remove .


def return_image_metadata(img_collection):
    # define metadata and img metadata
    metadata = {}
    img_list = []
    date_list = [] # date, time, lat, lon for tide level
    time_list = []
    lat_list = []
    lon_list = []
    cloud_list = []

    # add number of images to dict
    # return number of image in collection
    #metadata['number of images'] = img_collection.size().getInfo()

    # return image ids
    for i in img_collection.getInfo()['features']:
        # add to img_list
        img_list.append(i['id'])
        date_list.append(i['properties']['date_string'])
        time_list.append(i['properties']['interval_minutes'])
        lon_list.append(i['properties']['image_centroid_lon'])
        lat_list.append(i['properties']['image_centroid_lat'])
        cloud_list.append(round(float(i['properties']['region_cloudy_percent']), 3))

    # add image ids to dict
    metadata['image_id'] = img_list
    metadata['image_date'] = date_list
    metadata['image_time'] = time_list
    metadata['lon'] = lon_list
    metadata['lat'] = lat_list
    
    metadata['region_cloudy_percentage'] = cloud_list

    return metadata

def check_for_duplicate_images(image_directory):
    """
    function to remove images acquired on same date 
    """
    img_list = glob.glob(f"{image_directory}/*.kea")
    date_list = [i.split('_')[-1][:-4] for i in img_list]
    dup_list = []
    for i in date_list:
        n = date_list.count(i)
        if n > 1:
            if dup_list.count(i) == 0:
                dup_list.append(i)
    for d in dup_list:
        del_list = []
        for i in img_list:
            if d in i and not "S2" in i:
                print(f"Two images acquired on same day: deleting {i}")
                os.remove(i)

def return_tide_level_for_image(row):
    """
    function to return tide level for image using the niwa tide API and image metadata properties 
    adding tide relative to msl to image metadata.
    Args
    img - ee.Image object - all parameters acquired from image metadata properties
        - input_crs - from image_crs
        - lat - from image_centroid_coordinates
        - long - from image_centroid_coordinates
        - startDate - from date_string
        - interval - from interval_minutes
    """
    # define API params
    URL = "https://api.niwa.co.nz/tides/data"
    headers = {"x-apikey": open('niwakey').readline(),
               "Accept": "application/json"}

    # define parameters for tide api 
    parameters = {"lat": row.lat,
                  "long": row.lon,
                  "numberOfDays": 1, # only need to return tide for date & interval provided.
                  "startDate": row.image_date,
                  "datum": "MSL", # set to return tide relative to mean sea level
                  "interval": int(row.image_time)
    }
    
    # run in a while loop to in case rate limit exceeded on API
    total_retries = 3
    retries = 0
    while retries < total_retries:
        try:
            r = requests.get(URL, params=parameters, headers=headers) # get response from niwa tide API
            # print(r)
            if r.status_code == 429:
                sleep_seconds = int(10)
                # sleep for x seconds to refresh the count
                print(f'Num of API reqs exceeded, Sleeping for: {sleep_seconds} seconds...')
                time.sleep(sleep_seconds)
                retries += 1
            else:
                tide_level = r.json()['values'][0]['value'] # return tide_level from response
                row = row + (tide_level,)
                break
        except requests.exceptions.Timeout:
            print("request timed out. Consider handling this case.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    return row

def download_historical_imagery(gdf, down_dir):
    """
    function to download images in ee.ImageCollection\
    Args
    geodataframe - geodataframe containing geometry specifying region 
    down_dir - path to directory for downloaded files
    interval - number of weeks to search for imagery from today's date. 
    """
    cell_dir_path = f"{down_dir}/{gdf.cell_id.to_string(index=False)}"
    try:
        os.mkdir(cell_dir_path) 
    except:
        os.path.exists(cell_dir_path) == True

    return_initial_cls_img(gdf, cell_dir_path) # return initial classification image 
    cls_img_path = f"{cell_dir_path}/classification/2019.kea"
    gen_aoi_mask(cls_img_path, cell_dir_path) # return aoi mask 
    aoi_file_path = f"{cell_dir_path}/valid_data_mask.gpkg"

    
    img_dir_path = f"{cell_dir_path}/images" # create image directory
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    roi = featureutils.shp_to_featureCollection(aoi_file_path) ## return ee.FeatureCollection from shp

    s2 = imagecollectionutils.gen_s2_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=409,
        roi=roi)

    ls8 = imagecollectionutils.gen_ls_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=613,
        roi=roi,
        landsat_sensor_id='LS8')
    
    ls9 = imagecollectionutils.gen_ls_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=195,
        roi=roi,
        landsat_sensor_id='LS9')
    
    ls7 = imagecollectionutils.gen_ls_image_collection_for_region(
        date='2003-06-01',
        time_step=231,
        roi=roi,
        landsat_sensor_id='LS7',
        use_TOA=False)
    
    ls5 = imagecollectionutils.gen_ls_image_collection_for_region(
        date='2012-06-01',
        time_step=1474,
        roi=roi,
        landsat_sensor_id='LS5',
        use_TOA=False)
    
    for img_collection in s2, ls8, ls9, ls7, ls5:
        if img_collection.size().getInfo() != 0:
            
            print(f"number of cloud-free images to download:  {img_collection.size().getInfo()}")
    
            # img_collection = imagecollectionutils.add_tide_level_to_collection(img_collection, roi, multithreading=True)
    
            # calculate ndvi and mndwi
            img_collection = (img_collection.map(ndutils.apply_ndvi)
                            .map(ndutils.apply_mndwi))
            
            # return collection as list 
            img_list = img_collection.toList(img_collection.size().getInfo())
    
            def down_img_mt(img_id, img_collection_list, directory, roi, crs, scale, no_data_val):
                img = ee.Image(img_collection_list.get(img_id)).select(['ndvi', 'mndwi'])
                img = img.clip(roi).unmask(no_data_val) # clip img for export
                system_index = img.get("system:index").getInfo().split('_')[-3:]
                if len(system_index[-1]) == 6:
                    system_index = f"S2_{system_index[1][:8]}"
                else:
                    system_index = '_'.join([system_index[0], system_index[-1]]) 
                fn = f"{system_index}.tif"
                image_path = f"{directory}/{fn}"
                if os.path.exists(f"{image_path[:-4]}.kea"):
                    print(f"{fn} already downloaded.")
                else:
                    try:
                        imageutils.download_img_local(img.toFloat(), directory, fn, roi.geometry(), crs, scale)
                        convert_image(image_path, no_data_val, 'KEA')
                    except:
                        print(f"issue with {fn}, continuing")
            
            iterator = list(range(0, img_collection.size().getInfo()))
    
            thread_map(down_img_mt, iterator, repeat(img_list), repeat(img_dir_path), repeat(roi), repeat("EPSG:2193"), repeat(20), repeat(-99))
            print("images downloaded.")
            # return metadata_dict as json file 
            collection_metadata = return_image_metadata(img_collection)
            fn_meta = f"{cell_dir_path}/image_metadata.json"
            try: # check to see if metadata file exists
                with open(fn_meta, 'r') as existing_file:
                    existing_data = json.load(existing_file)
            except FileNotFoundError:
                existing_data = {}
            
            for k, v in collection_metadata.items(): # add new image metadata if metadata file already exists
                existing_data.setdefault(k, []).extend(v)
            
            meta_df = pd.DataFrame.from_dict(existing_data, orient='index').transpose() # read image_metadata as pandas df
            # drop duplicates based on date
            meta_df.drop_duplicates(subset=['image_date'], keep='first', inplace=True) 
            # dup_df = dup_df[dup_df.image_id.str.contains('S2')] # where there are duplicates kee
            # meta_df.drop_duplicates(subset=['image_date'], inplace=True, keep=False)
            # meta_df = pd.concat([meta_df, dup_df]).sort_values(by='image_date').reset_index(drop=True) 
            # add tide level
            with_tide_df = thread_map(return_tide_level_for_image, meta_df.itertuples(index=False)) 
            cols = list(meta_df.columns) + ['tide_level_msl']
            meta_df = pd.DataFrame(with_tide_df, columns=cols) # return df with tide level and write to json
            meta_dict = meta_df.to_dict(orient='list')
            with open(fn_meta, 'w') as file:
                file.write(json.dumps(meta_dict, indent=4))
    
        else: 
            print("Images contain too much cloud.")
    print("checking for duplicate images.")        
    check_for_duplicate_images(img_dir_path) # remove one of any images acquired on same day.



def download_images_in_collection(gdf, down_dir, interval=4):
    """
    function to download images in ee.ImageCollection\
    Args
    geodataframe - geodataframe containing geometry specifying region 
    down_dir - path to directory for downloaded files
    interval - number of weeks to search for imagery from today's date. 
    """
    cell_dir_path = f"{down_dir}/{gdf.cell_id.to_string(index=False)}"
    try:
        os.mkdir(cell_dir_path) 
    except:
        os.path.exists(cell_dir_path) == True

    aoi_file_path = f"{cell_dir_path}/valid_data_mask.gpkg" # get data mask from cell directory

    img_dir_path = f"{cell_dir_path}/images" # create image directory
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    roi = featureutils.shp_to_featureCollection(aoi_file_path) ## return ee.FeatureCollection from shp

    s2 = imagecollectionutils.gen_s2_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=interval,
        roi=roi)

    ls8 = imagecollectionutils.gen_ls_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=interval,
        roi=roi,
        landsat_sensor_id='LS8')
    
    ls9 = imagecollectionutils.gen_ls_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=interval,
        roi=roi,
        landsat_sensor_id='LS9')

    for img_collection in s2, ls8, ls9:
        if img_collection.size().getInfo() != 0:
            
            print(f"number of cloud-free images to download:  {img_collection.size().getInfo()}")

            # calculate ndvi and mndwi
            img_collection = (img_collection.map(ndutils.apply_ndvi)
                            .map(ndutils.apply_mndwi))
            
            # return collection as list 
            img_list = img_collection.toList(img_collection.size().getInfo())
    
            def down_img_mt(img_id, img_collection_list, directory, roi, crs, scale, no_data_val):
                img = ee.Image(img_collection_list.get(img_id)).select(['ndvi', 'mndwi'])
                img = img.clip(roi).unmask(no_data_val) # clip img for export
                system_index = img.get("system:index").getInfo().split('_')[-3:]
                if len(system_index[-1]) == 6:
                    system_index = f"S2_{system_index[1][:8]}"
                else:
                    system_index = '_'.join([system_index[0],system_index[-1]]) 
                fn = f"{system_index}.tif"
                image_path = f"{directory}/{fn}"
                imageutils.download_img_local(img.toFloat(), directory, fn, roi.geometry(), crs, scale)
                convert_image(image_path, no_data_val, 'KEA')
            
            iterator = list(range(0, img_collection.size().getInfo()))
    
            thread_map(down_img_mt, iterator, repeat(img_list), repeat(img_dir_path), repeat(roi), repeat("EPSG:2193"), repeat(20), repeat(-99))
            print("images downloaded.")
    
            # return metadata_dict as json file 
            collection_metadata = return_image_metadata(img_collection)
            fn_meta = f"{cell_dir_path}/image_metadata.json"
            try: # check to see if metadata file exists
                with open(fn_meta, 'r') as existing_file:
                    existing_data = json.load(existing_file)
            except FileNotFoundError:
                existing_data = {}
            
            for k, v in collection_metadata.items(): # add new image metadata if metadata file already exists
                existing_data.setdefault(k, []).extend(v)
            
            meta_df = pd.DataFrame.from_dict(existing_data, orient='index').transpose() # read image_metadata as pandas df
            # drop duplicates based on date
            meta_df.drop_duplicates(subset=['image_date'], keep='first', inplace=True)
            with_tide_df = thread_map(return_tide_level_for_image, meta_df.itertuples(index=False)) # add tide level
            cols = list(meta_df.columns) + ['tide_level_msl']
            meta_df = pd.DataFrame(with_tide_df, columns=cols) # return df with tide level and write to json
            meta_dict = meta_df.to_dict(orient='list')
            with open(fn_meta, 'w') as file:
                file.write(json.dumps(meta_dict, indent=4))
    
        else: 
            print("Images contain too much cloud.")
    print("checking for duplicate images.")        
    check_for_duplicate_images(img_dir_path) # remove one of images if acquired on same day.


def return_oldest_image(folder):
    """
    Returns the file path of the oldest image in a folder using the first part of the path before '_'.

    Args:
    folder - The folder to search for files

    Returns:
    The file path of the oldest image
    """
    # Get the list of files in the folder
    image_files = glob.glob(f"{folder}/*.kea")
    
    if not image_files:
        return None

    #print(files)

    date_strings = [f.split('_')[-1][:-4] for f in image_files] 
    oldest_date = min(datetime.strptime(date, "%Y%m%d") for date in date_strings)
    fp = [f for f in image_files if oldest_date.strftime("%Y%m%d") in f][0]
    
    return fp

def check_input_image(image, aoi_mask):
    array = rsgislib.imageutils.extract_img_pxl_vals_in_msk(image, [1], aoi_mask, 1)
    if -99 in array:
        print(f"{image} missing valid data, removing image.")
        os.remove(image)
        return False
    else:
        print(f"{image} valid.")
        return True

def run_change_detection(folder):
    """Wrapper function to apply change detection to directory containing inputs for a H3 cell
    """

    in_image_folder = f"{folder}/images" # define input image folder path

    tmp_dir_path = f"{folder}/tmp" # define tmp directory for outputs
    if not os.path.exists(tmp_dir_path):
        os.makedirs(tmp_dir_path)

    while True: # run process until all new images processed.
        input_image = return_oldest_image(in_image_folder) # return oldest image in folder
        if input_image is None: 
            return
        # check that input_image contains valid data across whole AOI
        aoi = f"{folder}/aoi_mask.kea"
        while not check_input_image(input_image, aoi):
            input_image = return_oldest_image(in_image_folder)
            # 
        ## CHANGE DETECTION ##
        # define class_img folder 
        class_folder_path = f"{folder}/classification"
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)

        #chg_detection_output_path = f"{folder}/change_detection_outputs" # define change outputs directory path

        # current_classifcation_image = glob.glob(f"{class_folder_path}/*.kea")[0] # get current class image from classification directory
        current_classifcation_image = f"{class_folder_path}/2019.kea"
        # define change detection args
        parameters = {
            "input_img": input_image,
            "class_img": current_classifcation_image,
            "out_folder": tmp_dir_path,
            "cell_id": f"{folder.split('/')[-1]}",
            "ndvi_band":1,
            "ndwi_band":2,
            "class_vals": {"sand": 1, "water": 2, "vegetation": 3}
        }

        # print(parameters)
        print(f"processing {input_image.split('/')[-1]}")
        workflows.return_change_output_otsu_merged_classes(**parameters)  # run change detection 
        # move image to archive 
        image_archive_dir = f"{in_image_folder}/archive"
        if not os.path.exists(image_archive_dir):
            os.makedirs(image_archive_dir)
        print(f"archiving input image: {image_archive_dir}/{input_image.split('/')[-1]}")
        shutil.move(input_image, f"{image_archive_dir}/{input_image.split('/')[-1]}")

        output_classification = f"{class_folder_path}/{input_image.split('_')[1][:8]}.kea"
        tools.return_new_class_image(tmp_dir_path, output_classification) # generate new classification image 
        print(f"new classification date: {output_classification}")

        ## BOUNDARY ANALYSIS ##
        # define inputs 
        ndvi_chg = f"{tmp_dir_path}/ndvi-chg.kea"
        ndwi_chg = f"{tmp_dir_path}/ndwi-chg.kea"

        img_date = input_image.split('_')[1][:8] # get image date

        out_dict = {} # dict to store variables
        out_dict['sensor'] = input_image.split('_')[0]
        out_dict['date'] = img_date

        # calc IW shoreline chg
        out_dict['IW_CDI'], out_dict['IW_shoreline_chg'] = tools.calc_cdi(ndwi_chg, 
            current_classifcation_image, tmp=tmp_dir_path, mask_vals=[3,4,5], eov_boundary=False)

        # calc EOV shoreline chg
        out_dict['EOV_CDI'], out_dict['EOV_shoreline_chg'] = tools.calc_cdi(ndvi_chg,
            current_classifcation_image, tmp=tmp_dir_path, mask_vals=[2,4,5], eov_boundary=True)

        # return change pixels, calc area change, total area for each class
        chg_pixels_img = f"{tmp_dir_path}/{img_date}-chg.kea"
        tools.return_chg_pixels(tmp_dir_path, current_classifcation_image, chg_pixels_img)
        out_dict['IW_area_change (Ha)'], out_dict['EOV_area_change (Ha)'] = tools.calc_area_change(chg_pixels_img)
        out_dict['sand_area (Ha)'], out_dict['water_area (Ha)'], out_dict['vegetation_area (Ha)'] = tools.return_new_class_area(output_classification)                                                                               

        # Check that water and sand were classified correctly
        if out_dict['sand_area (Ha)'] > out_dict['water_area (Ha)']: # if sand area is greater than water skip this iteration
            print(f"error with classification date: {output_classification}")
            print("removing classification on this iteration")
            os.remove(output_classification)

        else:
            # save results to .csv
            print(out_dict)
            new_results = pd.DataFrame.from_dict([out_dict])
            new_results['date'] = pd.to_datetime(new_results['date'])
            csv_fn = f"{folder}/cell_timeseries.csv"
    
            if not os.path.exists(csv_fn):
                new_results.to_csv(csv_fn)
            else: 
                df = pd.read_csv(csv_fn)
                df = pd.concat([df, new_results], ignore_index=True)
                df['date'] = pd.to_datetime(df['date'])
                df.drop_duplicates(subset='date', inplace=True)
                df.sort_values("date", inplace=True)
                #df.reset_index(drop=True, inplace=True)  # Reset the index to avoid extra index column
                df = df.round(2)
                df = df.drop_duplicates(subset='date', keep='last')  # Remove duplicates based on the 'date' column
                df.to_csv(csv_fn, index=False, sep=',')
            print(f"Change detection and boundary analysis for image acquired on: {img_date} complete.")
            print(f"Change analysis saved to {csv_fn}")
                
            # move current classification to archive folder
            # class_archive_dir = f"{class_folder_path}/archive"
            # if not os.path.exists(class_archive_dir):
            #     os.makedirs(class_archive_dir)
            # class_img_filename = current_classifcation_image.split('/')[-1] 
            # print(f"archiving classification from last iteration: {class_archive_dir}/{class_img_filename}")
            # shutil.move(current_classifcation_image, f"{class_archive_dir}/{class_img_filename}")

       
    shutil.rmtree(tmp_dir_path)  # remove tmp directory 

    
    












     


