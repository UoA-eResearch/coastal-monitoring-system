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
import ee
import tqdm
import json
import os
import rsgislib
import shutil
import glob
from tqdm.contrib.concurrent import thread_map
from tqdm.contrib.concurrent import process_map
from itertools import repeat



def convert_image(image, nodataVal, outFormat='KEA'):
    """
    function to convert all gdal supported image files in folder to specficed gdal_format format using gdal_translate

    Args
    folder - folder containing images to be converted
    inFormat - input file format
    outFormat - output format default = 'KEA'
    
    """
    #for f in tqdm.tqdm(glob.glob(folder + f'/*.{inFormat}')):
        #print(f)
    out_img_path = f"{image[:-4]}.{outFormat.lower()}"
    # get band names
    bandNames = rsgislib.imageutils.get_band_names(image)
    gdal_translate = f"gdal_translate -of {outFormat} -a_nodata {nodataVal} {image} {out_img_path}"
    print(gdal_translate)
    os.system(gdal_translate)
    rsgislib.imageutils.set_band_names(out_img_path, bandNames)
    rsgislib.imageutils.pop_img_stats(out_img_path, True,-99,True)
    os.remove(image) # remove .


def return_image_metadata(img_collection):
    # define metadata and img metadata
    metadata = {}
    img_list = []
    tide_list = []
    cloud_list = []

    # add number of images to dict
    # return number of image in collection
    #metadata['number of images'] = img_collection.size().getInfo()

    # return image ids
    for i in img_collection.getInfo()['features']:
        # add to img_list
        img_list.append(i['id'])
        tide_list.append(i['properties']['tide_level_msl'])
        cloud_list.append(round(float(i['properties']['region_cloudy_percent']), 3))

    # add image ids to dict
    metadata['image_id'] = img_list
    metadata['tide_level_MSL'] = tide_list
    metadata['region_cloudy_percentage'] = cloud_list

    return metadata

def download_images_in_collection(shp, down_dir):
    """
    function to download images in ee.ImageCollection\
    Args
    shp -shapefile specifying region 
    down_dir - path to directory for downloaded files
    """
    try:
        os.mkdir(f"{down_dir}") 
    except:
        os.path.exists(down_dir) == True

    # create image directory
    img_dir_path = f"{down_dir}/images"
    if not os.path.exists(img_dir_path):
        os.makedirs(img_dir_path)

    ## return ee.FeatureCollection from shp
    roi = featureutils.shp_to_featureCollection(shp)

    s2_img_collection = imagecollectionutils.gen_s2_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=52,
        roi=roi,
        cloud_cover=0.00
    )
    ls_sensors = ['LS8', 'LS9']
    for s in ls_sensors:
        img_collection = imagecollectionutils.gen_ls_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=52,
        roi=roi,
        landsat_sensor_id=s
        cloud_cover=0.00)
        s2_img_collection.merge(img_collection)

    img_collection_tide = imagecollectionutils.add_tide_level_to_collection(img_collection, roi, multithreading=True)
    
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
            fn = f"""{img.get("system:index").getInfo()}.tif"""
            image_path = f"{directory}/{fn}"
            imageutils.download_img_local(img.toFloat(), directory, fn, roi.geometry(), crs, scale)
            convert_image(image_path, no_data_val, 'KEA')
        
        iterator = list(range(0, img_collection.size().getInfo()))

        thread_map(down_img_mt, iterator, repeat(img_list), repeat(img_dir_path), repeat(roi), repeat("EPSG:2193"), repeat(20), repeat(-99))
        print("images downloaded.")

        # return metadata_dict as json file 
        collection_metadata = return_image_metadata(img_collection_tide)
        fn_meta = f"{down_dir}/image_metadata.json"
        try: # check to see if metadata file exsits
            with open(fn_meta, 'r') as existing_file:
                 existing_data = json.load(existing_file)
        except FileNotFoundError:
            existing_data = {}
        
        for k, v in collection_metadata.items(): # add new image metadata if metadata file already exists
                existing_data.setdefault(k, []).extend(v)
        
        with open(fn_meta, "w") as file:
                json.dump(collection_metadata, file)

    else: 
        print("Images contain too much cloud.")


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

    date_strings = [f.split('_')[1][:8] for f in image_files] 
    oldest_date = min(datetime.strptime(date, "%Y%m%d") for date in date_strings)
    fp = [f for f in image_files if oldest_date.strftime("%Y%m%d") in f][0]
    
    return fp

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
        
        ## CHANGE DETECTION ##
        # define class_img folder 
        class_folder_path = f"{folder}/classification"
        if not os.path.exists(class_folder_path):
            os.makedirs(class_folder_path)
            # clip classification by cell ROI and return to classification directory

        #chg_detection_output_path = f"{folder}/change_detection_outputs" # define change outputs directory path

        current_classifcation_image = glob.glob(f"{class_folder_path}/*.kea")[0] # get current class image from classification directory
        
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
        # boundary_analysis_folder_path = f"{folder}/boundary_analysis_outputs" # define class_img folder 
        # if not os.path.exists(boundary_analysis_folder_path):
        #     os.makedirs(boundary_analysis_folder_path)
        # define inputs 
        ndvi_chg = f"{tmp_dir_path}/ndvi-chg.kea"
        ndwi_chg = f"{tmp_dir_path}/ndwi-chg.kea"

        img_date = input_image.split('_')[1][:8] # get image date

        out_dict = {} # dict to store variables
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
            df.sort_values("date", inplace=True)
            #df.reset_index(drop=True, inplace=True)  # Reset the index to avoid extra index column
            df = df.round(2)
            df = df.drop_duplicates(subset='date', keep='last')  # Remove duplicates based on the 'date' column
            df.to_csv(csv_fn, index=False, sep=',')
        print(f"Change detection and boundary analysis for image acquired on: {img_date} complete.")
        print(f"Change analysis saved to {csv_fn}")
            
        # move current classification to archive folder
        class_archive_dir = f"{class_folder_path}/archive"
        if not os.path.exists(class_archive_dir):
            os.makedirs(class_archive_dir)
        class_img_filename = current_classifcation_image.split('/')[-1] 
        print(f"archiving classification from last iteration: {class_archive_dir}/{class_img_filename}")
        shutil.move(current_classifcation_image, f"{class_archive_dir}/{class_img_filename}")

       
        shutil.rmtree(tmp_dir_path)  # remove tmp directory 

    
    












     


