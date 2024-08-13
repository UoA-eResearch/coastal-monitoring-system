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
from datetime import datetime
import ee
import tqdm
import json

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')

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
    ## return ee.FeatureCollection from shp
    roi = featureutils.shp_to_featureCollection(shp)

    img_collection = imagecollectionutils.gen_s2_image_collection_for_region(
        date=datetime.today().strftime('%Y-%m-%d'),
        time_step=460,
        roi=roi
    )

    img_collection_tide = imagecollectionutils.add_tide_level_to_collection(img_collection, roi, multithreading=True)
    
    if img_collection.size().getInfo() != 0:
        
        print(f"number of cloud-free images to download:  {img_collection.size().getInfo()}")

        # calculate ndvi and mndwi
        img_collection = (img_collection.map(ndutils.apply_ndvi)
                        .map(ndutils.apply_mndwi))
        
        # return collection as list 
        img_list = img_collection.toList(img_collection.size().getInfo())

        for i in tqdm.tqdm(range(img_collection.size().getInfo())): # iterate over images in list and download (this can be parallelised if needed)
            

            img = ee.Image(img_list.get(i)).select(['ndvi', 'mndwi']) # select required bands

            # define download parameters
            fn = f"""{img.get("system:index").getInfo()}.tif"""

            # download image 
            imageutils.download_img_local(img, down_dir, fn, roi.geometry(), 'EPSG:2193', 10)
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