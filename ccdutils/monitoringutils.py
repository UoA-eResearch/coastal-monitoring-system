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

# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')

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
        time_step=3,
        roi=roi
    )
    
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

    else: 
        print("Images contain too much cloud.")