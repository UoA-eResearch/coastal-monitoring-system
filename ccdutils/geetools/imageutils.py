# import modules
from logging import exception
from tabnanny import check
import ee
import time
import os
import requests
from osgeo import gdal 
import rsgislib
import rsgislib.imageutils
from pyproj import Transformer
import json

def rename_img_bands(img_bands, band_names):
    """function to rename optical image bands for ee.Image in ee.ImageCollection when using .map function
    
    Args
    img_bands - list, list of band values to be renamed
    band_names - list, list of band names
    
    returns 
    ee.Image with renamed bands
    """

    def rename(image):
        bands = img_bands
        return image.select(bands).rename(band_names)
    
    return(rename)

def resample_image(image, crs='EPSG: 2193', pixel_size=20):
    """
    function to resample ee.image object 
    
    Args
    image - ee.image object to be resampled
    crs - reference system as epsg code
    pixel_size - resampled pixel size"""

    bands = image.bandNames()
    resampled_bands = image.select(bands).reproject({'crs': crs, 'scale': pixel_size})
    return resampled_bands

def run_task(task, mins):
    """
    function to run ee.batch.export and check status of task every number of minutes specified by mins
    """

    secs = mins * 60
    task.start()
    while task.active():
        print(task.status())
        time.sleep(secs)

def set_band_names(image, band_names):
    """
    Function to set band names
    :param image: input image
    :param band_names: list of band names
    """
    data = gdal.Open(image, gdal.GA_Update)
    for i in range(len(band_names)):
        band = i + 1
        bandName = band_names[i]

        imgBand = data.GetRasterBand(band)
        # Check the image band is available
        if not imgBand is None:
            imgBand.SetDescription(bandName)
        else:
            raise exception("Could not open the image band: ", band)


def download_img_local(ee_image, folder, name, region, crs, scale, format='GEO_TIFF'):
    """
    function to get download url from ee.Image
    
    Args
    ee_image - ee.Image object
    folder - local folder name to save image to
    name - file_name
    region - extent of image
    scale - output_resolution
    format - image format default GEO_TIFF

    Returns
    image downloaded to local folder specified
    """
    
    # get bandnames
    bands = ee_image.bandNames().getInfo()

    # join folder and file name
    down_path = os.path.join(folder, name)
    
    # define params to be parsed to getDownloadUrl
    params = {
        'bands': bands,
        'region': region,
        'crs': crs,
        'scale': scale,
        'format': format
    }
    
    # define url
    try: 
        url = ee_image.getDownloadUrl(params)
    except Exception as e:
        print('Error occurred during download.')
        print(e)
        return

    # get url 
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print('Error occurred during download.')
        print(response.json()["error"]["message"])
        return

    # download file 
    with open(down_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=1024):
            fd.write(chunk)

    # set band names
    set_band_names(down_path, bands)


def set_nodata_val(image, no_data_val):
    """
    function to set no data value for image
    
    Arg
    image - str, filepath to image requiring no_data value
    no_data_val - int, no data value to set
    """
    ds = gdal.OpenEx(image, gdal.GA_Update)
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i + 1).SetNoDataValue(no_data_val)

def mosaic(inputImageList, output_image, nodataVal=-99, outputFormat='KEA', dataType=rsgislib.TYPE_32FLOAT):
    """
    function to moasaic group of images in specified folder and populate image with statistics

    Args

    inputImageList - list of images to be mosaiced
    output_image - str, file path for output image
    nodataVal - float, output no data value
    innodataVal - float, no data value for input images
    band - int, input image band to get no data value from
    outputFormat - str, containing output image format default=KEA
    datatype - rsgislib datatype default=rsgislib.TYPE_32FLOAT
    inputFN - str, if images to be merged are in different subdirectories with same filename default=None 
    
    Returns

    outputImage - mosaiced image
    """
    
    # get band names
    bandNames = rsgislib.imageutils.get_band_names(inputImageList[0])
    print(bandNames)
    # define reamaining args
    overlapBehaviour = 2   
    skipBand = 1
    # mosaic with imageutils.createImageMosaic and populate image stats
    innodataVal = nodataVal
    rsgislib.imageutils.create_img_mosaic(inputImageList, output_image, nodataVal, innodataVal, skipBand, overlapBehaviour, outputFormat, dataType)
    rsgislib.imageutils.set_band_names(output_image, bandNames)
    set_nodata_val(output_image, nodataVal)

def clip_images_to_region(region):
    """
    function to clip images in ee.ImageCollection using .map() to region defined by featureCollection
    Args
    region - ee.featureCollection representing the region to be clipped to. 
    """
    def clip(img):
        return img.clipToCollection(region)
    return(clip)

def transform_coordinates(input_coordinates, input_crs, target_crs="EPSG:2193"):
    """
    function to transform and return latitude, longitude coordinates from one CRS to another
    Args
    input_coordinates - list containing latitude, longitude to be transformed
    input_crs - crs of input coordinates
    target_crs - crs of returned latitude, longitude 
    Returns
    latitude, longitude in target_crs
    """
    transform = Transformer.from_crs(input_crs, target_crs) # tranformer from pyproj

    lon, lat = transform.transform(input_coordinates[0], input_coordinates[1])

    return lon, lat

def return_tide_level_for_image(img, API_KEY):
    """
    function to return tide level for image using the niwa tide API and image metadata properties 
    and return tide level as image property 
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
    headers = {"x-apikey": API_KEY,
               "Accept": "application/json"}

    # define parameters for tide api 
    parameters = {"lat": img.get('image_centroid_lat').getInfo(),
                  "long": img.get('image_centroid_lon').getInfo(),
                  "numberOfDays": 1, # only need to return tide for date & interval provided.
                  "startDate": img.get('date_string').getInfo(),
                  "datum": "MSL", # set to return tide relative to mean sea level
                  "interval": int(img.get('interval_minutes').getInfo())
    }
    
    #print(parameters)

    # run in a while loop to in case rate limit exceeded on API
    total_retries = 3
    retries = 0
    while retries < total_retries:
        try:
            r = requests.get(URL, params=parameters, headers=headers) # get response from niwa tide API
            # print(r)
            if r.status_code == 429:
                sleep_seconds = int(20)
                # sleep for x seconds to refresh the count
                print(f'Num of API reqs exceeded, Sleeping for: {sleep_seconds} seconds...')
                time.sleep(sleep_seconds)
                retries += 1
            else:
                tide_level = r.json()['values'][0]['value'] # return tide_level from response
                img = img.set("tide_level_msl", ee.Number(tide_level))
                break
        except requests.exceptions.Timeout:
            print("request timed out. Consider handling this case.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    return img

def return_cloud_pxl_count(img):
    """"
    function to return number of cloudy pixels in sentinel-2 image. Image must contain cloud mask band where band name = 'cloud'
    """
    cloud_mask = img.select('clouds') # cloud band is called clouds
    band =  img.bandNames().get(0) # define first band name from image
    # updateMask to ensure only count of cloudy pixels are returned
    mask_img = img.select([band]).updateMask(cloud_mask)
    pxl_count = mask_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=img.geometry(),
        maxPixels=1e10
    )
    return img.set('mask_pixel_count', ee.Number(pxl_count.get(band)))

def return_region_pxl_count(img):
    """
    function to return number of pixels in image, based on one band in image defined by band_name. 
    """
    # def calc_pxl_count(img):
    band = img.bandNames().get(0) # define first band name from image
    pxl_count = img.select([band]).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=img.geometry(),
        maxPixels=1e10
    )
    return img.set('total_pixel_count', ee.Number(pxl_count.get(band)))

def add_cell_level_cloud_cover_property(img):
    """
    function to return cell level cloud cover as image metadata property.
    """
    return img.set('region_cloudy_percent', ee.Number(img.get('mask_pixel_count')).divide(ee.Number(img.get('total_pixel_count'))))

def add_roi_centroid_image_property(roi):
    """
    add image crs and image centroid lat long from roi 
    """
    def add_centroid(img):
        band = img.bandNames().get(0) # get first band name from image
        img_band = img.select([band]) # select first image band to get CRS
        centroid = roi.geometry().centroid()

        return img.set({"image_centroid_lon": ee.List(centroid.coordinates()).get(0),
                        "image_centroid_lat": ee.List(centroid.coordinates()).get(1),
                        "image_crs": ee.String(img_band.projection().crs())
                        })
    return(add_centroid)

def return_image_acquisition_time(img):
    """
    return image accquisition date and time (UTC) as image properties in order to return the tide at time of image acquisition from NIWA tide api 
    date format is string 'yyyy-mm-dd'
    time expressed as total minutes
    """
    timestamp = ee.Date(img.get('system:time_start'))
    hour = timestamp.get('hour')
    minute = timestamp.get('minute')
    hr_min = ee.Number(hour).multiply(ee.Number(60)).add(ee.Number(minute))
    
    date_str = ee.String(ee.String(timestamp.format('yyyy-MM-dd')))


    return img.set({
        'date_string': date_str,
        'interval_minutes': hr_min
    })