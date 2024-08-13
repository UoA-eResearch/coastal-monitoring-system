# import modules
import ee
import ccdutils.geetools.featureutils as featureutils
import ccdutils.geetools.imageutils as imageutils
import ccdutils.geetools.s2utils as s2utils
import ccdutils.geetools.lsutils as lsutils
from datetime import datetime, timedelta
import tqdm
from tqdm.contrib.concurrent import thread_map
from itertools import repeat


# define global variables
# define valid sensors
valid_optical_sensors = {'S2', 'LS4', 'LS5', 'LS7', 'LS8', 'LS9'}
valid_sar_sensors = {'S1'}
# dict containing sensor optical image bands 
img_bands = {'S2': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A', 'B11', 'B12'],
        'LS4': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        'LS4_sr': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        'LS5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        'LS5_sr': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        'LS7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        'LS7_sr': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        'LS8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        'LS8_sr': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        'S1': ['HH', 'HV', 'VV', 'VH', 'angle']}

# dict containing sensor GEE snippets for optical collections store SR and TOA collections as list
sensor_id = {'S2': ['COPERNICUS/S2_SR_HARMONIZED', 'COPERNICUS/S2_HARMONIZED'],
        'LS4': ['LANDSAT/LT04/C02/T1_L2', 'LANDSAT/LT04/C02/T1_TOA'],     
        'LS5': ['LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LT05/C02/T1_TOA'],
        'LS7': ['LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LE07/C02/T1_TOA'],
        'LS8': ['LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC08/C02/T1_TOA'],
        'LS9': ['LANDSAT/LC09/C02/T1_L2', 'LANDSAT/LC09/C02/T1_TOA'],
        'S1': ['COPERNICUS/S1_GRD']} 

# list of band names
band_names = ['blue', 'green', 'red', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'SWIR1', 'SWIR2']


def rename_img_bands(sensor):
    """function to rename optical image bands for ee.Image in ee.ImageCollection when using .map function
    
    Args
    sensor -  string sensor type that bands are being renamed
    
    returns 
    ee.Image with renamed bands
    """

    def rename(image):
        bands = img_bands[sensor]
        names = []
        if sensor == 'S2':
                names = band_names
        else:
                names = band_names[:3] + band_names[6:7] + band_names[-2:]
        
        return image.select(bands).rename(names)
    
    return(rename)


def gen_imageCollection_from_shp(year, region_shp, sensor):
    """
    function that returns annual ee.ImageCollection for Landsat or Sentinel surface reflectance and top-of-atmosphere images.  
    
    Args
    year - year as integer eg. 2019
    sensor - sensor type to build composite image as string (S2, LS7, LS8)
    region_shp - shapefile defining region for composite, accepts polygons and lines, if polyline representing coastline output will be coast
            zone defined as 3km buffer zone around coastline
    
    returns
    ee.ImageCollection object for specified sensor, region and year
    """
    
    # define date ranges 
    start_date = str(year) + '-01-01'
    # if sensor = LS4 composite is from 1988 - 1990
    if sensor == 'LS4':
           end_date = f'{year + 2}-01-01'
    else:
        end_date = str(year + 1) + '-01-01'

    # raise error if sensor isn't compatible
    if sensor not in valid_optical_sensors:
            raise ValueError(sensor + ' is not compatible, must be S2, LS4, LS5, LS7 or LS8.')
    
    print("Generating composite image for {} for {}".format(sensor, year))

    # convert region to ee.featureCollection 
    roi = featureutils.shp_to_featureCollection(region_shp)

    # define sr image collection
    collection = ee.ImageCollection(sensor_id[sensor][0]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    return collection

def gen_imageCollection(year, roi, sensor, cloud_cover=None, surface_reflectance=True):
        """
        function that returns annual ee.ImageCollection for Landsat or Sentinel surface reflectance and top-of-atmosphere images.  

        Args
        year - year as integer eg. 2019
        sensor - sensor type to build composite image as string (S2, LS7, LS8)
        roi - ee.featureCollection object defining region of interest
        cloud_cover - integer representing cloud cover % for scenes to be included. Default=None and all scenes are considered. 

        returns
        ee.ImageCollection object for specified sensor, region and year
        """

        # define date ranges 
        start_date = str(year) + '-01-01'
       # if sensor = LS4 composite is from 1988 - 1990
        if sensor == 'LS4':
                end_date = f'{year + 2}-01-01'
        else:
                end_date = str(year + 1) + '-01-01'

        # raise error if sensor isn't compatible
        if sensor not in valid_optical_sensors:
                raise ValueError(sensor + ' is not compatible, must be S2, LS4, LS5, LS7 or LS8.')

        #print("Generating composite image for {} for {}".format(sensor, year))

        # define sr image collection
        collection = ee.ImageCollection(sensor_id[sensor][0]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        
        # perform sentinel cloudmasking 
        if sensor == 'S2':
                # filter collection by cloud cover if cloud_cover is not none
                if cloud_cover is not None: 
                        collection = collection.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_cover)
                
                # join sentinel cloud probabilty
                img_collection = s2utils.join_S2_cld_prob(collection, roi, start_date, end_date)

                # map cloud masking workflow over collection
                img_collection = (img_collection 
                        # add is_clouds band
                        .map(s2utils.add_cloud_shadow_mask) 
                        # add cloud_shdw_mask # use default buffer value (50m)
                        # mask clouds
                        .map(s2utils.mask_clouds)
                        # rename bands
                        .map(rename_img_bands(sensor)))
                
        # perform landsat cloudmasking
        else:
                # add SR if surface_reflectance=True for band names
                if surface_reflectance == True:
                       sensor = f'{sensor}_sr'
                else:
                       sensor = sensor
                # filter collection by cloud cover if cloud_cover is not none
                if cloud_cover is not None: 
                        collection = collection.filterMetadata('CLOUD_COVER', 'less_than', cloud_cover)
                
                # run landsat cloudmasking and rename bands
                img_collection = (collection 
                        .map(lsutils.mask_clouds_LS_qa) 
                        .map(rename_img_bands(sensor)))

        return img_collection

def return_least_cloudy_image(year, roi, sensor, cloud_cover=None, return_least_cloudy=True):
        """
        function that returns annual ee.ImageCollection for Landsat or Sentinel surface reflectance and top-of-atmosphere images.  

        Args
        year - year as integer eg. 2019
        sensor - sensor type to build composite image as string (S2, LS7, LS8)
        roi - ee.featureCollection object defining region of interest
        cloud_cover - integer representing cloud cover % for scenes to be included. Default=None and all scenes are considered. 

        returns
        ee.ImageCollection object for specified sensor, region and year
        """

        # define date ranges 
        start_date = str(year) + '-01-01'
        end_date = str(year + 1) + '-01-01'

        # raise error if sensor isn't compatible
        if sensor not in valid_optical_sensors:
                raise ValueError(sensor + ' is not compatible, must be S2, LS5, LS7 or LS8.')

        #print("Generating composite image for {} for {}".format(sensor, year))

        # define sr image collection
        collection = ee.ImageCollection(sensor_id[sensor][1]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        
        if sensor == 'S2':
                if cloud_cover is not None: 
                        collection = collection.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_cover) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE', return_least_cloudy) \
                        .map(rename_img_bands(sensor))
        else:
                if cloud_cover is not None: 
                        collection = collection.filterMetadata('CLOUD_COVER', 'less_than', cloud_cover) \
                        .sort('CLOUD_COVER', return_least_cloudy) \
                        .map(rename_img_bands(sensor))

        return ee.Image(collection.first())

def gen_s2_image_collection_for_region(date, time_step, roi, cloud_cover=0.10, cloud_prob_score=60):
        """
        function to return s2 surface reflectance image collection defined by region where cloud cover is calculated for region.
        Args
        date - date as yyyy-mm-dd 
        time_step - integer indicating number of weeks from date to filter the image collection
        roi - ee.featureCollection object defining region 
        cloud_cover - float representing % cloud cover for region to be included in collection default=10
        cloud_prob_score - integer representing value at which a pixel is considered cloud in sentinel-2 cloud probablity mask
        pxl_size - 
        """
        # define dates as str
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(weeks=time_step)).strftime("%Y-%m-%d")

        print(f"Looking for images between {start_date} and {date}")

        # define image collection
        # # define sr image collection
        img_collection = (ee.ImageCollection(sensor_id['S2'][0]) 
                .filterBounds(roi) 
                .filterDate(start_date, date))

        # join sentinel cloud probabilty image and add cloud band 
        img_collection = (s2utils.join_S2_cld_prob(img_collection, roi, start_date, date)
                        .map(rename_img_bands('S2')) # select required bands and rename
                        .map(s2utils.add_cloud_bands_to_img_collection(cloud_prob_score)) # adds cloud band based on cloud_prob_score
                        .map(imageutils.clip_images_to_region(roi))) # clip to region/cell
        
        print(f"number of available images: {img_collection.size().getInfo()}")

        # calc pxl counts for cloud mask and total region
        img_collection = (img_collection.map(s2utils.return_region_pxl_count)
                        .map(s2utils.return_cloud_pxl_count)
                        .map(s2utils.add_cell_level_cloud_cover_property)
                        .filterMetadata('region_cloudy_percent', 'less_than', cloud_cover)) # filter collection by region_cloudy_percent
        
        print(f"number of cloud-free images: {img_collection.size().getInfo()}")

        # perform cloud masking
        img_collection = (img_collection 
                # add is_clouds band
                .map(s2utils.add_cloud_shadow_mask) 
                # add cloud_shdw_mask # use default buffer value (50m)
                # mask clouds
                .map(s2utils.mask_clouds))
        return img_collection

def add_tide_level_to_collection(ee_image_collection, roi, multithreading=False):
        """
        function to return tide level relative to MSL for all images in image colletion using the Niwa tide API
        """

        img_collection = (ee_image_collection.map(s2utils.add_roi_centroid_image_property(roi))
                        .map(s2utils.return_image_acquisition_time))
        
        # add tide level to image
        img_list = img_collection.toList(img_collection.size().getInfo()) # needs to be run as for loop due to mixing client/server operations

        img_collection_tide_list = ee.List([]) # define empty ee.List

        if multithreading == True:
               def add_tide_level_to_collection_mt(img_id, list_of_images):
                        img = ee.Image(list_of_images.get(img_id))
                        img = imageutils.return_tide_level_for_image(img)
                        return img
               iterable = list(range(0, img_collection.size().getInfo()))
               results = thread_map(add_tide_level_to_collection_mt, iterable, repeat(img_list))
               img_collection_tide_list = ee.List(results)
        else:
                for i in tqdm.tqdm(range(img_collection.size().getInfo())): # iterate over images in list to return tide level
                        img = ee.Image(img_list.get(i))
                        img = imageutils.return_tide_level_for_image(img)
                        img_collection_tide_list = img_collection_tide_list.add(img) # add to ee.List to generate collection with tide level property 

        return ee.ImageCollection(img_collection_tide_list) # redefine image collection

  
