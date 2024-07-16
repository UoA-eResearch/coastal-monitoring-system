# import modules
import ee
import ccdutils.geetools.featureutils as featureutils
import ccdutils.geetools.imageutils as imageutils
import ccdutils.geetools.s2utils as s2_utils
import ccdutils.geetools.lsutils as lsutils

# Initialize GEE
ee.Initialize()

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
sensor_id = {'S2': ['COPERNICUS/S2_SR', 'COPERNICUS/S2'],
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
                img_collection = s2_utils.join_S2_cld_prob(collection, roi, start_date, end_date)

                # map cloud masking workflow over collection
                img_collection = (img_collection 
                        # add is_clouds band
                        .map(s2_utils.add_cloud_shadow_mask) 
                        # add cloud_shdw_mask # use default buffer value (50m)
                        # mask clouds
                        .map(s2_utils.mask_clouds)
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