# import modules
import ee 

def mask_clouds_S2_QA60(image):
    """function to mask Sentinel-2 ee.Image object using QA60 band

    Args
    image = ee.Image object

    Returns
    ee.image object with updated cloud mask
    """

    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # clear if both flags set to zero.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)


def mask_clouds_S2_probablity(probabilty_val):
    """
    function to mask sentinel-2 ee.image object using cloud probability band
    
    Returns
    Sentinel-2 ee.image object with updated cloud mask
    """
    def apply_mask(image):
        clouds = ee.Image(image.get('cloud_mask')).select('probability')
        isNotCloud = clouds.lt(probabilty_val)
        return image.updateMask(isNotCloud)
    return(apply_mask)

def add_cloud_bands_to_img(image):
    # define cloud probabilty threshold - values > threshold considered cloud
    cloud_threshold=60
    # get corresponding s2 cloud probability image for input image
    cld_prb_path = 'COPERNICUS/S2_CLOUD_PROBABILITY/{}'.format(image.id().getInfo())
    
    cld_prb = ee.Image(cld_prb_path)

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cloud_threshold).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return image.addBands(ee.Image([cld_prb, is_cloud]))

def add_cloud_bands_to_img_collection(cloud_prob_threshold=60):
    """
    function to add cloud bands to images in ee.ImageCollection as a .map() function
    args 
    cloud_prob_treshold - probability that pixel is cloudy if greater than treshold value default=60

    returns
    ee.ImageCollection with cloud band
    """
    
    # function to add cloud_band to images in ee.ImageCollection
    def add_cld_bands(image):
        # get cloud prob from cloud_mask img
        cld_prb = ee.Image(image.get('cloud_mask')).select('probability')

        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(cloud_prob_threshold).rename('clouds')

        # Add the cloud probability layer and cloud mask as image bands.
        return image.addBands(ee.Image([cld_prb, is_cloud]))
    return(add_cld_bands)


def add_shadow_bands_to_img_collection(nir_drk_thresh=0.15):
    """
    function to add shadow bands to images in ee.ImageCollection as a .map() function
    args 
    nir_drk_thresh - NIR threshold which is considered cloud 

    returns
    ee.ImageCollection with shadow bands
    """
    # define map function
    def add_shadow_bands(image):
        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = image.select('B8').lt(nir_drk_thresh*SR_BAND_SCALE).rename('dark_pixels')

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (image.select('clouds').directionalDistanceTransform(shadow_azimuth, 1*10)
            .reproject(**{'crs': image.select(0).projection(), 'scale': 100}) 
            .select('distance') 
            .mask()
            .rename('cloud_transform'))

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))
    return(add_shadow_bands)

 # define map function
def add_shadow_bands_to_img(image):
    nir_drk_thresh = 0.15
    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = image.select('NIR').lt(nir_drk_thresh*SR_BAND_SCALE).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (image.select('clouds').directionalDistanceTransform(shadow_azimuth, 1*10)
        .reproject(**{'crs': image.select(0).projection(), 'scale': 100}) 
        .select('distance') 
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return image.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask_to_img(img):
    # define buffer value to dilate edge of cloud objects in metres
    buffer = 50

    # Add cloud component bands.
    img_cloud = add_cloud_bands_to_img(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands_to_img(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(buffer*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def add_cld_shdw_mask_to_img_collection(buffer=50):
    """
    function to add cloud and shadow mask to sentinel image collection
    args
    buffer - value in metres that clouds will edges will be dilated by default=50

    returns
    sentinel ee.ImageCollection with cloud and shadow mask band
    """
    
    # define function to passed to ee.ImageCollection.map()
    def add_cloud_shdw_band(img):
        # Add cloud component bands.
        img_cloud = add_cloud_bands_to_img_collection(img)

        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands_to_img_collection(img_cloud)

        def add_band(img):
            # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
            is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

            # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
            # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
            is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(buffer*2/20)
                .reproject(**{'crs': img.select([0]).projection(), 'scale': 10})
                .rename('cloudmask'))

            # Add the final cloud-shadow mask to the image.
            return img_cloud_shadow.addBands(is_cld_shdw)
        return(add_band)
    # return add cloud shdw band function
    return(add_cloud_shdw_band)

def mask_clouds(image):
    # select cloudmask band and invert so clouds/shdw = 0 and valid pixels = 1
    not_cld_shdw = image.select('cloudmask').Not()

    # return image with cloud and shdw masked
    return image.updateMask(not_cld_shdw)


def join_S2_cld_prob(img_collection, roi, start_date, end_date):
    """
    function to join cloud probility and S2 image collection
    args
    img_collection - image collection cloud probaility will be joined to
    roi - region of interest
    start_date - start date of collection
    end_date - end date of collection

    returns
    ee.ImageCollection including cloud probability 
    """
    # define and filter cloud probability layer
    cld_prob_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                    .filterBounds(roi)
                    .filterDate(start_date, end_date))
    
    # join collections using 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('cloud_mask').apply(
        img_collection,
        cld_prob_col,
        ee.Filter.equals(leftField = 'system:index', rightField = 'system:index')
    ))

def add_cloud_shadow_mask(img):
    try:
        img.select("clouds") # get cloud band from image which is last band
    except:
        img = add_cloud_bands_to_img(img)

    img_cloud_shadow = add_shadow_bands_to_img(img) # add cloud shadow bands

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(50*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 10})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img_cloud_shadow.addBands(is_cld_shdw)

def return_cloud_pxl_count(img):
    """"
    function to return number of cloudy pixels in sentinel-2 image. Image must contain cloud mask band defined by cld_band_name
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
    acquisition_datetime = ee.String(img.get('system:index')).split('_').get(0)

    date_time = ee.String(acquisition_datetime).split('T') # split intro date and time 
    # return date as yyyy-mm-dd
    date = date_time.get(0) 
    y = ee.String(date).slice(0,4)
    m = ee.String(date).slice(-4, -2)
    d = ee.String(date).slice(-2)

    date_str = ee.String(y.cat('-').cat(m).cat('-').cat(d))

    # return hours and minutes as minutes 
    time = date_time.get(1)
    min = ee.Number.parse(ee.String(time).slice(2,4))
    hr = ee.Number.parse(ee.String(time).slice(0,2))
    hr_min = hr.multiply(ee.Number(60)).add(min)

    return img.set({
        'date_string': date_str,
        'interval_minutes': hr_min
    })








