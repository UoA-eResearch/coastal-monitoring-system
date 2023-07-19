# import modules
import ee
import geemap

# authenticate ee
ee.Initialize()

def mask_clouds_LS_qa(image):
    """
    function to mask Landsat ee.image object using QA_pixel band from Fmask
    
    Args
    landsat ee.image object
    
    Returns
    landsat ee.image object with cloud masked
    """
    # define bit_masks
    shadow_bit_mask = (1 << 4)
    cloud_bit_mask = (1 << 3)
    dcloudBitMask = (1 << 1)
    # get qa image band
    qa = image.select('QA_PIXEL')

    # define mask
    mask = qa.bitwiseAnd(shadow_bit_mask).eq(0) \
        .And(qa.bitwiseAnd(cloud_bit_mask).eq(0)) \
        .And(qa.bitwiseAnd(dcloudBitMask).eq(0))
    
    return image.updateMask(mask)


def apply_scale_factors(image):
    """
    function to apply scale factors to landsat SR collection 2 image
    """

    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    return image.addBands(opticalBands, None, True) \
              .addBands(thermalBands, None, True)

def ps_pca(sensor, geometry):
    """
    function to pan sharpen LS7 and LS8 images with PCA method using geeSharp
    https://github.com/aazuspan/geeSharp/wiki/Sharpening-Functions#PCA

    args
    sensor - string defining whether its LS7 or LS8
    geometry - ee.Geometry defining region to calculate imagery stats for (will come from h3 cell)
    """
    # use geemap.require to get JS package
    geeSharp = geemap.require("users/aazuspan/geeSharp:geeSharp")

    # define pan band and bands to be sharpened


    
