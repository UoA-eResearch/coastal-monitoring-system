# import modules
import ee

# Initialize GEE
ee.Initialize()

def apply_ndvi(image):
    """function to calculate ndvi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndvi  band
    """
    calc_ndvi = image.normalizedDifference(['NIR', 'red'])
    ndvi = calc_ndvi.select([0], ['ndvi'])
    
    return ee.Image.cat([image, ndvi])


def apply_ndwi(image):
    """function to calculate ndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndwi  band
    """
    calc_ndwi = image.normalizedDifference(['green', 'NIR'])
    ndwi = calc_ndwi.select([0], ['ndwi'])
    
    return ee.Image.cat([image, ndwi])

def apply_mndwi(image):
    """function to calculate mndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with mndwi  band
    """
    calc_mndwi = image.normalizedDifference(['green', 'SWIR1'])
    mndwi = calc_mndwi.select([0], ['mndwi'])
    
    return ee.Image.cat([image, mndwi])

def apply_ndmi(image):
    """function to calculate ndmi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with mndwi  band
    """
    calc_ndmi = image.normalizedDifference(['NIR', 'SWIR1'])
    ndmi = calc_ndmi.select([0], ['ndmi'])
    
    return ee.Image.cat([image, ndmi])


def apply_awei(image):
    """function to calculate ndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndwi  band
    """
    calc_awei = image.expression("4*(b('green')-b('SWIR1'))-(0.25*b('NIR')+2.75*b('SWIR2'))")
    awei = calc_awei.select([0], ['awei'])
    
    return ee.Image.cat([image, awei])

