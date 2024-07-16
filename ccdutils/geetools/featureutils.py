# import modules 
import ee
import geopandas as gpd
import json



def shp_to_featureCollection(shapefile):
    """
    function to read a shapefile as a ee.featureCollection using geopandas
    
    Args
    shapefile - path to shapefile to be read as featureCollection GEE object
    
    Returns
    ee.FeatureCollection object"""
    # read shapefile as gdf
    gdf = gpd.read_file(shapefile)
    # convert gdf to featureCollection with gdf_to_featureCollection
    return gdf_to_featureCollection(gdf)

    
def buffer(buffer):
    """
    function to buffer features in featureCollection
    Args 
    featureCollection - ee.object to be buffered 
    buffer - buffer distance in metres

    returns
    buffered ee.featureCollection
    """
    def apply_buffer(feature):
        return feature.buffer(buffer)
    
    return apply_buffer


def gdf_to_featureCollection(gdf):
    '''
    function to read a geopandas dataframe as a ee.featureCollection
    Args
    gdf - geopandas dataframe to be read as featureCollection GEE object
    
    Returns
    ee.FeatureCollection object
    '''
    # raise error if gdf is not LineString or Polygon
    valid_geometry = {'LineString','Polygon'}
    if gdf.geom_type[0] not in valid_geometry:
        raise ValueError('Shapefile must be LineString or Polygon.')

    #convert to json_dict
    if gdf.geom_type[0] == 'LineString':
        #gdf = gdf.buffer(1500)
        gdf = gdf.to_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    geo_json = gdf.to_json()
    json_dict = json.loads(geo_json)
    features = []
    # iterate over json features convert ee.Geometries and read as ee.Feature
    for feature in json_dict['features']:
        # derive ee.Geometry type from json_dict
        if feature['geometry']['type'] == 'LineString':
            line = ee.Feature(ee.Geometry.LineString(feature['geometry']['coordinates']))
            features.append(line.buffer(1500))
        if feature['geometry']['type'] == 'Polygon':
            features.append(ee.Feature(ee.Geometry.Polygon(feature['geometry']['coordinates'])))
    
    return ee.FeatureCollection(features)

def item_to_featureCollection(dict_item):
    """
    function to return ee.FeatureCollection from dict_item generated from pandas iterfeatures
    
    args
    dict_item - geodataframe item to be read as featureCollection

    returns 
    ee.FeatureCollection object
    """

    # raise error if gdf is not LineString or Polygon
    valid_geometry = {'LineString','Polygon','Point'}
    if dict_item['geometry']['type'] not in valid_geometry:
        raise ValueError('Shapefile must be LineString or Polygon.')

    features = []

    if dict_item['geometry']['type'] == 'Polygon':
        ee_geometry = ee.Geometry.Polygon(dict_item['geometry']['coordinates'])
    if dict_item['geometry']['type'] == 'LineString':
        ee_geometry = ee.Geometry.LineString(dict_item['geometry']['coordinates'])
    if dict_item['geometry']['type'] == 'Point':
        ee_geometry = ee.Geometry.Point(dict_item['geometry']['coordinates'])
    
    features.append(ee.Feature(ee_geometry))

    return ee.FeatureCollection(features)