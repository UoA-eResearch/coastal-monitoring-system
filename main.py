import ccdutils.monitoringutils as monitoringutils
import ee

# init gee with project 
ee.Initialize()


# download images
monitoringutils.download_images_in_collection("data/inputs/vector/test.gpkg", "data/inputs/raster")
