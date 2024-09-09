import ccdutils.monitoringutils as monitoringutils
import ee
from datetime import datetime
import time
import glob
import os


# surpress TIFF warnings from gdal
from osgeo import gdal
# ... and suppress errors
gdal.SetConfigOption('CPL_LOG', '/dev/null')
os.environ["TQDM_DISABLE"] = "True"

# init gee with project 
ee.Initialize()


# Start time tracking
start_time = time.time()



monitoringutils.download_images_in_collection("data/85bb58c7fffffff/valid_data_mask_cell.gpkg", "data/85bb58c7fffffff") # Download images

monitoringutils.run_change_detection("data/85bb58c7fffffff") # run change detection





# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")

