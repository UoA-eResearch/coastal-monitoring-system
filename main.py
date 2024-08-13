import ccdutils.monitoringutils as monitoringutils
import ee
from datetime import datetime
import time

# init gee with project 
ee.Initialize()


# Start time tracking
start_time = time.time()

# Download images
monitoringutils.download_images_in_collection("data/inputs/vector/test.gpkg", "data/inputs/raster/pre_24-08-12")

# End time tracking
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time:.2f} seconds")

