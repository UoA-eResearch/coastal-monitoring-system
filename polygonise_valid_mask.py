from rsgislib import imageutils
import rsgislib.vectorutils.createvectors

inputImage = 'data/85bb58c7fffffff/classification/archive/2019_classification_r10.kea'
outputImage = 'data/85bb58c7fffffff/valid_data_mask.kea'

imageutils.gen_valid_mask(inputImage, outputImage, 'KEA', 0.0)

out_vec_file = 'data/85bb58c7fffffff/valid_data_mask.gpkg'
out_vec_lyr ='valid_data_mask'
out_format = 'GPKG'

pxl_val_fieldname = 'pxl_val'

rsgislib.vectorutils.createvectors.polygonise_raster_to_vec_lyr(out_vec_file, out_vec_lyr, out_format, outputImage, pxl_val_fieldname=pxl_val_fieldname)