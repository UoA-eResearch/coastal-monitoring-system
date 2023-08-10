from boundary_analysis import boundary_functions
from geeutils import image_utils

import glob


# generate list of imgs
img_list = glob.glob('/cd-data/HR5/*/boundary-analysis/tmp/2014-chg-pixels.kea', recursive =True)

print(len(img_list))

out_img = '/cd-data/HR5-2014-chg-pixels.kea'

image_utils.mosaic(img_list, out_img)