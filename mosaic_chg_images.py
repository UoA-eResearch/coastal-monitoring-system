import rsgislib.imageutils
import rsgislib.classification
import rsgislib.rastergis
from geeutils import image_utils
from osgeo import gdal
from rios import rat
import numpy as np
import glob


# generate list of imgs
img_list = glob.glob('/Users/ben/Desktop/national-scale-change/HR6/data/*/boundary-analysis/outputs/2022-chg-pixels.kea',
                      recursive =True)

print(len(img_list))

out_img = '/Users/ben/Desktop/national-scale-change/HR6/2022-chg-pixels.kea'

out_img_3857 = '/Users/ben/Desktop/national-scale-change/validation/HR6/2022-chg-pixels-3857.kea'

image_utils.mosaic(img_list, out_img, nodataVal=0)
rsgislib.imageutils.pop_img_stats(out_img,True,0,True)
rsgislib.imageutils.gdal_warp(out_img,out_img_3857, 3857, gdalformat='KEA')
rsgislib.imageutils.pop_img_stats(out_img_3857,True,0,True)

# populate classification with class names for validation
rsgislib.rastergis.pop_rat_img_stats(out_img_3857, add_clr_tab=True, calc_pyramids=True, ignore_zero=True)

    
ratDataset = gdal.Open(out_img_3857, gdal.GA_Update)
red = rat.readColumn(ratDataset, 'Red')
green = rat.readColumn(ratDataset, 'Green')
blue = rat.readColumn(ratDataset, 'Blue')
ClassName = np.empty_like(red, dtype=np.dtype('a255'))
ClassName[...] = ""


red[1] = 51
blue[1] = 153
green[1] = 255
ClassName[1] = 'sand-water'

red[2] = 255
blue[2] = 255
green[2] = 51
ClassName[2] = 'water-sand'

red[3] = 0
blue[3] = 204
green[3] = 0
ClassName[3] = 'Vegetation-sand'

red[4] = 0
blue[4] = 5
green[4] = 61
ClassName[4] = 'sand-vegetation'

red[5] = 150
blue[5] = 75
green[5] = 45
ClassName[5] = 'no change'

rat.writeColumn(ratDataset, 'Red', red)
rat.writeColumn(ratDataset, 'Green', green)
rat.writeColumn(ratDataset, 'Blue', blue)
rat.writeColumn(ratDataset, 'ClassName', ClassName)
ratDataset = None

# generate accuracy points
rsgislib.classification.generate_stratified_random_accuracy_pts(
    input_img=out_img_3857,
    out_vec_file='/Users/ben/Desktop/national-scale-change/validation/HR6/accuracy-points-2022.gpkg',
    out_vec_lyr='accuracy-points',

    out_format='GPKG',
    rat_class_col='ClassName',
    vec_class_col='cls-pts',
    vec_ref_col='ref-pts',
    num_pts=1000,
    seed=35,
    del_exist_vec=True
    )