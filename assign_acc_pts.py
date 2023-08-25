# Import the rsgislib classification module
import rsgislib.classification

# Import rsgislib vectorutils module 
import rsgislib.vectorutils

# Import the rsgislib tools plotting module for visualisation
import rsgislib.tools.plotting

# Import the rsgislib tools utils module
import rsgislib.tools.utils

from rsgislib import imagecalc



import geopandas as gpd


# define inputs
HR6_chg_img = '/Users/ben/Desktop/national-scale-change/validation/HR5/2014/2014-chg-pixels-3857.kea'

ref_pts = '/Users/ben/Desktop/national-scale-change/validation/HR5/2014/accuracy-points.gpkg'
ref_lyr = 'accuracy-points'

#points_gdf = gpd.read_file(ref_pts)

ref_cpy = f'{ref_pts[:-5]}-copy.gpkg'
ref_cpy_lyr = f'{ref_lyr}-copy'

# copy points to avoid overwriting
# rsgislib.vectorutils.vector_translate(ref_pts, ref_lyr, ref_cpy, 
#                                       ref_cpy_lyr, out_format='GPKG', del_exist_vec=True)

# populate reference points
# rsgislib.classification.pop_class_info_accuracy_pts(
#     input_img=HR6_chg_img,
#     vec_file=ref_cpy,
#     vec_lyr=ref_cpy_lyr,
#     rat_class_col="ClassName",
#     vec_class_col="HR6_chg",
#     vec_ref_col=None,
#     vec_process_col=None,
# )

# read pts as gdf
ref_gdf = gpd.read_file(ref_cpy)

vec_refpts_vld_file = '/Users/ben/Desktop/national-scale-change/validation/HR6/2014-ref-points.gpkg'
vec_refpts_vld_lyr = '2014-ref-points'

points_gdf = ref_gdf.drop(ref_gdf[ref_gdf["HR6_chg"] == "NA"].index)
points_gdf.to_file(vec_refpts_vld_file, driver="GPKG")

# # generate dict of area values for classes incl other change 
# # count pixels for each class in image
# counts = imagecalc.count_pxls_of_val(HR6_chg_img, [1,2,3,4, 5], 1)

# area_dict = {'sand-water': counts[0]*400,
#              'water-sand': counts[1]*400,
#              'Vegetation-sand': counts[2]*400,
#              'sand-vegetation': counts[3]*400,
#              'no change': counts[4]*400,
#              'other change': len(points_gdf[points_gdf['ref-pts'] == 'other change'])*400/10000}

# print(area_dict)

# Import the calc_acc_metrics_vecsamples function from the 
# rsgislib.classification.classaccuracymetrics module.
from rsgislib.classification.classaccuracymetrics import calc_acc_metrics_vecsamples_img


out_json_file = '/Users/ben/Desktop/national-scale-change/validation/HR5/2014-accuracy-metrics.json'
out_csv_file = '/Users/ben/Desktop/national-scale-change/validation/HR5/2014-accuracy-metrics.csv'

calc_acc_metrics_vecsamples_img(
    vec_file=ref_cpy,
    vec_lyr=ref_cpy_lyr,
    ref_col="ref-pts",
    cls_col="cls-pts",
    cls_img=HR6_chg_img,
    img_hist_col = 'Histogram',
    out_json_file=out_json_file,
    out_csv_file=out_csv_file,
)

