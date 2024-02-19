# Import the rsgislib classification module
import rsgislib.classification

# Import rsgislib vectorutils module 
import rsgislib.vectorutils

# Import the rsgislib tools plotting module for visualisation
import rsgislib.tools.plotting

# Import the rsgislib tools utils module
import rsgislib.tools.utils

from rsgislib import imagecalc

# Import the calc_acc_metrics_vecsamples function from the 
# rsgislib.classification.classaccuracymetrics module.
from rsgislib.classification.classaccuracymetrics import calc_acc_metrics_vecsamples_img


import geopandas as gpd


# define inputs
HR6_sentinel_chg_img = '/Users/ben/Desktop/national-scale-change/HR6-run-2/2022-chg-pixels-3857.kea'

HR6_ls_chg_img = '/Users/ben/Desktop/national-scale-change/HR5/cell-chg-pixels-3857-aoi-masked.kea'

sentinel_points = '/Users/ben/Desktop/national-scale-change/validation/HR5/sentinel-run-2-points.gpkg'

ls_points = '/Users/ben/Desktop/national-scale-change/validation/HR5/run-2-points.gpkg'

ref_lyr = 'accuracy-points'

#points_gdf = gpd.read_file(ref_pts)

# s2_cpy = f'{sentinel_points[:-5]}-copy.gpkg'
# print(s2_cpy)
# s2_cpy_lyr = f'{s2_cpy}-copy'

# # copy points to avoid overwriting
# rsgislib.vectorutils.vector_translate(sentinel_points, 'sentinel-run-2-points', s2_cpy, 
#                                       'sentinel-run-2-points-copy', out_format='GPKG', del_exist_vec=True)

# # populate reference points
# rsgislib.classification.pop_class_info_accuracy_pts(
#     input_img=HR6_sentinel_chg_img,
#     vec_file=s2_cpy,
#     vec_lyr='sentinel-run-2-points-copy',
#     rat_class_col="ClassName",
#     vec_class_col="S2_HR6_chg",
#     vec_ref_col=None,
#     vec_process_col=None,
# )

# # read pts as gdf
# ref_gdf_s2 = gpd.read_file(s2_cpy)

# vec_refpts_vld_file_s2 = '/Users/ben/Desktop/national-scale-change/validation/HR6/sentinel-run-2-points-valid.gpkg'
# vec_refpts_vld_lyr_s2 = 'sentinel-run-2-points-valid'

# points_gdf = ref_gdf_s2.drop(ref_gdf_s2[ref_gdf_s2["S2_HR6_chg"] == "NA"].index)
# points_gdf.to_file(vec_refpts_vld_file_s2, driver="GPKG")

# out_csv_file_s2 = '/Users/ben/Desktop/national-scale-change/validation/HR6/sentinel-accuracy-metrics.csv'

# calc_acc_metrics_vecsamples_img(
#     vec_file=vec_refpts_vld_file_s2,
#     vec_lyr=vec_refpts_vld_lyr_s2,
#     ref_col="ref-pts",
#     cls_col="S2_HR6_chg",
#     cls_img=HR6_ls_chg_img,
#     img_hist_col = 'Histogram',
#     out_csv_file=out_csv_file_s2,
# )

ls_cpy = f'{ls_points[:-5]}-copy.gpkg'
ls_cpy_lyr = f'{ls_cpy}-copy'

# copy points to avoid overwriting
# rsgislib.vectorutils.vector_translate(ls_points, 'run-2-points', ls_cpy, 
#                                       'run-2-points-copy', out_format='GPKG', del_exist_vec=True)

# populate reference points
rsgislib.classification.pop_class_info_accuracy_pts(
    input_img=HR6_ls_chg_img,
    vec_file=ls_cpy,
    vec_lyr='run-2-points-copy',
    rat_class_col="ClassName",
    vec_class_col="LS_HR5_run1_chg_aoi",
    vec_ref_col=None,
    vec_process_col=None,
)

# read pts as gdf
ref_gdf = gpd.read_file(ls_cpy)

vec_refpts_vld_file_ls = '/Users/ben/Desktop/national-scale-change/validation/HR6/landsat-run-2-points-valid.gpkg'
vec_refpts_vld_lyr_ls = 'landsat-run-2-points-valid'

points_gdf = ref_gdf.drop(ref_gdf[ref_gdf["LS_HR5_run1_chg"] == "NA"].index)
points_gdf.to_file(vec_refpts_vld_file_ls, driver="GPKG")

out_csv_file = '/Users/ben/Desktop/national-scale-change/validation/HR5/run1-cell-acc-metrics-aoi-masked.csv'
out_json_file = '/Users/ben/Desktop/national-scale-change/validation/HR5/run1-cell-acc-metrics-aoi-masked.json'


calc_acc_metrics_vecsamples_img(
    vec_file=vec_refpts_vld_file_ls,
    vec_lyr=vec_refpts_vld_lyr_ls,
    ref_col="ref-pts",
    cls_col="LS_HR5_run1_chg",
    cls_img=HR6_ls_chg_img,
    img_hist_col = 'Histogram',
    out_csv_file=out_csv_file,
    out_json_file=out_json_file,
)

