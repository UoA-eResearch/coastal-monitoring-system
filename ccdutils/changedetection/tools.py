# import modules
import os
import shutil
import sys
import glob
import rsgislib
import rsgislib.changedetect.pxloutlierchng
import rsgislib.imageutils
import rsgislib.rastergis
import rsgislib.imagecalc
from rsgislib import rastergis
from skimage.filters import threshold_otsu  
import pandas as pd
import numpy as np
#import tqdm
import time
from osgeo_utils.gdal_sieve import gdal_sieve


### DEFINE FUNCTIONS ###

def return_stats(input_img, img_band, cls_img, cls_val, below_thres=True):
    """
    function to return stats for values extracted from image for given class including the sum of outlier pixels

    input_img - image that values will be extracted from
    img_band - band that values will be extracted from
    cls_img - classification containing class that values will be extracted from 
    cls_val - value of class in cls_img
    below_thres - Boolean stating whether outliers are above or below otsu threshold - True = below treshold 

    returns pd.DataFrame of stats
    """
    array = rsgislib.imageutils.extract_img_pxl_vals_in_msk(input_img, [img_band], cls_img, cls_val)
    if array.shape[0] > 0:
        # remove nodata values
        array = array[array != -99]
        iqr = np.subtract(*np.percentile(array,[75, 25]))
        otsu = threshold_otsu(array)
        if below_thres == True:
            est_outliers = (array < otsu).sum()
        if below_thres == False:
            est_outliers = (array > otsu).sum()
        df = pd.DataFrame(array)
        skew = df.skew()
        kurt = df.kurt()
        stats = df.describe()
        stats.loc['iqr'] = iqr
        stats.loc['skewness'] = skew
        stats.loc['kurtosis'] = kurt
        stats.loc['threshold'] = otsu
        stats.loc['outliers'] = est_outliers
        stats.loc['outliers-percent'] = est_outliers/array.size*100
        return stats    
    
    else:
        pass


def return_array(input_img, img_band, cls_img, cls_val, no_data_val=-99):
    """
    function that returns array of the input image values that are contained within the specified landcover class

    input_img - image that values will be extracted from
    img_band - band that values will be extracted from
    cls_img - classification containing class that values will be extracted from 
    cls_val - value of class in cls_img

    return - np.array of image values for given class
    """
    array = rsgislib.imageutils.extract_img_pxl_vals_in_msk(input_img, [img_band], cls_img, cls_val, no_data_val=no_data_val)
    return array

        
def find_class_otsu_outliers(
    input_img: str,
    in_msk_img: str,
    output_img: str,
    low_thres: bool,
    img_mask_val: int = 1,
    img_band: int = 1,
    img_val_no_data: float = None,
    gdalformat: str = "KEA",
    plot_thres_file: str = None,
) -> float:
    """
    This function to find outliers within a class using an otsu thresholding. It is
    assumed that the input_img is from a different date than the mask
    (classification) and therefore the outliers will related to class changes.

    :param input_img: the input image for the analysis. Just a single band will be used.
    :param in_msk_img: input image mask use to define the region (class) of interest.
    :param output_img:  output image with pixel over of 1 for within mask but
                         not outlier and 2 for in mask and outlier.
    :param low_thres: a boolean as to whether the threshold is on the upper or lower
                      side of the histogram. If True (default) then outliers will be
                      identified as values below the threshold. If False then outliers
                      will be above the threshold.
    :param img_mask_val: the pixel value within the in_msk_img specifying the class
                         of interest.
    :param img_band: the input_img image band to be used for the analysis.
    :param img_val_no_data: the input_img image not data value. If None then the value
                            will be read from the image header.
    :param gdalformat: the output image file format. (Default: KEA)
    :param plot_thres_file: A file path for a plot of the histogram with the
                            threshold. If None then ignored.
    :return: The threshold identified.

    """
    import rsgislib.tools.stats

    if img_val_no_data is None:
        img_val_no_data = rsgislib.imageutils.get_img_no_data_value(input_img)

    msk_arr_vals = rsgislib.imageutils.extract_img_pxl_vals_in_msk(
        input_img, [img_band], in_msk_img, img_mask_val, img_val_no_data
    )
    print("There were {} pixels within the mask.".format(msk_arr_vals.shape[0]))
    
    if msk_arr_vals.shape[0] > 0:

        chng_thres = threshold_otsu(msk_arr_vals)

        band_defns = list()
        band_defns.append(rsgislib.imagecalc.BandDefn("msk", in_msk_img, 1))
        band_defns.append(rsgislib.imagecalc.BandDefn("val", input_img, img_band))
        if low_thres:
            exp = (
                f"(val=={img_val_no_data})?0:(msk=={img_mask_val})&&"
                f"(val<{chng_thres})?2:(msk=={img_mask_val})?1:0"
            )
        else:
            exp = (
                f"(val=={img_val_no_data})?0:(msk=={img_mask_val})&&"
                f"(val>{chng_thres})?2:(msk=={img_mask_val})?1:0"
            )
        rsgislib.imagecalc.band_math(
            output_img, exp, gdalformat, rsgislib.TYPE_8UINT, band_defns
        )

        if gdalformat == "KEA":
            rsgislib.rastergis.pop_rat_img_stats(
                clumps_img=output_img,
                add_clr_tab=True,
                calc_pyramids=True,
                ignore_zero=True,
            )
            class_info_dict = dict()
            class_info_dict[1] = {"classname": "no_chng", "red": 0, "green": 255, "blue": 0}
            class_info_dict[2] = {"classname": "chng", "red": 255, "green": 0, "blue": 0}
            rsgislib.rastergis.set_class_names_colours(
                output_img, "chng_cls", class_info_dict
            )
        else:
            rsgislib.imageutils.pop_thmt_img_stats(
                output_img, add_clr_tab=True, calc_pyramids=True, ignore_zero=True
            )

        if plot_thres_file is not None:
            import rsgislib.tools.plotting

            rsgislib.tools.plotting.plot_histogram_threshold(
                msk_arr_vals[..., 0], plot_thres_file, chng_thres
            )
        
        return chng_thres
    
    else:
        pass

    
def return_chg_img_with_boundary(chg_image, cls_image, output_img):
    """
    function that returns change output image with opposing boundary pixels that were removed for analysis.
    """
    # define band info for input images
    bandDefns = []
    bandDefns.append(rsgislib.imagecalc.BandDefn("chg_img", chg_image, 1))
    bandDefns.append(rsgislib.imagecalc.BandDefn("cls_img", cls_image, 1))
    # define expression to return non-change pixels and change pixels
    exp = f"(chg_img==2)?2:(cls_img==1)?1:0"
    # band math function to create new image
    rsgislib.imagecalc.band_math(output_img, exp, 'KEA', rsgislib.TYPE_8UINT, bandDefns)
    rsgislib.imageutils.pop_img_stats(output_img,True,0.0,True)          


def sieve(input_img, output_img, num_pxls, connectedness=4):
    """
    perform gdal_sieve to remove noise pixels
    """
    # perform sieve operation
    gdal_sieve(src_filename=input_img, dst_filename=output_img, threshold=num_pxls, connectedness=connectedness)

    
def identify_chg_img(folder):
    vals = [1,2]
    
    ndvi_count = rsgislib.imagecalc.count_pxls_of_val(ndvi_img, vals)
    ndwi_count = rsgislib.imagecalc.count_pxls_of_val(ndwi_img, vals)
    
    if ndvi_count[1] > ndwi_count[1]:
        chg = True # ndvi img contains chg pixels
    else:
        chg = False # ndwi img contains chg pixels
    return chg

def return_chg_outputs(folder, out_folder):
    """
    function to return valid change outputs in folder
    folder - folder containing change outputs
    out_folder - final destination of valid outputs
    """

    # iterate over folder to find imgs with change
    for img in glob.glob(folder + '/*i.kea'):
        fn = img.split('/')[-1]
        # get vals from img
        img_vals = rsgislib.imagecalc.get_unique_values(img, img_band=1)
        # move imgs with  chg to out_folder
        if 2 in img_vals:
            print('{} contains  change pixels: moving'.format(fn))
            os.rename(img, os.path.join(out_folder, fn))
            
        
def mask_opposing_boundary_pixels(class_image, masked_class_image, tmp_folder, mask_vals):
    """
    function to mask boundary of classes that aren't required for analysis

    class_image - classification image
    masked_class_image - classification containing classes required for analysis with opposing boundary pixels masked
    tmp_folder - tmp folder for intermediary outputs
    mask_vals - list of class vals to be masked

    """
    # mask class of boundary under investigation e.g. water class for sand_ndwi
    out_mask = os.path.join(tmp_folder, 'out-mask.kea')
    rsgislib.imageutils.mask_img(class_image, class_image, out_mask,'KEA', rsgislib.TYPE_16INT, 0, mask_vals)
    rsgislib.imageutils.pop_img_stats(out_mask, True, 0, True)

    # return opposing boundary pixels
    boundary_img = os.path.join(tmp_folder, 'boundary-pxls.kea')
    rastergis.find_boundary_pixels(out_mask, boundary_img)

    # mask classification to remove opposing boundary pixels
    rsgislib.imageutils.mask_img(class_image, boundary_img, masked_class_image, 'KEA', rsgislib.TYPE_16INT, 0, [1])
    rsgislib.imageutils.pop_img_stats(masked_class_image, True, 0, True)

    # remove intermediary outputs
    os.remove(out_mask)
    os.remove(boundary_img)


def mask_opposing_class(class_image, masked_class_image, mask_vals):
    """
    function to mask classes that aren't required for analysis

    class_image - classification image
    masked_class_img - classification containing classes required for analysis
    mask_vals - list of class vals to be masked
    """
    # mask class of boundary under investigation e.g. water class for sand_ndwi
    rsgislib.imageutils.mask_img(class_image, class_image, masked_class_image,'KEA', rsgislib.TYPE_16INT, 0, mask_vals)
    rsgislib.imageutils.gen_valid_mask(masked_class_image, masked_class_image, 'KEA', 0.0)
    rsgislib.imageutils.pop_img_stats(masked_class_image, True, 0, True)