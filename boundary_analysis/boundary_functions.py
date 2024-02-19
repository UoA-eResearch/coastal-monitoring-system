# import modules
import rsgislib
import rsgislib.rastergis
import rsgislib.imageutils
from rsgislib import imagecalc
import rsgislib.classification
import numpy as np
import pandas as pd
import glob
import os
import re
from sklearn.metrics import mean_squared_error
from osgeo_utils.gdal_sieve import gdal_sieve
from osgeo import gdal
from rios import rat


#### FUNCTIONS ###

def sieve(input_img, output_img, num_pxls, connectedness=4):
    """
    function to perform gdal_sieve to remove small clusters of pixels

    input_img - image sieve will be performed on
    output_img - output image that has been sieved
    num_pxls - maximum size of cluster of pixels that will be removed
    connectedness - either 4 indicating that diagonal pixels are not considered directly 
    adjacent for polygon membership purposes or 8 indicating they are.
    """
    # add 1 to num_pxls to ensure clusters up to the size of num_pxls are removed.
    thres = num_pxls+1
    # perform sieve operation
    gdal_sieve(src_filename=input_img, dst_filename=output_img, threshold=thres, connectedness=connectedness)


def return_boundary(class_image, tmp_folder, mask_vals):
    # mask class of boundary under investigation e.g. water class for sand_ndwi
    out_mask = os.path.join(tmp_folder, 'out-mask-{}.kea'.format(mask_vals))
    rsgislib.imageutils.mask_img(class_image, class_image, out_mask,'KEA', rsgislib.TYPE_16INT, 0, mask_vals)
    rsgislib.imageutils.pop_img_stats(out_mask, True, 0, True)

    boundary_img = os.path.join(tmp_folder, 'boundary-pxls-{}.kea'.format(str(mask_vals)))
    # return opposing boundary 
    rsgislib.rastergis.find_boundary_pixels(out_mask, boundary_img)

    return boundary_img

def count_chg_pxls_in_boundary(chg_img, boundary_pxls_img, chg_vals):
    """
    chg_img - change output, either ndvi or water indice
    boundary_img - corresponding boundary image - veg for ndvi change and water for ndwi change
    chg_vals - dict of classes and corresponding values in change image e.g. 1 = sand 2 = water for ndwi change. 
    """
    # extract boundary pixels
    array = rsgislib.imageutils.extract_img_pxl_vals_in_msk(chg_img, [1], boundary_pxls_img, 1)

    # get counts for each class and return as a nested array
    cls, counts = np.unique(array, return_counts=True)
    con = np.asarray((cls, counts)).T

    print(con)
    
    count_dict = {}
    
    for key, val in chg_vals.items():
        count_dict[key] = 0
        for i in con:
            if i[0] == val:
                count_dict[key] = i[1]
    
    # return total boundary pxls for %
        try:
            if con.shape == (3,2):
                total = con[1] + con[2]
                total = total[1]
            else:
                total = con[0] + con[1]
                total = total[1]
            count_dict['total'] = total
        except:
            count_dict['total'] = 0
    
    return count_dict

def calc_cdi(input_chg_img, class_img, tmp, mask_vals, eov_boundary):
    """
    function to return change detection index from merged change detection analysis

    input_chg_img - chg image that cdi will be generated for
    class_img - classification image that boundary will be extracted from 
    tmp - temporary folder for intermediary outputs
    mask_vals -  list, classification values to be masked
    eiv_boundary - boolean, if true analysis will be EOV if false will be IW boundary
    """
    # return boundary image
    boundary_img = return_boundary(class_img, tmp, mask_vals)

    
    # define dict containing change values keys defined by eov_boundary
    if eov_boundary == True:
            chg_values = {'vegetation': 2, 'sand': 1}
    else:
        chg_values = {'water': 2, 'sand': 1}
    # return count of change pixels in boundary
    boundary_pxls = count_chg_pxls_in_boundary(input_chg_img, boundary_img, chg_values)

    # if boundary_pxls is not 0 calc cdi else cdi and est chg = np.nan
    # return cdi based based on eov_boundary
    if boundary_pxls['total'] != 0:
        if eov_boundary == True:
            cdi = round((boundary_pxls['vegetation'] - boundary_pxls['sand']) / (boundary_pxls['vegetation'] + boundary_pxls['sand']), 3)
        else: 
            cdi = round((boundary_pxls['sand'] - boundary_pxls['water']) / (boundary_pxls['sand'] + boundary_pxls['water']), 3)
        
        # calc estimated change
        # get img res
        xRes, yRes = rsgislib.imageutils.get_img_res(input_chg_img)
        estimated_change = cdi * xRes
    else:
        cdi = np.nan
        estimated_change = np.nan

    # remove tmp folder 
    #os.rmdir(tmp)

    # define cdi key based on eov_boundary
    if eov_boundary == True:
        cdi_k = 'cdi_eov'
        chg_k = 'est_eov_chg'
    else:
        cdi_k = 'cdi_iw'
        chg_k = 'est_iw_chg'

    # return cdi_val and estimated chg
    return {cdi_k: cdi, 
            chg_k: estimated_change}

def return_chg_pixels(folder, initial_class_img, output_cls_img):
    """
    function that returns the pixels that have changed between classes
    folder - folder containing ndwi and ndvi change images
    initial_class_img - initial classification image used in change detection analysis

    returns output_cls_img - image containing 4 classes:
                                            1 - water-sand change
                                            2 - sand-water change
                                            3 - vegetation-sand change
                                            4 - sand-vegetation change
    """
    print(folder)
    try:
        # return ndwi and ndvi imgs
        for img in glob.glob(folder + '/*.kea'):
            print(img)
            if img.split('/')[-1][:4] == 'ndwi':
                ndwi = img
            else: 
                ndvi = img
        # band math 
        band_defns = []
        band_defns.append(imagecalc.BandDefn('ndwi', ndwi, 1))
        band_defns.append(imagecalc.BandDefn('ndvi', ndvi, 1))
        band_defns.append(imagecalc.BandDefn('class', initial_class_img, 1))
        # define expression where water-sand = 1 sand-water = 2 vegetation-sand = 3 and sand-vegetation = 4
        exp = (
            "(class==1)&&(ndwi==2)?1:(class==2)&&(ndwi==1)?2:(class==3)&&(ndvi==1)?3:(class==1)&&(ndvi==2)?4:"
            "(class==1)&&(ndvi==1)||(class==2)&&(ndwi==2)||(class==3)&&(ndvi==2)?5:0"
            )
        # gen output using band math
        imagecalc.band_math(
            output_cls_img, exp, 'KEA', rsgislib.TYPE_8UINT, band_defns
        )
        #populate classification with class names for validation
        rsgislib.rastergis.pop_rat_img_stats(output_cls_img, add_clr_tab=True, calc_pyramids=True, ignore_zero=True)

    
        ratDataset = gdal.Open(output_cls_img, gdal.GA_Update)
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
    except:
        pass

def calc_area_change(change_pxls_img):
    """
    function to return estimated area change from image containing change pixels between classes
    change_pxls_img - image containing change pixels between classes

    returns - dictionary containing difference in pxl counts between water-sand and vegetation-sand
            negative values indicate erosion and positive values indicate accretion 
    """
    # count pixels for each class in image
    counts = imagecalc.count_pxls_of_val(change_pxls_img, [1,2,3,4], 1)

    # define change pixels
    sand_to_water = counts[0] 
    water_to_sand = counts[1] 
    veg_to_sand = counts[2]
    sand_to_veg = counts[3]

    # get img res
    xRes, yRes = rsgislib.imageutils.get_img_res(change_pxls_img)

    # define change for each set of classes as hectares
    waterline_area_chg = (sand_to_water - water_to_sand)*(xRes*yRes)/10000

    eov_area_chg = (veg_to_sand - sand_to_veg )*(xRes*yRes)/10000

    return {'area_iw': waterline_area_chg,
            'area_eov': eov_area_chg}

def return_new_class_image(folder, output_cls_img):
    """
    function that returns new classification image based on outputs from change detection analysis

    returns output_cls_img - image containing 3 classes:
                                            1 - sand/sediment
                                            2 - water
                                            3 - vegetation
                                            
    """
    print(folder)
    # return ndwi and ndvi imgs
    for img in glob.glob(folder + '/*.kea'):
        print(img)
        if img.split('/')[-1][:4] == 'ndwi':
            ndwi = img
        elif img.split('/')[-1][:4] == 'ndvi':  
            ndvi = img
    # band math 
    band_defns = []
    band_defns.append(imagecalc.BandDefn('ndwi', ndwi, 1))
    band_defns.append(imagecalc.BandDefn('ndvi', ndvi, 1))
    # define expression where sand = 1 water = 2 vegetation-sand = 3 and sand-vegetation = 4
    exp = (
        "(ndwi==2)?2:(ndvi==2)?3:(ndvi==1)||(ndwi==1)?1:0"
        )
    # gen output using band math
    imagecalc.band_math(
        output_cls_img, exp, 'KEA', rsgislib.TYPE_8UINT, band_defns
    )
    #populate classification with class names for validation
    rsgislib.rastergis.pop_rat_img_stats(output_cls_img, add_clr_tab=True, calc_pyramids=True, ignore_zero=True)

    try:
        ratDataset = gdal.Open(output_cls_img, gdal.GA_Update)
        red = rat.readColumn(ratDataset, 'Red')
        green = rat.readColumn(ratDataset, 'Green')
        blue = rat.readColumn(ratDataset, 'Blue')
        ClassName = np.empty_like(red, dtype=np.dtype('a255'))
        ClassName[...] = ""


        red[1] = 252
        blue[1] = 219
        green[1] = 3
        ClassName[1] = 'sand'

        red[2] = 3
        blue[2] = 102
        green[2] = 252
        ClassName[2] = 'water'

        red[3] = 23
        blue[3] = 110
        green[3] = 5
        ClassName[3] = 'vegetation'

        rat.writeColumn(ratDataset, 'Red', red)
        rat.writeColumn(ratDataset, 'Green', green)
        rat.writeColumn(ratDataset, 'Blue', blue)
        rat.writeColumn(ratDataset, 'ClassName', ClassName)
        ratDataset = None
    except:
        pass

def return_new_class_area(new_cls_img):
    """
    function to return estimated area from new class image generated from change outputs for a given year. 

    returns - dictionary containing area of sand water and vegetation in hectares
    """
    # count pixels for each class in image
    counts = imagecalc.count_pxls_of_val(new_cls_img, [1,2,3], 1)

    # get img res
    xRes, yRes = rsgislib.imageutils.get_img_res(new_cls_img)

    # define area of class in hectares
    sand = (counts[0]*(abs(xRes)*abs(yRes)))/10000
    water = (counts[1]*(abs(xRes)*abs(yRes)))/10000
    veg = (counts[2]*(abs(xRes)*abs(yRes)))/10000


    return {'sand_area': sand,
            'water_area': water,
            'vegetation_area':veg}