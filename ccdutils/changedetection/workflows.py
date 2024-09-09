import os
import shutil
from ccdutils.changedetection import tools


### FUNCTION TO RETURN CHANGE USING OTSU OUTLIERS METHOD ON CLASS-BY-CLASS BASIS ####
def return_change_output_otsu(input_img, class_img, out_folder, cell_id, ndvi_band, ndwi_band, class_vals):
    """
    input_img - image that change will be derived from
    class_img - classification image that is from different time instance to change image
    out_folder - folder where outputs will be saved
    cell_id - H3 cell index code 
    ndvi_band - band value for ndvi_band
    ndwi_band - band value for water indices band
    class_vals - dict of class value for water sediment and vegetation
    mask_vals - list of values in classification to be ignored
    """

    # generate a tmp folder to all outputs for each chg_type
    tmp = os.path.join(out_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # generate sand-water-class-img by removing opposing boundary
    sand_water_class_img = os.path.join(tmp, 'water-sand-cls-img.kea')
    tools.mask_opposing_boundary_pixels(class_img, sand_water_class_img, tmp, [class_vals['water'],4,5]) # 

    # generate sand-veg-class-img by remocing opposing boundary
    sand_veg_class_img = os.path.join(tmp, 'veg-sand-cls-img.kea')
    tools.mask_opposing_boundary_pixels(class_img, sand_veg_class_img, tmp, [class_vals['vegetation'],4,5])
    
    # return chg outputs for all classes
    # create dict of inputs
    input_bands = {'sand-ndvi': ndvi_band, 'sand-ndwi': ndwi_band,
                'water-ndwi': ndwi_band, 'veg-ndvi': ndvi_band}
    
    # iterate over list of bands to gen outputs
    for key,val in input_bands.items():
        print('identifying change for {} in {} and {}'.format(key, input_img, cell_id))
        print(key)
        # return initial threshold (otsu) and min/max img vals within class
        # set cls based on key and remove opposing boundary pixels for sand_ndvi and sand_ndwi
        if key == 'sand-ndwi':
            in_class_img =  sand_water_class_img
            cls = class_vals['sand']
            chg_img = os.path.join(tmp, '{}-chg.kea'.format(key))
        elif key == 'sand-ndvi':
            in_class_img =  sand_veg_class_img
            cls = class_vals['sand']
            chg_img = os.path.join(tmp, '{}-chg.kea'.format(key))
        elif key == 'veg-ndvi':
            in_class_img = class_img
            cls = class_vals['vegetation']
            chg_img = os.path.join(tmp, '{}.kea'.format(key))
        else:
            in_class_img = class_img
            cls = class_vals['water']
            chg_img = os.path.join(tmp, '{}.kea'.format(key))

        print('class: ', cls)
        print('img: ', val)
        
        # set threshold above or below depending on class 
        if key[:4] == 'sand': ## True = outliers below threshold, False = outliers above threshold. water/veg classes = True
            low_thres = False
        else:
            low_thres = True
        
        stats = tools.return_stats(input_img, val, in_class_img, cls, below_thres=low_thres)
        # continue if class is not present in hex cell
        if stats is None:
            print('class not present in cell: {}'.format(cell_id))
            continue

        # identify change pixels
        try:
            tools.find_class_otsu_outliers(input_img, in_class_img, chg_img, low_thres, cls,
                                    val, -999, 'KEA', None) 
        except:
            pass
        
        # return chg_img with boundary for sand change images
        if key[:4] == 'sand':
            out_img = os.path.join(tmp, '{}.kea'.format(key))
            tools.return_chg_img_with_boundary(chg_img, class_img, out_img)
        else:
            pass
   
    # check tmp folder to see which outputs indicate chg and move to out_folder
    tools.return_chg_outputs(tmp, out_folder)    
    # remove tmp folder
    #shutil.rmtree(tmp)


### FUNCTION TO RETURN CHANGE OUTPUTS BASED ON MERGED CLASSESE ###
def return_change_output_otsu_merged_classes(input_img, class_img, out_folder, cell_id, ndvi_band, ndwi_band, class_vals):
    """
    input_img - image that change will be derived from
    class_img - classification image that is from different time instance to change image
    out_folder - folder where outputs will be saved
    cell_id - H3 cell index code 
    ndvi_band - band value for ndvi_band
    ndwi_band - band value for water indices band
    mask_vals - list of values in classification to be ignored
    """

    # generate a tmp folder to all outputs for each chg_type
    tmp = os.path.join(out_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # generate sand-water-class-img by removing opposing class
    # define masked class img
    sand_water_class_img = os.path.join(tmp, 'water-sand-cls-img.kea')
    # define mask val based on cls to be masked
    tools.mask_opposing_class(class_img, sand_water_class_img, [class_vals['vegetation'], 4, 5]) # 

    # generate sand-veg-class-img by remocing opposing class
    # define masked class img
    sand_veg_class_img = os.path.join(tmp, 'veg-sand-cls-img.kea')
    # define mask val based on cls to be masked
    #msk_val = class_vals['water']
    tools.mask_opposing_class(class_img, sand_veg_class_img, [class_vals['water'], 4,5])
    
    # return chg outputs for all classes
    # create dict of inputs
    input_bands = {'ndvi': ndvi_band, 'ndwi': ndwi_band}

    # iterate over list of bands to gen outputs
    for key,val in input_bands.items():
        print('identifying change for {} in {} and {}'.format(key, input_img, cell_id))
        print(key)
        # return initial threshold (otsu) and min/max img vals within class
        # set cls based on key and remove opposing boundary pixels for sand_ndvi and sand_ndwi
        if key == 'ndwi':
            in_class_img =  sand_water_class_img
            cls = 1
            chg_img = os.path.join(out_folder, '{}-chg.kea'.format(key))
        elif key == 'ndvi':
            in_class_img =  sand_veg_class_img
            cls = 1
            chg_img = os.path.join(out_folder, '{}-chg.kea'.format(key))
        
        print('class: ', cls)
        print('img: ', val)

        try:
            stats = tools.return_stats(input_img, val, in_class_img, cls, below_thres=low_thres)
        # continue if class is not present in hex cell
            if stats is None:
                print('class not present in cell: {}'.format(cell_id))
                continue
        except:
            pass
        
        # return change output        
        try:
            # below_thres = False = sand will have a value of 1 and water and vegetation will be 2
            low_thres = False
            tools.find_class_otsu_outliers(input_img, in_class_img, chg_img, low_thres, cls,
                                    val, -99, 'KEA', None) 
            tools.sieve(chg_img, chg_img, 75)
        except:
            pass

    # remove processing folder
    shutil.rmtree(tmp)