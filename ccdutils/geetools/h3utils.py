import pandas as pd
import geopandas as gpd


def get_children(index, gdf):
    """
    function that returns geopandas dataframe containing children h3 cells
    
    args
    index - index or list of indexes that children will be returned for
    gdf - h3 geodataframe

    returns geodataframe with children cells
    """
    if index != list:
        df = gdf.query('index == @index or parent_id == @index')
    else:
        df_list = []
        for i in index:
            i_df = gdf.query('index == @i or parent_id == @i')
            df_list.append(i_df)
        df = pd.concat(df_list)
    return df

def get_index_by_res(res, gdf):
    """
    function to return the indexes of a given h3 resolution

    args
    res - resolution that indexes will be returned for
    gdf - h3 geodataframe

    returns list of indexes as specified resolution 
    """
    df = gdf.query('resolution == @res')
    return df['index'].tolist()

def get_resolution(index, gdf):
    """
    function to get resolution of specified h3 cell
    
    args
    index - h3 cell index
    gdf - h3 geodataframe
    
    returns - resolution of h3 cell
    """
    df = gdf.query('index == @index')
    return int(df['resolution'])


def get_child_cells(gdf, index, resolution=None): # if resolution is none all children returned at all resolutions. 
    """
    function to get children cells for a given h3 index

    args 
    gdf - h3 geodataframe
    index - index that children will be found for
    resolution - default = None, if resolution is specified only children at that resolution will be returned

    returns geodataframe containing children cells
    """
    # get index resolution
    index_res = get_resolution(index, gdf)
    # get children of cell specified by index param
    df = get_children(index, gdf)
    # get children of finer resolutions
    if index_res == 5:
        res = [6,7]
    if index_res == 6:
        res = [7]
    else:
        res = [5,6,7]
        
    for r in res:
        i = get_index_by_res(r, df)
        #print(i)
        test_r = get_children(i, gdf)
        #print(test_r.head())
        df = pd.concat([df, test_r])
    
    # drop duplicates and orginal cell
    df.drop_duplicates('index', inplace=True)
    df.drop(df.loc[df['index']==index].index, inplace=True)
    
    if resolution != None:
        df = df[df.resolution == resolution]
    
    return df


