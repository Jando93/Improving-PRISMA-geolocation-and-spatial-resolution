import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
import os
from arosics import COREG_LOCAL


# In[ ]:


path = 'C:\\Users\\IBE_TEST\\Documents\\CNR-IBE\\Dati\\PRISMA_pansharpening_project\\ARBOREA\\'


# In[ ]:


im_target = path+'PRS_L2D_STD_20220530102847_20220530102851_0001_VNIR_SWIR.tif'
im_reference    = path+'S2A_MSIL2A_20220531T101611_N0400_R065_T32SM_20220531T163910_mosaicked_B4_Rescaled0-1_clipped.tif'


# In[ ]:


'''
VNIR
B2-492.3 = B12-489.8
B3-558.9 = B21-559.0
B4-66.4.9 = B33-664.9
B5-703.8 = B37-703.7
B6-739.1 = B40-734.0
B7-779.7 = B44-775.3
B8-832.9 = B49-827.9
B8A-864.0 = B52 - 860.0

SWIR
B11-1610.4 = B64-1606.5
B12-2185.7 = B128-183.4

'''


# In[ ]:


sentinel_bands = [1, #B2
                  2, #B3
                  3, #B4
                  4, #B5
                  5, #B6
                  6, #B7
                  7, #B8
                  8, #B11
                  9  #B12    
                 ]

sentinel_bands_name = [2,
                  3,
                  4,
                  5,
                  6,
                  7,
                  8,
                  11,
                  12   
                 ]

prisma_bands = [12,
                21,
                33,
                37,
                40,
                44,
                49,
                132,
                196
               ]


# In[ ]:


CRL = COREG_LOCAL(im_reference,
                  im_target, 
                  grid_res = 30,
                  window_size =(50,50),
                  max_shift = 30,
                  align_grids =False,
                  match_gsd=False,
                  resamp_alg_deshift = 'nearest',
                  resamp_alg_calc = 'nearest',
                  path_out = path+'AROSICS_output\\Registered_20220530102847_AROSICS_Local_B3_B33_____.tif',
                  fmt_out = 'GTIFF',
                  max_iter = 25,
                  r_b4match = 1,
                  s_b4match = 33,
                  ignore_errors = True)

CRL.correct_shifts()
points = CRL.CoRegPoints_table
CRL.tiepoint_grid.to_PointShapefile(path_out=path+'AROSICS_output\\AROSICS_TiePoints_Local_B3_PAN_20220530102847.shp')


# In[ ]:


CRL.correct_shifts()


# In[ ]:


points = CRL.CoRegPoints_table


# In[ ]:


CRL.tiepoint_grid.to_PointShapefile(path_out=path+'AROSICS_output\\AROSICS_TiePoints_Local_B3_B33_20220530102847_____.shp')


# In[ ]:


n = 0
while n < len(sentinel_bands):
    CRL = COREG_LOCAL(im_reference,
                      im_target, 
                      grid_res = 15,
                      window_size =(50,50),
                      max_shift = 30,
                      align_grids =False,
                      match_gsd=False,
                      resamp_alg_deshift = 'nearest',
                      resamp_alg_calc = 'nearest',
                      path_out = path+'AROSICS_output\\Registered_20220530102847_AROSICS_Local_B{0}_B{1}.tif'.format(sentinel_bands_name[n], prisma_bands[n]),
                      fmt_out = 'GTIFF',
                      max_iter = 50,
                      r_b4match = sentinel_bands[n],
                      s_b4match = prisma_bands[n],
                      ignore_errors = True)
    
    CRL.correct_shifts()
    points = CRL.CoRegPoints_table
    CRL.tiepoint_grid.to_PointShapefile(path_out=path+'AROSICS_output\\AROSICS_TiePoints_Local_B{0}_B{1}_20220530102847.shp'.format(sentinel_bands_name[n], prisma_bands[n]))

    n+=1


# In[ ]:





# In[ ]:


====================================================================================================================
'''OPTIMIZAZTION TIE POINTS'''
====================================================================================================================


# In[ ]:


path = 'C:\\Users\\IBE_TEST\\Documents\\CNR-IBE\\Dati\\PRISMA_pansharpening_project\\ARBOREA\\AROSICS_output\\'


# In[ ]:


lista_files = []
dirFileList = os.listdir(path) #legge la cartella 1
os.chdir(path) #setta la cartella
for file in dirFileList: #per ciascun file nella cartella 1
    if os.path.splitext(file)[-1] == '.shp': #le cui ultime lettere sono ".img"
        lista_files.append(os.path.join(path, file)) #aggiungile alla lista vuot

lista_files


# In[ ]:


order_bands = []
i = 0
while i < len(lista_files):
    if i == 0:
        f = gpd.read_file(lista_files[i])
        name_s2_band = lista_files[i].split("_")[-3] 
        order_bands.append(name_s2_band)
        f['S2_Band'] = name_s2_band
    if i > 0:
        f2 = gpd.read_file(lista_files[i])
        name_s2_band = lista_files[i].split("_")[-3] 
        order_bands.append(name_s2_band)
        f2['S2_Band'] = name_s2_band
        f = f.merge(f2,on='POINT_ID', how='inner', suffixes=(None, '_'+name_s2_band))
    
    i+=1


# In[ ]:


#print columns
np.unique(f.columns)


# In[ ]:


b = dict()

b['RELIABILITY'] = np.array(f[['RELIABILIT', 'RELIABILIT_B12', 'RELIABILIT_B2',
                               'RELIABILIT_B3', 'RELIABILIT_B4', 'RELIABILIT_B5', 'RELIABILIT_B6',
                               'RELIABILIT_B7', 'RELIABILIT_B8']])
b['POINT_ID'] = np.array(f['POINT_ID'])
b['X_IM'] = np.array(f[['X_IM', 'X_IM_B12', 'X_IM_B2', 'X_IM_B3',
       'X_IM_B4', 'X_IM_B5', 'X_IM_B6', 'X_IM_B7', 'X_IM_B8']])
b['X_MAP'] = np.array(f[['X_MAP',
       'X_MAP_B12', 'X_MAP_B2', 'X_MAP_B3', 'X_MAP_B4', 'X_MAP_B5',
       'X_MAP_B6', 'X_MAP_B7', 'X_MAP_B8']])
b['Y_IM'] = np.array(f[['Y_IM', 'Y_IM_B12', 'Y_IM_B2',
       'Y_IM_B3', 'Y_IM_B4', 'Y_IM_B5', 'Y_IM_B6', 'Y_IM_B7', 'Y_IM_B8']])
b['Y_MAP'] = np.array(f[['Y_MAP', 'Y_MAP_B12', 'Y_MAP_B2', 'Y_MAP_B3', 'Y_MAP_B4',
       'Y_MAP_B5', 'Y_MAP_B6', 'Y_MAP_B7', 'Y_MAP_B8']])
b['S2_Band'] = np.array(f[['S2_Band', 'S2_Band_B12',
       'S2_Band_B2', 'S2_Band_B3', 'S2_Band_B4', 'S2_Band_B5',
       'S2_Band_B6', 'S2_Band_B7', 'S2_Band_B8']])

b['ABS_SHIFT'] = np.array(f[['ABS_SHIFT', 'ABS_SHIFT_B12', 'ABS_SHIFT_B2', 'ABS_SHIFT_B3',
       'ABS_SHIFT_B4', 'ABS_SHIFT_B5', 'ABS_SHIFT_B6', 'ABS_SHIFT_B7',
       'ABS_SHIFT_B8']])
b['ANGLE'] = np.array(f[['ANGLE', 'ANGLE_B12', 'ANGLE_B2', 'ANGLE_B3',
       'ANGLE_B4', 'ANGLE_B5', 'ANGLE_B6', 'ANGLE_B7', 'ANGLE_B8']])
b['L1_OUTLIER'] = np.array(f[['L1_OUTLIER', 'L1_OUTLIER_B12', 'L1_OUTLIER_B2', 'L1_OUTLIER_B3',
       'L1_OUTLIER_B4', 'L1_OUTLIER_B5', 'L1_OUTLIER_B6', 'L1_OUTLIER_B7',
       'L1_OUTLIER_B8' ]])
b['L2_OUTLIER'] = np.array(f[['L2_OUTLIER', 'L2_OUTLIER_B12', 'L2_OUTLIER_B2', 'L2_OUTLIER_B3',
       'L2_OUTLIER_B4', 'L2_OUTLIER_B5', 'L2_OUTLIER_B6', 'L2_OUTLIER_B7',
       'L2_OUTLIER_B8' ]])
b['L3_OUTLIER'] = np.array(f[['L3_OUTLIER', 'L3_OUTLIER_B12', 'L3_OUTLIER_B2', 'L3_OUTLIER_B3',
       'L3_OUTLIER_B4', 'L3_OUTLIER_B5', 'L3_OUTLIER_B6', 'L3_OUTLIER_B7',
       'L3_OUTLIER_B8']])
b['LAST_ERR'] = np.array(f[['LAST_ERR', 'LAST_ERR_B12', 'LAST_ERR_B2', 'LAST_ERR_B3', 'LAST_ERR_B4',
       'LAST_ERR_B5', 'LAST_ERR_B6', 'LAST_ERR_B7', 'LAST_ERR_B8']])
b['OUTLIER'] = np.array(f[['OUTLIER', 'OUTLIER_B12', 'OUTLIER_B2', 'OUTLIER_B3', 'OUTLIER_B4',
       'OUTLIER_B5', 'OUTLIER_B6', 'OUTLIER_B7', 'OUTLIER_B8']])
b['REF_BADDAT'] = np.array(f[['REF_BADDAT', 'REF_BADDAT_B12', 'REF_BADDAT_B2', 'REF_BADDAT_B3',
       'REF_BADDAT_B4', 'REF_BADDAT_B5', 'REF_BADDAT_B6', 'REF_BADDAT_B7',
       'REF_BADDAT_B8' ]])
b['SSIM_AFTER'] = np.array(f[['SSIM_AFTER',
       'SSIM_AFTER_B12', 'SSIM_AFTER_B2', 'SSIM_AFTER_B3',
       'SSIM_AFTER_B4', 'SSIM_AFTER_B5', 'SSIM_AFTER_B6', 'SSIM_AFTER_B7',
       'SSIM_AFTER_B8' ]])
b['SSIM_BEFOR'] = np.array(f[['SSIM_BEFOR', 'SSIM_BEFOR_B12', 'SSIM_BEFOR_B2',
       'SSIM_BEFOR_B3', 'SSIM_BEFOR_B4', 'SSIM_BEFOR_B5', 'SSIM_BEFOR_B6',
       'SSIM_BEFOR_B7', 'SSIM_BEFOR_B8']])
b['SSIM_IMPRO'] = np.array(f[['SSIM_IMPRO', 'SSIM_IMPRO_B12',
       'SSIM_IMPRO_B2', 'SSIM_IMPRO_B3', 'SSIM_IMPRO_B4', 'SSIM_IMPRO_B5',
       'SSIM_IMPRO_B6', 'SSIM_IMPRO_B7', 'SSIM_IMPRO_B8' ]])
b['TGT_BADDAT'] = np.array(f[['TGT_BADDAT',
       'TGT_BADDAT_B12', 'TGT_BADDAT_B2', 'TGT_BADDAT_B3',
       'TGT_BADDAT_B4', 'TGT_BADDAT_B5', 'TGT_BADDAT_B6', 'TGT_BADDAT_B7',
       'TGT_BADDAT_B8' ]])
b['X_SHIFT_M'] = np.array(f[['X_SHIFT_M', 'X_SHIFT_M_B12',
       'X_SHIFT_M_B2', 'X_SHIFT_M_B3', 'X_SHIFT_M_B4', 'X_SHIFT_M_B5',
       'X_SHIFT_M_B6', 'X_SHIFT_M_B7', 'X_SHIFT_M_B8']])
b['X_SHIFT_PX'] = np.array(f[['X_SHIFT_PX',
       'X_SHIFT_PX_B12', 'X_SHIFT_PX_B2', 'X_SHIFT_PX_B3',
       'X_SHIFT_PX_B4', 'X_SHIFT_PX_B5', 'X_SHIFT_PX_B6', 'X_SHIFT_PX_B7',
       'X_SHIFT_PX_B8' ]])
b['X_WIN_SIZE'] = np.array(f[['X_WIN_SIZE', 'X_WIN_SIZE_B12', 'X_WIN_SIZE_B2',
       'X_WIN_SIZE_B3', 'X_WIN_SIZE_B4', 'X_WIN_SIZE_B5', 'X_WIN_SIZE_B6',
       'X_WIN_SIZE_B7', 'X_WIN_SIZE_B8' ]])
b['Y_SHIFT_M'] = np.array(f[['Y_SHIFT_M',
       'Y_SHIFT_M_B12', 'Y_SHIFT_M_B2', 'Y_SHIFT_M_B3', 'Y_SHIFT_M_B4',
       'Y_SHIFT_M_B5', 'Y_SHIFT_M_B6', 'Y_SHIFT_M_B7', 'Y_SHIFT_M_B8']])
b['Y_SHIFT_PX'] = np.array(f[['Y_SHIFT_PX', 'Y_SHIFT_PX_B12', 'Y_SHIFT_PX_B2', 'Y_SHIFT_PX_B3',
       'Y_SHIFT_PX_B4', 'Y_SHIFT_PX_B5', 'Y_SHIFT_PX_B6', 'Y_SHIFT_PX_B7',
       'Y_SHIFT_PX_B8' ]])
b['Y_WIN_SIZE'] = np.array(f[['Y_WIN_SIZE', 'Y_WIN_SIZE_B12', 'Y_WIN_SIZE_B2',
       'Y_WIN_SIZE_B3', 'Y_WIN_SIZE_B4', 'Y_WIN_SIZE_B5', 'Y_WIN_SIZE_B6',
       'Y_WIN_SIZE_B7', 'Y_WIN_SIZE_B8' ]])

b['geometry'] = np.array(f[['geometry', 'geometry_B12',
       'geometry_B2', 'geometry_B3', 'geometry_B4', 'geometry_B5',
       'geometry_B6', 'geometry_B7', 'geometry_B8']])


# In[ ]:


#applichiamo 

x = np.argsort(b['RELIABILITY']) #feature prescelta per argsort()


RELIABILITY_sorted = np.take_along_axis(b['RELIABILITY'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
X_IM = np.take_along_axis(b['X_IM'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
X_MAP = np.take_along_axis(b['X_MAP'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
Y_IM = np.take_along_axis(b['Y_IM'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
Y_MAP  = np.take_along_axis(b['Y_MAP'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
S2_Band  = np.take_along_axis(b['S2_Band'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
geometry = np.take_along_axis(b['geometry'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)

ABS_SHIFT = np.take_along_axis(b['ABS_SHIFT'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
ANGLE= np.take_along_axis(b['ANGLE'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
L1_OUTLIER= np.take_along_axis(b['L1_OUTLIER'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
L2_OUTLIER= np.take_along_axis(b['L2_OUTLIER'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
L3_OUTLIER= np.take_along_axis(b['L3_OUTLIER'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
LAST_ERR= np.take_along_axis(b['LAST_ERR'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
OUTLIER= np.take_along_axis(b['OUTLIER'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
REF_BADDAT= np.take_along_axis(b['REF_BADDAT'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
SSIM_AFTER= np.take_along_axis(b['SSIM_AFTER'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
SSIM_BEFOR= np.take_along_axis(b['SSIM_BEFOR'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
SSIM_IMPRO= np.take_along_axis(b['SSIM_IMPRO'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
TGT_BADDAT= np.take_along_axis(b['TGT_BADDAT'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
X_SHIFT_M= np.take_along_axis(b['X_SHIFT_M'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
X_SHIFT_PX= np.take_along_axis(b['X_SHIFT_PX'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
X_WIN_SIZE= np.take_along_axis(b['X_WIN_SIZE'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
Y_SHIFT_M= np.take_along_axis(b['Y_SHIFT_M'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
Y_SHIFT_PX= np.take_along_axis(b['Y_SHIFT_PX'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)
Y_WIN_SIZE= np.take_along_axis(b['Y_WIN_SIZE'], x, axis=1) #ordinare in funzione di x (quindi della feature prescelta)


# In[ ]:


final_composite = np.stack((b['POINT_ID'],
                            X_IM[:,-1], #essendo in ordine decrescente, prendiamo l'ultima colonna così da avere i valori corrispondenti ai valori più alti della feature prescelta (x)
                            Y_IM[:,-1],
                            X_MAP[:,-1], 
                            Y_MAP[:,-1], 
                            RELIABILITY_sorted[:,-1],
                            S2_Band[:,-1],
                            
                            ABS_SHIFT[:, -1],
                            ANGLE[:, -1],
                            L1_OUTLIER[:, -1],
                            L2_OUTLIER[:, -1],
                            L3_OUTLIER[:, -1],
                            LAST_ERR[:, -1],
                            OUTLIER[:, -1],
                            REF_BADDAT[:, -1],
                            SSIM_AFTER[:, -1],
                            SSIM_BEFOR[:, -1],
                            SSIM_IMPRO[:, -1],
                            TGT_BADDAT[:, -1],
                            X_SHIFT_M[:, -1],
                            X_SHIFT_PX[:, -1],
                            X_WIN_SIZE[:, -1],
                            Y_SHIFT_M[:, -1],
                            Y_SHIFT_PX[:, -1],
                            Y_WIN_SIZE[:, -1],
                            geometry[:,-1]), 
                           axis=-1)



final_composite = gpd.GeoDataFrame(final_composite, columns= ['POINT_ID', 'X_IM', 'Y_IM', 'X_MAP', 'Y_MAP', 'RELIABILIT', 'S2_Band', 'ABS_SHIFT','ANGLE','L1_OUTLIER','L2_OUTLIER','L3_OUTLIER','LAST_ERR','OUTLIER','REF_BADDAT','SSIM_AFTER','SSIM_BEFOR','SSIM_IMPRO','TGT_BADDAT','X_SHIFT_M','X_SHIFT_PX','X_WIN_SIZE','Y_SHIFT_M','Y_SHIFT_PX','Y_WIN_SIZE', 'geometry'])

final_composite = final_composite.astype({'POINT_ID': 'int64', 
                        'X_IM':  'int64', 
                        'Y_IM':  'int64', 
                        'X_MAP':  'float64', 
                        'Y_MAP': 'float64' , 
                        'X_SHIFT_M': 'float64' ,
                        'X_SHIFT_PX': 'float64' ,
                        'X_WIN_SIZE': 'float64' ,
                        'Y_SHIFT_M': 'float64' ,
                        'Y_SHIFT_PX': 'float64' ,
                        'Y_WIN_SIZE': 'float64'})


# In[ ]:


'''GET GCPS FROM AROSICS Tie Points TABLE'''

#apply shifting to the points
#applying the modificaiton during the dataframe construction


'''
se volessimo usare un dataframe esterno. Ma attenzione perchè delle volte fallisce gdal.Warp
a causa del dtype del dataframe

points = gpd.read_file(path+'AROSICS_TiePoints_Local_HigherReliability_20200801.shp')

gcps = pd.DataFrame({'X': pd.to_numeric(points['X_MAP'])+pd.to_numeric(points['X_SHIFT_M']), 
                     'Y': pd.to_numeric(points['Y_MAP'])+pd.to_numeric(points['Y_SHIFT_M']),
                     'Col': pd.to_numeric(points['X_IM']), 
                     'Row': pd.to_numeric(points['Y_IM'])}) 
'''



#quindi, utilizziamo direttamente le colonne dell'arrey sorted, selezioniamo comunque i punti che hanno raggiunto RELIABILITY>70%:
gcps_VNIR_SWIR = pd.DataFrame({'X': X_MAP[:,-1][RELIABILITY_sorted[:,-1]>70]+X_SHIFT_M[:,-1][RELIABILITY_sorted[:,-1]>70], 
                     'Y': Y_MAP[:,-1][RELIABILITY_sorted[:,-1]>70]+Y_SHIFT_M[:,-1][RELIABILITY_sorted[:,-1]>70],
                     'Col': X_IM[:,-1][RELIABILITY_sorted[:,-1]>70], 
                     'Row': Y_IM[:,-1][RELIABILITY_sorted[:,-1]>70]})

#gcps for PANCHROMATIC image: since it is at 5m instead of 30m, the position of the X/Y-IM must be multiplied for 6
gcps_PAN = pd.DataFrame({'X': X_MAP[:,-1][RELIABILITY_sorted[:,-1]>70]+X_SHIFT_M[:,-1][RELIABILITY_sorted[:,-1]>70], 
                     'Y': Y_MAP[:,-1][RELIABILITY_sorted[:,-1]>70]+Y_SHIFT_M[:,-1][RELIABILITY_sorted[:,-1]>70],
                     'Col': X_IM[:,-1][RELIABILITY_sorted[:,-1]>70]*6, 
                     'Row': Y_IM[:,-1][RELIABILITY_sorted[:,-1]>70]*6})


'''
gcps = pd.DataFrame({'X': X_MAP[:,-1]+X_SHIFT_M[:,-1], 
                     'Y': Y_MAP[:,-1]+Y_SHIFT_M[:,-1],
                     'Col': X_IM[:,-1], 
                     'Row': Y_IM[:,-1]})




#gcps for PANCHROMATIC image: since it is at 5m instead of 30m, the position of the X/Y-IM must be multiplied for 6
gcps_PAN = pd.DataFrame({'X': X_MAP[:,-1]+X_SHIFT_M[:,-1], 
                     'Y': Y_MAP[:,-1]+Y_SHIFT_M[:,-1],
                     'Col': X_IM[:,-1]*6, 
                     'Row': Y_IM[:,-1]*6})
'''

#create GDAL GCPs
gcps_gdal_VNIR_SWIR = [gdal.GCP(row['X'], row['Y'],0, row['Col'], row['Row']) for index, row in gcps_VNIR_SWIR.iterrows()] ### "gcps_VNIR_SWIR" or "gcps_PAN"
gcps_gdal_PAN = [gdal.GCP(row['X'], row['Y'],0, row['Col'], row['Row']) for index, row in gcps_PAN.iterrows()] ### "gcps_VNIR_SWIR" or "gcps_PAN"


# In[ ]:


path = 'C:\\Users\\IBE_TEST\\Documents\\CNR-IBE\\Dati\\PRISMA_pansharpening_project\\ARBOREA\\'


# In[ ]:


kwargs = {
    'format': 'GTiff',
    'outputType': gdal.GDT_Float32}

output_image = path+'output_translate_poly2.tif'
ds_gcp = gdal.Translate(output_image, 
                        path+'PRS_L2D_STD_20200627102358_20200627102402_0001_VNIR_SWIR.tif', 
                        outputSRS='EPSG:32632', 
                        GCPs=gcps_gdal_VNIR_SWIR,
                        **kwargs)


options = gdal.WarpOptions(dstSRS='EPSG:32632', polynomialOrder=2, targetAlignedPixels=True, xRes=30, yRes =30)
ds = gdal.Warp(path+'AROSICS_output\\PRS_L2D_STD_20200627102358_20200627102402_0001_VNIR_SWIR_Warped_poly2aligned.tif', ds_gcp, dstNodata = np.nan, options=options)
ds_gcp = None
ds = None






output_image_pan = path+'output_translate_poly2_PAN.tif'
ds_gcp_pan = gdal.Translate(output_image_pan, 
                        path+'PRS_L2D_STD_20200627102358_20200627102402_0001_PAN.tif', 
                        outputSRS='EPSG:32632', 
                        GCPs=gcps_gdal_PAN,
                        **kwargs)


options_pan = gdal.WarpOptions(dstSRS='EPSG:32632', polynomialOrder=2, targetAlignedPixels=True, xRes=5, yRes =5)
ds_pan = gdal.Warp(path+'AROSICS_output\\PRS_L2D_STD_20200627102358_20200627102402_0001_PAN_Warped_poly2aligned.tif', ds_gcp_pan, dstNodata = np.nan, options=options_pan)
ds_gcp_pan = None
ds_pan = None


# In[ ]:




