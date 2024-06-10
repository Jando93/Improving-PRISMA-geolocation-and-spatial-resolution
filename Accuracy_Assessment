from osgeo import gdal, osr, gdal_array
import json, re, itertools, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path_HSMS = 'path_HSMS\\' #path where fused HSMS PRISMA output images are stored
path_HSPAN = 'path_HSPAN\\' #path where pansharpened HSPAN PRISMA output images are stored

S2 = 'path_sentinel\\Sentinel-2'

reference_airborne_VNIR_image_30m = 'path_airborne\\VNIR_30m'
reference_airborne_VNIR_image_10m = 'path_airborne\\VNIR_10m'
reference_airborne_VNIR_image_5m = 'path_airborne\\VNIR_5m'

reference_airborne_SWIR_image_30m =  'path_airborne\\SWIR_30m
reference_airborne_SWIR_image_10m =  'path_airborne\\SWIR_10m
reference_airborne_SWIR_image_5m = 'path_airborne\\SWIR_5m'




=============================================================================================================
'''ACCURACY METRICS'''
=============================================================================================================
def get_bands(image):
    image = gdal.Open(image)
    image = image.ReadAsArray()
    n_bands, y, x = image.shape
    return image, n_bands, y, x


def iteration(path):
    dirFileList = os.listdir(path)
    lista_files=[]
    name_files=[]
    for file in os.listdir(path):
        if os.path.splitext(file)[-1] == '.tif':
            lista_files.append(os.path.join(path, file))
            name_files.append(file.replace('.tif',''))
    return lista_files, name_files


def save_as_tiff_singleband(reference, array, x, y, output):
    reference = gdal.Open(reference)
    geotransform = reference.GetGeoTransform()
    wkt = reference.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    new_img = driver.Create(output,
                        x, #X_size
                        y, #Y_size
                        1,
                        gdal.GDT_Float32)
    new_img.GetRasterBand(1).WriteArray(array)
    new_img.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    new_img.SetProjection( srs.ExportToWkt() )
    new_img = None


def RMSE(image, reference):
    rmsep = np.sqrt(np.mean((image-reference)**2, axis=0))
    rmseb = np.sqrt(np.mean((image-reference)**2, axis=1))
    rmsep_mean= rmsep.mean()
    rmseb_mean= rmseb.mean()
    rmsep_std = rmsep.std()
    rmseb_std = rmseb.std()
    return rmsep, rmseb, rmsep_mean,rmseb_mean, rmsep_std, rmseb_std


def SAM(image, reference):
    sam = np.arccos(np.sum(image*reference, axis=0)/(np.linalg.norm(image, axis=0)*np.linalg.norm(reference, axis=0)))
    sam_mean = np.nanmean(sam)
    sam_std = np.nanstd(sam)
    return sam, sam_mean, sam_std

def PSNR(image, reference):
    psnr = 10*np.log10((np.nanmax(image, axis=1)**2)/(np.linalg.norm(reference-image, axis=1)**2/image.shape[1]))
    psnr_mean = np.nanmean(psnr)
    psnr_std = np.nanstd(psnr)
    return psnr, psnr_mean, psnr_std

def ERGAS(image, reference, ratio):
    rmse= np.sqrt(np.nanmean((image-reference)**2, axis=1))
    media = np.nanmean(image, axis=1)
    ergas = (100/ratio)*np.sqrt(np.nanmean((rmse/media)**2))
    return ergas

def UIQUI(image, reference):
    lista_uniqui_b=[]
    for n in range(len(image)):
        cov = np.cov(image[n], reference[n])[1][0]
        '''
        The numpy.cov() function returns a 2D array in which the value at index [0][0] is the covariance between a1 and a1,
        the value at index [0][1] is the covariance between a1 and a2, 
        the value at index [1][0] is the covariance between a2 and a1, 
        the value at index [1][1] is the covariance between a2 and a2
        '''
        uiqi = (4*cov*image[n].mean()* reference[n].mean()) / ((image[n].std()**2+reference[n].std()**2)*(image[n].mean()**2+reference[n].mean()**2))
        lista_uniqui_b.append(uiqi)
    uiqi_mean = np.array(lista_uniqui_b).mean()
    uiqi_std = np.array(lista_uniqui_b).std()
    return lista_uniqui_b, uiqi_mean, uiqi_std



def calculate_indices(name=None, fused_file=None, reference_VNIR_file=None, reference_SWIR_file=None, 
                      ratio=None #ratio between GSDs of original HS image and MS or PAN image):
    fused, n_bands, y, x = get_bands(fused_file)
    fused = np.reshape(fused, (n_bands, y*x))
    
    n_bands = fused.shape[0]
    
    if reference_VNIR_file is not None and reference_SWIR_file is not None:
        reference_VNIR = reference_VNIR_file.ReadAsArray()
        reference_SWIR = reference_SWIR_file.ReadAsArray()
        reference = np.vstack((reference_VNIR, reference_SWIR))
        reference = np.reshape(reference, (n_bands, y*x))
                
        reference = reference/10000   #eventually: rescale pixel values at the same scale of fused/pansharpened scale; [0-1] in this specific scale
        
        reference_VNIR = None
        reference_SWIR = None
        
        rmsep, rmseb, rmsep_mean,rmseb_mean, rmsep_std, rmseb_std = RMSE(fused, reference)
        sam, sam_mean, sam_std = SAM(fused, reference)
        psnr, psnr_mean, psnr_std = PSNR(fused, reference)
        ergas = ERGAS(fused, reference, ratio)
        lista_uniqui_b, uiqi_mean, uiqi_std = UIQUI(fused, reference)
        
        #save values to excel
        excel_file = 'path_output_indices_tabled\\Indices_{}.xlsx'.format(name)
            
        with pd.ExcelWriter(excel_file) as writer:
            pd.Series(rmseb).to_excel(writer, sheet_name="RMSEb", index=False)
            pd.Series(rmseb).to_excel(writer, sheet_name="RMSEb", index=False)
            pd.Series(rmsep_mean).to_excel(writer, sheet_name="RMSEp_mean", index=False)
            pd.Series(rmsep_std).to_excel(writer, sheet_name="RMSEp_std", index=False)
            pd.Series(rmseb_mean).to_excel(writer, sheet_name="RMSEb_mean", index=False)
            pd.Series(rmseb_std).to_excel(writer, sheet_name="RMSEb_std", index=False)
            
            pd.Series(sam_mean).to_excel(writer, sheet_name="SAM_mean", index=False)
            pd.Series(sam_std).to_excel(writer, sheet_name="SAM_std", index=False)
            
            pd.Series(psnr).to_excel(writer, sheet_name="PSNR", index=False)
            pd.Series(psnr_mean).to_excel(writer, sheet_name="PSNR_mean", index=False)
            pd.Series(psnr_std).to_excel(writer, sheet_name="PSNR_std", index=False)
            
            pd.Series(ergas).to_excel(writer, sheet_name="ERGAS", index=False)
            
            pd.Series(lista_uniqui_b).to_excel(writer, sheet_name="UNIQUIb", index=False)
            pd.Series(uiqi_mean).to_excel(writer, sheet_name="UNIQUI_mean", index=False)
            pd.Series(uiqi_std).to_excel(writer, sheet_name="UNIQUI_std", index=False)

        #save to image
        rmsep = np.reshape(rmsep, (y, x))
        sam = np.reshape(sam, (y, x))
        
        tiff_file = 'path\\'
        save_as_tiff_singleband(fused_file, rmsep, x, y,  tiff_file+"RMSEp_"+name+".tif")
        save_as_tiff_singleband(fused_file, sam, x, y,  tiff_file+"SAM_"+name+".tif")
    
    
    if reference_VNIR_file is not None and reference_SWIR_file is None:
        reference = reference_VNIR_file.ReadAsArray()
        reference = np.reshape(reference, (reference.shape[0], y*x))
        
        reference = reference/10000
        
        n_bands = fused.shape[0]
        
        rmsep, rmseb, rmsep_mean,rmseb_mean, rmsep_std, rmseb_std = RMSE(fused, reference)
        sam, sam_mean, sam_std = SAM(fused, reference)
        psnr, psnr_mean, psnr_std = PSNR(fused, reference)
        ergas = ERGAS(fused, reference, ratio)
        lista_uniqui_b, uiqi_mean, uiqi_std = UIQUI(fused, reference)

        #save values to excel
        excel_file = 'path_output_indices_tabled\\Indices_{}.xlsx'.format(name)
            
        with pd.ExcelWriter(excel_file) as writer:
            pd.Series(rmseb).to_excel(writer, sheet_name="RMSEb", index=False)
            pd.Series(rmsep_mean).to_excel(writer, sheet_name="RMSEp_mean", index=False)
            pd.Series(rmsep_std).to_excel(writer, sheet_name="RMSEp_std", index=False)
            pd.Series(rmseb_mean).to_excel(writer, sheet_name="RMSEb_mean", index=False)
            pd.Series(rmseb_std).to_excel(writer, sheet_name="RMSEb_std", index=False)
            
            pd.Series(sam_mean).to_excel(writer, sheet_name="SAM_mean", index=False)
            pd.Series(sam_std).to_excel(writer, sheet_name="SAM_std", index=False)
            
            pd.Series(psnr).to_excel(writer, sheet_name="PSNR", index=False)
            pd.Series(psnr_mean).to_excel(writer, sheet_name="PSNR_mean", index=False)
            pd.Series(psnr_std).to_excel(writer, sheet_name="PSNR_std", index=False)
            
            pd.Series(ergas).to_excel(writer, sheet_name="ERGAS", index=False)
            
            pd.Series(lista_uniqui_b).to_excel(writer, sheet_name="UNIQUIb", index=False)
            pd.Series(uiqi_mean).to_excel(writer, sheet_name="UNIQUI_mean", index=False)
            pd.Series(uiqi_std).to_excel(writer, sheet_name="UNIQUI_std", index=False)
              
        #save to image
        rmsep = np.reshape(rmsep, (y, x))
        sam = np.reshape(sam, (y, x))
        tiff_file = 'path\\'
        save_as_tiff_singleband(fused_file, rmsep, x, y,  tiff_file+"RMSEp_"+name+".tif")
        save_as_tiff_singleband(fused_file, sam, x, y, tiff_file+"SAM_"+name+".tif")



'''
#To apply indices calculation:


fused_HSMS_files = iteration(path_HSMS)[0]
fused_HSMS_files_names = iteration(path_HSMS)[1]
im = 0
while im < len(fused_HSMS_files):
    calculate_indices(name=fused_HSMS_files_names[im],
             fused_file = fused_HSMS_files[im],
             reference_VNIR_file=reference_airborne_VNIR_image_10m,
             reference_SWIR_file=reference_airborne_SWIR_image_10m, #reference SWIR file was excluded for ARBOREA study area
             ratio = 30/10)
    im+=1


pansharped_HSPAN_files = iteration(path_HSPAN)[0]
pansharped_HSPAN_names = iteration(path_HSPAN)[1]
im = 0
while im < len(pansharped_HSPAN_files):
    calculate_indices(name=pansharped_HSPAN_names[im],
             fused_file = pansharped_HSPAN_files[im],
             reference_VNIR_file=reference_airborne_VNIR_image_5m,
             reference_SWIR_file=reference_airborne_SWIR_image_5m,  #reference SWIR file was excluded for ARBOREA study area
             ratio = 30/5)
    im+=1
'''




