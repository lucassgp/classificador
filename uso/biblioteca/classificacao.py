# sequencia de codigos para carregar o modelo treinado e classificar a imagem inteira

# dados
import numpy as np

# ML
import joblib

# outros
import os
from time import time

# geograficos
from osgeo import gdal
import skimage.io as io


def classificar_arquivo(path_raster, model):
    '''Funcao para classificar um arquivo unico, exportando da mesma pasta da imagem original'''

    pasta_classificacao = os.path.dirname(path_raster)
    nome = path_raster.split('/')[-1].split('.')[0]

    if isinstance(model, str):
        print(f'Carregando modelo: {model}')
        classifier = joblib.load(model)
    else:
        classifier = model
    try:
        naip_fn = path_raster

        print(naip_fn)

        img_ds = io.imread(naip_fn)
        img = np.array(img_ds, dtype='float64')
        xshape = img[:, :, 0].shape

        # dividindo em 2x2
        block_img = np.zeros(xshape)
        im_h, im_w = img.shape[:2]
        bl_h, bl_w = int(im_h / 2), int(im_w / 2)
        for row in np.arange(im_h - bl_h + 1, step=bl_h):
            for col in np.arange(im_w - bl_w + 1, step=bl_w):
                new_image = img[row:row + bl_h, col:col + bl_w]
                new_shape = new_image.shape[0] * new_image.shape[1], new_image.shape[2]
                img_as_array = new_image[:, :, :6].reshape(new_shape)

                start_time = time()

                class_prediction = classifier.predict(img_as_array)
                class_prediction = class_prediction.reshape(new_image[:, :, 0].shape)

                block_img[row:row + bl_h, col:col + bl_w] = class_prediction
                print("--- %s seconds ---" % (time() - start_time))

        driverTiff = gdal.GetDriverByName('GTiff')
        naip_ds = gdal.Open(naip_fn)
        nbands = naip_ds.RasterCount

        classification = os.path.join(pasta_classificacao, f'classificada_{nome}.tif')
        classification_ds = driverTiff.Create(classification, naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Byte)
        classification_ds.SetGeoTransform(naip_ds.GetGeoTransform())
        classification_ds.SetProjection(naip_ds.GetProjectionRef())
        classification_ds.GetRasterBand(1).WriteArray(block_img)
        classification_ds = None
        naip_ds = None
    except Exception as e:
        print(e)

    print('Classificado')