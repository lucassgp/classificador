# sequencia de codigos para extrair as informacoes da imagem e converter para DataFrame

# importacoes

# geograficos
import geopandas as gpd
from osgeo import gdal
import skimage.io as io

# dados
import pandas as pd
import numpy as np

# graficos
import matplotlib.pyplot as plt
import seaborn as sns

# outros
import tempfile
import sys

def validar_dados(amostra:str, raster:str):
    retorno = {}
    erro = []
    # testes de abertura
    imagem=gdal.Open(raster)

    if imagem is None:
        print('Erro na abertura da imagem')
        sys.exit()

    else:
        retorno['raster'] = imagem

    try:
        gdf_amostras=gpd.read_file(amostra)
        gdf_amostras['id'] = gdf_amostras.index
        retorno['amostras'] = gdf_amostras
    except:
        print('Erro na abertura das amostras')
        sys.exit()

    # Procurando coluna label
    if not 'label' in gdf_amostras:
        erro.append('Arquivo sem a coluna "label"')

    # Geometrias nulas
    if len(gdf_amostras[(gdf_amostras['geometry'].is_empty) | (gdf_amostras['geometry'] == None)])>0:
        erro.append('Geometrias nulas detectadas')

    # Geometrias invalidas
    if False in gdf_amostras.is_valid:
        erro.append('Geometrias inválidas detectada')

    # Tamanho da imagem
    xsize = imagem.RasterXSize
    ysize = imagem.RasterYSize
    total_size = xsize*ysize

    if total_size > 50000000:
        erro.append(f'Limite de pixels atingido {total_size} | x: {xsize} / {ysize}')

    # Mesmo CRS
    if not imagem.GetProjection() == gdf_amostras.crs:
        erro.append('Imagem e amostras não possuem o mesmo CRS')

    if len(erro)>0:
        print('Arquivos reprovados pela validação')
        print(*erro, sep='\n')

    retorno['erro'] = erro
    return retorno

def listaArray_arquivo(shp_amt, vrt_base, pasta_reclass):
    '''
    Transforma em array dados de um raster recortado por um shp de amostra.
    Esse script trabalha com um arquivo raster e um arquivo shp como amostra.
    Util para analise mais especificas de dados
    '''

    amostra = list()
    imagem = list()

    naip_ds = gdal.Open(vrt_base)
    nbands = naip_ds.RasterCount

    gdf = shp_amt

    reclasse     = pasta_reclass + '/reclassific.shp'
    reclasse_tif = pasta_reclass + '/reclassific_raster.tif'
    gdf.to_file(reclasse)

    gdal.UseExceptions()
    dataset_poligonos = gdal.OpenEx(reclasse, gdal.OF_VECTOR)

    gdal_driver = gdal.GetDriverByName('GTiff')  # Salva em arquivo - PARA DESENVOLVIMENTO.

    raster_dados = gdal_driver.Create(reclasse_tif, naip_ds.RasterXSize, naip_ds.RasterYSize, 1, gdal.GDT_Byte)

    raster_dados.SetGeoTransform(naip_ds.GetGeoTransform())
    raster_dados.SetProjection(naip_ds.GetProjection())

    raster_dados.GetRasterBand(1).SetNoDataValue(9999)
    gdal.Rasterize(raster_dados, dataset_poligonos, options=gdal.RasterizeOptions(attribute="id"))
    matriz_talhoes = raster_dados.ReadAsArray()

    amostra.append(matriz_talhoes)

    img_ds = io.imread(vrt_base)
    img = np.array(img_ds, dtype='float64')

    imagem.append(img)

    return amostra, imagem, nbands

def getxy(amostra, imagem, n):
    '''Funcao para separar o X e Y das amostras, retornados pela funcao "listaArray_arquivo" e retornar como um array "tabelavel"'''

    amostras = np.concatenate(tuple(amostra))
    imagens = np.concatenate(tuple(imagem))

    amostras.shape, imagens.shape
    try:
        X_train = imagens[amostras<=n,:]
    except:
        X_train = imagens[amostras<=n]

    Y_train = amostras[amostras<=n]

    return X_train, Y_train

def exploratoria(dados_brutos, amostras):
    '''Plota um boxplot e um histplot por banda, segmentada pelo label'''

    list_colunas = list(dados_brutos.drop(columns='id').columns)
    dados_graficos = pd.merge(dados_brutos, amostras[['label', 'id']], on='id')

    sns.countplot(x=dados_graficos.label.astype(int))
    plt.suptitle(f'Distribuição das amostras\nNúmero de pixels amostrados: {len(dados_graficos)}')
    plt.show()

    for banda in list_colunas:
        dados = dados_graficos[[banda, 'label']]
        fig, ax = plt.subplots(1,2)
        sns.boxplot(dados, y=banda, x='label', ax=ax[0])
        sns.histplot(dados, x=banda ,hue='label', element='step', fill=False, bins=50, ax=ax[1])
        plt.suptitle(f'Análise exploratória das amostras da banda: {banda}')
        plt.show()

def tabelar(shp: gpd.GeoDataFrame, raster:str, reclass=tempfile.mkdtemp(), graficos=True) -> pd.DataFrame:
    '''Organiza os valores das bandas de cada pixel em um dataframe'''
    amt, img, nbandas = listaArray_arquivo(shp, raster, reclass)
    x_amt, y_amt = getxy(amt, img, len(shp))

    colunas = [f'Banda_{x+1}' for x in range(nbandas)]

    tabela_x = pd.DataFrame(x_amt, columns=colunas)
    tabela_y = pd.DataFrame(y_amt, columns=['id'])

    tabela = pd.concat([tabela_x, tabela_y], axis=1)

    if graficos:
        exploratoria(tabela, shp)

    return tabela




