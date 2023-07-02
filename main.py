import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import rasterio
from dwf import *
import matplotlib.pyplot as plt

blue = rasterio.open('./data/LT05_037034_20090420_Blue.tif').read(1)
green = rasterio.open('./data/LT05_037034_20090420_Green.tif').read(1)
red = rasterio.open('./data/LT05_037034_20090420_Red.tif').read(1)
nir = rasterio.open('./data/LT05_037034_20090420_NIR.tif').read(1)
swir1 = rasterio.open('./data/LT05_037034_20090420_SWIR1.tif').read(1)
swir2 = rasterio.open('./data/LT05_037034_20090420_SWIR2.tif').read(1)
qa = rasterio.open('./data/LT05_037034_20090420_QA.tif').read(1)

result = dwf_prediction(blue, green, red, nir, swir1, swir2, qa)

plt.figure(figsize=(20, 20))
plt.imshow(result)
plt.title('DWFPrediction', fontsize=20)
plt.savefig('./result.png')

np.save('./result.npy', result)
