from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
method = 'ssemnet'
img = imread('H:/bs/desData/{}/datapair.tif'.format(method))
max = np.max(img)

for i in range(img.shape[0]):
    if i%2==0:
        path = 'H:/bs/desData/{}/'.format(method) + str(int(i/2)) + '_warp1.tif'
        imsave(path, img[i])
    else:
        path = 'H:/bs/desData/{}/'.format(method) + str(int((i-1)/2)) + '_warp2.tif'
        imsave(path, img[i])