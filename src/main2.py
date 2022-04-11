import cv2
import numpy as np
import math
import os
import rasterio
from osgeo import gdal
from matplotlib import pyplot 
from dilation import erosion, runDilationFunction
from connectedComponents import connectedComponents

#path = r'C:\Users\peeyu\Projects\Research Paper\QgisFiles'
path = r'C:\Users\peeyu\Projects\Research Paper\js2'
#path = r'C:\Users\peeyu\Projects\Research Paper\Jharia3'
#path = r'C:\Users\peeyu\Projects\Research Paper\Mesra Qgis'
os.chdir(path)

image = cv2.imread('croppedJhariaRaster.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

array = cv2.GaussianBlur( image, [5,5],0 )
erosion_size = 5
array = (array*255).astype(np.uint8)
# array = cv2.bitwise_not(array)
# runDilationFunction(array)
array2 = array


element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
img = cv2.morphologyEx(array, cv2.MORPH_CLOSE ,element)

erosion_size2 = 1
element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size2 + 1, 2 * erosion_size2 + 1),
                                       (erosion_size2, erosion_size2))
dilatation_dst = cv2.erode(array2 ,element2)

cv2.imshow("morph close", img)
cv2.imshow("erode", dilatation_dst)
cv2.waitKey()