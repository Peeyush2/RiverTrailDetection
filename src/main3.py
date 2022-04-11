import cv2
import numpy as np
import math
import os
import rasterio
from osgeo import gdal
from matplotlib import pyplot 
from dilation import erosion, runDilationFunction
from connectedComponents import connectedComponents
from skimage.transform import (hough_line, hough_line_peaks)


path = r'C:\Users\peeyu\Projects\Research Paper\jharia4'
os.chdir(path)
# Ksize = 40, 
# ksize = 10
def create_gaborfilter2(ksize = 40, sigma = 1.0, lambd = 3.0, gamma = 0.5):
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 1024
    psi = 1.0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        #pyplot.imshow(kern)
        #pyplot.show()  
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
    # This general function is designed to apply filters to our image
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

# croppedMndwi
# croppedNdwi
# croppedAwei
# croppedNdmi
image = cv2.imread('croppedNdwi2.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

# image2 = cv2.imread('croppedBI.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

gfilters = create_gaborfilter2()
upated_ndwi = apply_filter(image, gfilters)

# gfilters = create_gaborfilter2(ksize= 200, sigma= 1, lambd= 4, gamma= 0.5)
# upated_ndwi2 = apply_filter(image2, gfilters)

cv2.imshow('Gabor applied', upated_ndwi)
cv2.waitKey()
cv2.destroyAllWindows()

# upated_ndwi = upated_ndwi2

# runDilationFunction(upated_ndwi)

upated_ndwi = (upated_ndwi*255).astype(np.uint8)
dst = cv2.Canny(upated_ndwi,100, 200, apertureSize = 5 )

# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
# hspace, theta, dist = hough_line(upated_ndwi, tested_angles)

# pyplot.figure()
# pyplot.imshow(hspace)  
# pyplot.show()

cv2.imshow('canny applied only', dst)
cv2.waitKey()
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 20, None, 10, 1)
onlyLines = np.zeros_like(dst)
   
if lines is not None:
    print(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,0), 1, cv2.LINE_AA)
else:
    print(lines)

cv2.imshow('removed straight lines', dst)
cv2.waitKey()
cv2.destroyAllWindows()
# scale = 1
# delta = 1
# ddepth = cv2.CV_16S
# grad_x = cv2.Sobel(upated_ndwi, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# cv2.imshow('grad_x', grad_x)
# cv2.waitKey()
# cv2.destroyAllWindows()


dst = connectedComponents(dst)

dst = cv2.Canny(dst,100, 200, apertureSize = 5 )

cv2.imshow('connected components applied', dst)
cv2.waitKey()
cv2.destroyAllWindows()


# cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, None, 1, 1)
# onlyLines = np.zeros_like(cdst)
   
# if lines is not None:
#     print(len(lines))
#     for i in range(0, len(lines)):
#         l = lines[i][0]
#         cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv2.LINE_AA)
#         #cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
# else:
#     print(lines)
# cv2.imshow('Hough Lines', onlyLines)
# cv2.waitKey()
# cv2.destroyAllWindows()

# croppedMndwi
# croppedNdwi
# croppedAwei
# croppedNdmi
#croppedNdrli
#croppedRi

# lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, None, 10, 1)
# onlyLines = np.zeros_like(dst)
   
# if lines is not None:
#     print(len(lines))
#     for i in range(0, len(lines)):
#         l = lines[i][0]
#         cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv2.LINE_AA)
#         cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,0), 1, cv2.LINE_AA)
# else:
#     print(lines)

# cv2.imshow('Canny', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()