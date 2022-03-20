import cv2
import numpy as np
import math
import os
import rasterio
from matplotlib import pyplot 
from dilation import runDilationFunction

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 1000
    ksize =  100  # The local area to evaluate 1000
    sigma = 1.0  # Larger Values produce more edges  4.0
    lambd = 4.0 # 4
    gamma = 0.5
    psi = 1  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        #pyplot.imshow(kern)
        #pyplot.show()  
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def create_gaborfilter2(ksize = 100, sigma = 1.0, lambd = 4.0, gamma = 0.5):
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 1000
    psi = 1  # Offset value - lower generates cleaner results
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




#path = r'C:\Users\peeyu\Projects\Research Paper\QgisFiles'
path = r'C:\Users\peeyu\Projects\Research Paper\js2'
#path = r'C:\Users\peeyu\Projects\Research Paper\Mesra Qgis'
os.chdir(path)

# Reading the image
# aweiCropped
# NDVICropped
# ndwiCropped
# ndwiCroppedJharia
# croppedNdwi3
# clippedJharia4
# clippedJharia5
# croppedJharia6
# type2CroppedJharia
# type2ClippedJharia3
# type2NdwiClipped6
# type2CroppedJharia4
# clippedMesra
# croppedJharia
# jhariaRaster2
image = cv2.imread('croppedJhariaRaster.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)
srcRas = rasterio.open('croppedJhariaRaster.tif')
array = srcRas.read(1)
cv2.imshow('nothing applied', array)
cv2.waitKey()
pyplot.axis('off')
pyplot.title('original image')
pyplot.imshow(array)
pyplot.show()  

array = cv2.GaussianBlur( image, [5,5],0 )
pyplot.axis('off')
pyplot.title('Gaussian Blur Applied')
pyplot.imshow(array)
pyplot.show()  
gfilters = create_gaborfilter2()
upated_ndwi = apply_filter(array, gfilters)

pyplot.axis('off')
pyplot.title('Gabor Applied')
pyplot.imshow(upated_ndwi)
#cv2.imshow('gabor applied', upated_ndwi)
pyplot.show()  

upated_ndwi = (upated_ndwi*255).astype(np.uint8)
dst = cv2.Canny(upated_ndwi,100, 200, apertureSize = 5 )
#dst = upated_ndwi

pyplot.axis('off')
pyplot.title('Canny Applied')
pyplot.imshow(dst)
pyplot.show()

cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, None, 1, 1)
onlyLines = np.zeros_like(cdst)
   
if lines is not None:
    print(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
else:
    print(lines)
cv2.imshow('Canny', dst)

pyplot.axis('off')
pyplot.title('Hough line Applied')
pyplot.imshow(onlyLines)
pyplot.show()  

print('running dilation')

runDilationFunction(cdst)

#cv2.imshow('lines', )
cv2.waitKey()
cv2.destroyAllWindows()
#pyplot.savefig('foo.png')
#cv2.imwrite('upatedNdwiResult.tif',upated_ndwi )



#img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)


#cv2.imshow('Original', image)

#cv2.imshow?




#pyplot.imshow(array)


# Creating the kernel(2d convolution matrix)
"""
ksize = 15  #Use size that makes sense to the image and fetaure size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5 #Large sigma on small features will fully miss the features. 
theta = 1*np.pi  #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/4  #1/4 works best for angled. 
gamma= 0  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 1  #Phase offset. I leave it to 0. (For hidden pic use 0.8)

kernel = cv2.getGaborKernel((ksize, ksize), sigma, 1*np.pi/4, lamda, gamma, phi, ktype=cv2.CV_32F)

kerne2 = cv2.getGaborKernel((ksize, ksize), sigma, 1*np.pi, lamda, gamma, phi, ktype=cv2.CV_32F)

#kernel1 = np.ones((5, 5), np.float32)/30
  
# Applying the filter2D() function
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
img2 = cv2.filter2D(src=image, ddepth=-1, kernel=kerne2)
  
# Shoeing the original and output image
#cv2.imshow('Original', array)
#cv2.imshow('Kernel Blur', img)
#cv2.imshow('Kernel Blur2', img2)

#pyplot.imshow(img2)
#pyplot.show() 
  
cv2.waitKey()
cv2.destroyAllWindows()
"""