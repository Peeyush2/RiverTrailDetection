#code starts from line 106
import cv2
import numpy as np
import os

path = r'./assets'
os.chdir(path)

def create_gaborfilter(ksize = 40, sigma = 1.0, lambd = 3.0, gamma = 0.5):
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 1024
    psi = 1.0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
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

def connectedComponentsM(img) :

    #step6 - morph close followed by dilation
    #morph close ( to fill the small holes in the image)
    dilation_size = 4
    
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE ,element)

    cv2.imshow("morph close", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))

    #dilating to connect the river parts
    img = cv2.dilate(img, element)
    

    #step7 - thresholding the image
    #selecting pixels above 127 threshold
    #making rest of the pixels 0
    ret, binary_map = cv2.threshold(img,127,255,0)

    #step8 - calculating connected components
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    #making a blank canvas to plot the connected components
    result = np.zeros((labels.shape), np.uint8)

    #step9 - selecting connected component followed by erosion
    for i in range(0, nlabels - 1):
        if areas[i] >= 10000:   #threshold for area, selecting area greater than 10,000 ( may subject to change on different areas)
            #setting pixel value to white for area above 10,000
            result[labels == i + 1] = 255
    
    cv2.imshow("conected components extracted", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #erosion
    erosion_size = 5

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (-1, -1))

    dilatation_dst = cv2.erode(result ,element)


    #dst = cv2.Canny(dilatation_dst,100, 200, apertureSize = 5 )
    cv2.imshow("final result", dilatation_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dilatation_dst





#starting point
#step-1
#reading image
image = cv2.imread('croppedNdwi.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

#step-2 generating gabor filters
gfilters = create_gaborfilter()

#step-3 applying gabor filter
upated_ndwi = apply_filter(image, gfilters)

cv2.imshow('Gabor applied', upated_ndwi)

#step-4 edge detection
#converting converting image to datatype unit8 and applying canny
upated_ndwi = (upated_ndwi*255).astype(np.uint8)
dst = cv2.Canny(upated_ndwi,100, 200, apertureSize = 5 )

#step-5 removing straight lines
# generating possible lines ( for line detection in image)
lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 20, None, 20, 1)
onlyLines = np.zeros_like(dst)

if lines is not None:
    print(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        #detected lines
        cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv2.LINE_AA)

        #removing detected lines from image
        cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,0), 1, cv2.LINE_AA)
else:
    print(lines)

#image of the lines removed 
cv2.imshow('removed straight lines', onlyLines)
cv2.waitKey()
cv2.destroyAllWindows()

#image after removing lines
cv2.imshow('image after removing lines', dst)
cv2.waitKey()


dst = connectedComponentsM(dst)

#dst contains the final result


