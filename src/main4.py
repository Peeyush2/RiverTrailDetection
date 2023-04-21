from random import randint, random, sample
import cv2
import numpy as np
import math
import os
import rasterio
from osgeo import gdal
from matplotlib import pyplot
from connecteComponentM import connectedComponentsM 
from dilation import erosion, runDilationFunction
from skimage.transform import (hough_line, hough_line_peaks)
import pandas as pd

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


#starting
# croppedMndwi
# croppedNdwi
# croppedAwei
# croppedNdmi
image = cv2.imread('croppedNdwi.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

# image2 = cv2.imread('croppedBI.tif',cv2.IMREAD_LOAD_GDAL | cv2.IMREAD_ANYCOLOR)

image2 = cv2.imread('resize3.png')

gfilters = create_gaborfilter2()
upated_ndwi = apply_filter(image, gfilters)
upated_ndwi2 = upated_ndwi
# gfilters = create_gaborfilter2(ksize= 200, sigma= 1, lambd= 4, gamma= 0.5)
# upated_ndwi2 = apply_filter(image2, gfilters)

# half = cv2.rotate(image2, cv2.ROTATE_90_COUNTERCLOCKWISE)

# half = half[9:]


half = cv2.resize(image2, (len(image[0]), len(image)))




print( len(upated_ndwi), len(upated_ndwi[0]) ) 
print( len(half), len(half[0]) )

extractedBlue = np.zeros_like( half )
accurateBlue = np.zeros_like( upated_ndwi )
extractedNoise = np.zeros_like( half )

print("blue blue")
print(upated_ndwi.shape)

print("blue blue2")
print(half.shape)


for x in range(len(half)):
    for y in range(len(half[0])):
        newVal = half[x][y]
        if newVal[2] == 0 and newVal[1] != 0 and  newVal[0] != 0:
            extractedBlue[x][y] = newVal
            extractedNoise[x][y] = [0,0,0]
        else :
            extractedBlue[x][y] = [0,0,0]
            extractedNoise[x][y] = newVal

for x in range(len(extractedBlue)):
    for y in range(len(half[0])):
        newVal = extractedBlue[x][y]
        if newVal[2] == 0 and newVal[1] != 0 and  newVal[0] != 0:
            if upated_ndwi[x][y] != 0 :
                accurateBlue[x][y] = upated_ndwi[x][y]

print("upated_ndwi shap")
print(upated_ndwi.shape)
print( accurateBlue.shape )
# dilatation_size = 2

# element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
#                                        (-1, -1))

# extractedBlue = cv2.erode(extractedBlue ,element)

print(upated_ndwi)
print(half)

M = np.float32([[1, 0, -10], [0, 1, 0]])
shifted = extractedBlue #cv2.warpAffine(extractedBlue, M, (extractedBlue.shape[1], extractedBlue.shape[0]))
cv2.imshow("only accurate river", accurateBlue)

# for x in range(len(shifted)):
#     for y in range(len(shifted[0])):
#         newVal = shifted[x][y]
#         if newVal[2] == 0 and newVal[1] != 0 and  newVal[0] != 0:
#             upated_ndwi[x][y] = 0.5

cv2.imshow('Gabor applied', upated_ndwi)
cv2.imshow('new image applied2', extractedNoise)
cv2.imwrite("newblue.png", extractedBlue )
cv2.waitKey()
cv2.destroyAllWindows()

extractedBlue = accurateBlue

# upated_ndwi = upated_ndwi2

# runDilationFunction(upated_ndwi)

upated_ndwi = (upated_ndwi*255).astype(np.uint8)
dst = cv2.Canny(upated_ndwi,100, 200, apertureSize = 5 )
dst2 = cv2.Canny(upated_ndwi,100, 200, apertureSize = 5)

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
hspace, theta, dist = hough_line(upated_ndwi, tested_angles)

pyplot.figure()
pyplot.imshow(hspace)  
pyplot.show()

cv2.imshow('canny applied only', dst)
cv2.waitKey()
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 20, None, 20, 1)
onlyLines = np.zeros_like(dst)
   
if lines is not None:
    print(len(lines))
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(onlyLines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(dst, (l[0], l[1]), (l[2], l[3]), (0,0,0), 1, cv2.LINE_AA)
else:
    print(lines)

cv2.imshow('removed straight lines', onlyLines)
cv2.waitKey()
cv2.destroyAllWindows()
# scale = 1
# delta = 1
# ddepth = cv2.CV_16S
# grad_x = cv2.Sobel(upated_ndwi, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# cv2.imshow('grad_x', grad_x)
# cv2.waitKey()
# cv2.destroyAllWindows()


dst = connectedComponentsM(dst)

# dst = cv2.Canny(dst,100, 200, apertureSize = 5 )

count = 0
count2 = 0

count3 = 0
count4 = 0
count5 = 0
for x in range(len(half)):
    for y in range(len(half[0])):
        flag = 0
        if extractedBlue[x][y] >0 :
            flag = 1
            count += 1 #expected count
            if dst[x][y] != 0 :
                count2 += 1 #true positive
            else : 
                count5 += 1 #false positive
        elif dst[x][y] > 0 :
            count3+=1 #true negative
        else :
            count4 += 1 #false NEGATIVE

totalCountOriginal = 0
totalCountOutput = 0

for x in range(len(half)):
    for y in range(len(half[0])):
        if extractedBlue[x][y] >0 :
            totalCountOriginal += 1 #expected count
        if dst[x][y] != 0 :
            totalCountOutput += 1 #true positive


print("count",count2, count5, count3, count4)
print("total count original and output",totalCountOriginal, totalCountOutput )
cv2.imshow('result', dst)
cv2.imshow('input', dst2)
cv2.waitKey()
cv2.destroyAllWindows()


for x in range(len(dst)):
    for y in range(len(dst[0])):
        newVal = dst[x][y]
        if newVal > 0:
            upated_ndwi2[x][y] = 0

count5 = 0
count6 = 0
for x in range(len(extractedNoise)):
    for y in range(len(extractedNoise[0])):
        if extractedNoise[x][y][0] != 0 or extractedNoise[x][y][1] != 0 or extractedNoise[x][y][2] != 0 :
            count5 += 1
            if upated_ndwi2[x][y] > 0 :
                count6 += 1

print("false neg", count5, count6)
cv2.imshow('noise real', extractedNoise)
cv2.imshow('noise fake', upated_ndwi2)
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

count = 100
score = []
i = 0
xyArray = []
xyArray2 = []

count = 0
count2 = 0
for x in range(len(extractedBlue)):
    for y in range(len(extractedBlue[0])):
        newVal = extractedBlue[x][y] 
        if newVal > 0 : #[2] == 0 and newVal[1] != 0 and  newVal[0] != 0:
            xyArray.append([x,y])
            count += 1
        if dst[x][y] > 0 :
            count2 += 1 
            xyArray2.append([x,y])

print("count checking ", count, count2)
lenghtXy = len(xyArray)

scoreList = []
for i in range(10):
    tempScore = [0,0]
    xyArrayTemp = sample(xyArray, 100)
    xyArrayTemp2 =sample(xyArray2,100)
    print(xyArray)
    print(xyArray2)
    superxyArray = xyArrayTemp + xyArrayTemp2
    superSuperSample = sample(superxyArray, 100)
    for j in range(100) :
        randomPoint = randint(0,99)
        print(randomPoint)
        print(len(superSuperSample))
        x = superSuperSample[randomPoint][0]
        y = superSuperSample[randomPoint][1]
        newVal = extractedBlue[x][y] 
        if newVal > 0 : #[2] == 0 and newVal[1] != 0 and  newVal[0] != 0:
            tempScore[0] += 1
        if dst[x][y] > 0:
            tempScore[1] += 1
    scoreList.append(tempScore)

df = pd.DataFrame(scoreList)
writer = pd.ExcelWriter('ttestvalues3.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='welcome', index=False)
writer.save()

print(tempScore)