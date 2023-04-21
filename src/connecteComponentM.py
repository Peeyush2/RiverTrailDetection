import cv2
import numpy as np

def connectedComponentsM(img) :

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
    

    #selecting pixels above 127 threshold
    #making rest of the pixels 0
    ret, binary_map = cv2.threshold(img,127,255,0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    #making a blank canvas to plot the connected components
    result = np.zeros((labels.shape), np.uint8)

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


