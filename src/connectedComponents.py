import cv2
import numpy as np
from dilation import runDilationFunction
from matplotlib import pyplot

def connectedComponents(img) :

    # runDilationFunction(img) 
    erosion_size = 4
    
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    cv2.imshow("morph close element", element)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE ,element)

    cv2.imshow("morph close 2", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pyplot.imshow(element)
    pyplot.show()  

    erosion_size = 4

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    img = cv2.dilate(img, element)

    cv2.imshow("dilated 2", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pyplot.imshow(element)
    pyplot.show()  
    
    cv2.imshow("image with morph open",img)

    ret, binary_map = cv2.threshold(img,127,255,0)

    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

    #get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:,cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    print(nlabels, len(areas))
    for i in range(0, nlabels - 1):
        if areas[i] > 500 :
            results2 = np.zeros((labels.shape), np.uint8)
            results2[labels == i + 1] = 255
            # cv2.imshow(str(i),results2)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # cv2.imwrite("label_"+str(i)+".png",results2)
            # print( i, areas[i] )
        if areas[i] >= 10000:   #keep
            result[labels == i + 1] = 255
    

    #cv2.imshow("Binary", binary_map)
    cv2.imshow("conected components", result)
    # return result
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dst = cv2.Canny(result,100, 200, apertureSize = 5 )
    cv2.imshow("conected canny", dst)
    # return result
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dilatation_size = 5

    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (-1, -1))

    dilatation_dst = cv2.erode(result ,element)

    dst = cv2.Canny(dilatation_dst,100, 200, apertureSize = 5 )
    cv2.imshow("final result", dilatation_dst)
    # return result
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow("eroded", dilatation_dst)
    # # return result
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dilatation_dst


