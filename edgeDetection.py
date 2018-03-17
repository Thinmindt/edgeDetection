#!/usr/bin/env python


import cv2 as cv
import numpy as np
import sys
import os
import time




def makeCannyEdges(grayImages):
    cannyEdges = []
    t0 = time.time()

    for gray in grayImages:
        edge = cv.Canny(gray, 2000, 4000, apertureSize=5)
        cannyEdges.append(edge)

    t1 = time.time()
    print "Canny processing took: " + str(t1-t0) + " seconds"
    return cannyEdges

def makePrewittEdges(grayImages):
    prewittEdges = []
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    t0 = time.time()

    for gray in grayImages:
        img_gaussian = cv.GaussianBlur(gray,(3,3),0)
        prewittX = cv.filter2D(img_gaussian, -1, kernelx)
        prewittY = cv.filter2D(img_gaussian, -1, kernely)

        prewittEdges.append(prewittX + prewittY)

    t1 = time.time()
    print "Prewitt processing took: " + str(t1-t0) + " seconds"
    return prewittEdges

def makeSobelEdges(grayImages):
    sobelEdges = []
    t0 = time.time()

    for gray in grayImages:
        sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        sobelEdges.append(sobelX + sobelY)

    t1 = time.time()
    print "Sobel processing took: " + str(t1-t0) + " seconds"
    return sobelEdges

def makeLaplacianEdges(grayImages):
    laplacianEdges = []
    t0 = time.time()

    for gray in grayImages:
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        laplacianEdges.append(laplacian)

    t1 = time.time()
    print "Laplacian processing took: " + str(t1-t0) + " seconds"
    return laplacianEdges

def getGrays():
    grayImages = []
    for file in os.listdir('/home/greg/Downloads/Photos/'):
        img = cv.imread('/home/greg/Downloads/Photos/' + file)
        height, width = img.shape[:2]
        img = cv.resize(img, (width/4, height/4))

        grayImages.append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

    return grayImages

def showImage(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

grayImages = getGrays()
cannyEdges = makeCannyEdges(grayImages)
prewittEdges = makePrewittEdges(grayImages)
sobelEdges = makeSobelEdges(grayImages) 
laplacianEdges = makeLaplacianEdges(grayImages)

def averageSquaredDifference(listA, listB):
    sum = 0
    for imageA, imageB in map(None, listA, listB):
        sum += squaredDifference(imageA, imageB)
    return sum / len(listA)

def squaredDifference(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

print "Prewitt SSD = " + str(averageSquaredDifference(cannyEdges, prewittEdges))
print "Sobel SSD = " + str(averageSquaredDifference(cannyEdges, sobelEdges))
print "Laplacian SSD = " + str(averageSquaredDifference(cannyEdges, laplacianEdges))