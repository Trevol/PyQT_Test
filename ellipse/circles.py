#!/usr/bin/python

'''
This example illustrates how to use cv.HoughCircles() function.

Usage:
    houghcircles.py [<image_name>]
    image argument defaults to ../data/board.jpg
'''

# Python 2/3 compatibility
# from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

def drawEllipse(im):
    '''
    Parameters
        img	Image.
        center	Center of the ellipse.
        axes	Half of the size of the ellipse main axes.
        angle	Ellipse rotation angle in degrees.
        startAngle	Starting angle of the elliptic arc in degrees.
        endAngle	Ending angle of the elliptic arc in degrees.
        color	Ellipse color.
        thickness	Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that a filled ellipse sector is to be drawn.
        lineType	Type of the ellipse boundary. See the line description.
        shift	Number of fractional bits in the coordinates of the center and values of axes.
    '''
    cv.ellipse(im, (100, 100), (20, 20), 0, 0, 360, (250, 0, 0), -1, 8, 0)
    # ellipse( img, Point(dx+150, dy+100), Size(100,70), 0, 0, 360, white, -1, 8, 0 );

if __name__ == '__main__':
    print(__doc__)

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "images/contacts/3contacts.jpg"

    src = cv.imread(fn, 1)
    # src = (np.ones((201, 201, 3), dtype=np.uint8) * 255).astype(np.uint8)
    # drawEllipse(src)

    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    cimg = src.copy() # numpy function

    #circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    #circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20)

    if circles is not None and len(circles) > 0: # Check if circles have been found and only then iterate over these and add them to the image
        a, b, c = circles.shape
        for i in range(b):
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv.LINE_AA)  # draw center of circle

        cv.imshow("detected circles", cimg)

    cv.imshow("source", src)
    cv.waitKey(0)
