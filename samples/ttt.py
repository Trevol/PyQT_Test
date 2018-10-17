import cv2
import numpy as np
import math


def intt(iterable):
    '''
    returns tuple with items rounded to int
    '''
    return tuple(round(i) for i in iterable)


def box_to_ellipse(center, axes, angle):
    return center, (axes[0] / 2, axes[1] / 2), angle


def main():
    im = np.zeros((200, 200), dtype=np.uint8)

    # draw ellipse
    cv2.ellipse(im, (50, 50), (30, 20), 15.78, startAngle=0, endAngle=220, color=(255, 0, 0), thickness=1)

    edges = (im != 0).astype(np.uint8) * 255
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # centroid, points count
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(len(contours), len(contours[0]), cv2.contourArea(contours[0]))

    imContours = np.zeros_like(edges)
    cv2.drawContours(imContours, contours, 0, 255, thickness=1)
    cv2.circle(imContours, (cX, cY), 1, 255)

    #ellipse (center, axes, angle, area)
    center, axes, angle = box_to_ellipse(*cv2.fitEllipseDirect(contours[0]))
    imFittedEllipse = np.zeros_like(edges)
    cv2.ellipse(imFittedEllipse, intt(center), intt(axes), angle, 0, 360, 255, -1)

    cv2.imshow('im', im)
    cv2.imshow('edges', edges)
    cv2.imshow('contours', imContours)
    cv2.imshow('FittedEllipse', imFittedEllipse)
    cv2.waitKey()


main()
