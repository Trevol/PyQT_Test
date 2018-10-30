from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
    image = cv2.imread(frame)
    cv2.imshow('Orig', image.copy())
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 21)
    #cv2.imshow('Shifted', shifted)
    #shifted = image


    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        _, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(cnts))
        # cnts = cnts[-2]
        # c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for cntr in cnts:
            cv2.drawContours(image, [cntr], 0, (0, 255, 0))

    # show the output image
    cv2.imshow("Output", image)

    cv2.waitKey()


main()
