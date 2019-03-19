import cv2
import numpy as np
import utils
from cv_named_window import CvNamedWindow as Wnd


def testImage():
    img = np.zeros((200, 400), np.uint8)
    cv2.ellipse(img, (100, 100), (50, 40), 0, 0, 360, 255, -1)
    # cv2.circle(img, (100, 100), 10, 0, -1)
    # cv2.circle(img, (197, 100), 50, 255, -1)
    cv2.ellipse(img, (197, 100), (50, 40), 0, 0, 360, 255, -1)

    return img


def realImage():
    imread = cv2.imread('230_47_121_35_1552309842.126161.png', flags=cv2.IMREAD_GRAYSCALE)
    return cv2.threshold(imread, 1, 255, cv2.THRESH_BINARY)[1]


def main():
    resultWnd, markers1Wnd, markers2Wnd = Wnd.create(result=cv2.WINDOW_NORMAL, markers1=cv2.WINDOW_NORMAL, markers2=cv2.WINDOW_NORMAL)

    # img = cv2.imread('water_coins.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh = realImage()
    # thresh = testImage()

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L1, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers1Wnd.imshow(markers / markers.max())

    # markers = cv2.watershed(img, markers)
    img = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)

    markers2Wnd.imshow(markers / markers.max())

    # [-1  1  2  3  4  5]
    img[markers == 3] = [0, 255, 0]
    img[markers == -1] = [255, 0, 0]

    cv2.imshow('thr', thresh)
    cv2.imshow('op', opening)
    cv2.imshow('sure_bg', sure_bg)
    cv2.imshow('sure_fg', sure_fg)
    resultWnd.imshow(img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
