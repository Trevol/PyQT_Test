import numpy as np
import cv2
from cv_named_window import CvNamedWindow, cvWaitKeys


def testImage():
    # img = np.zeros((200, 200), np.uint8)
    # cv2.circle(img, (100, 100), 50, 255, -1)
    # cv2.circle(img, (100, 100), 10, 0, -1)
    #
    # cv2.circle(img, (188, 188), 10, 2, -1)
    # cv2.circle(img, (0, 0), 50, 100, -1)
    # return img

    img = np.zeros((200, 400), np.uint8)
    cv2.circle(img, (100, 100), 50, 255, -1)
    cv2.circle(img, (100, 100), 10, 0, -1)
    cv2.circle(img, (197, 100), 50, 255, -1)
    return img

def realImage():
    imread = cv2.imread('230_47_121_35_1552309842.126161.png', flags=cv2.IMREAD_GRAYSCALE)
    return cv2.threshold(imread, 1, 255, cv2.THRESH_BINARY)[1]

def main():
    # img = testImage()
    img = realImage()
    n = 1
    sz = n * 2 + 1  # 3, 5 ...
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    iterations = 1
    mode = None  # erode dilate
    wnd = CvNamedWindow()

    def images():
        while True:
            if mode == 'dilate':
                cv2.dilate(img, kernel, dst=img, iterations=iterations)
            elif mode == 'erode':
                cv2.erode(img, kernel, dst=img, iterations=iterations)

            yield img

    for im in images():
        wnd.imshow(im)
        key = cvWaitKeys(27, '1', '2')
        if key == 27:
            break
        elif key == ord('1'):
            mode = 'erode'
        elif key == ord('2'):
            mode = 'dilate'


def main_():
    img = testImage()
    cv2.imshow('', img)
    ret, labels = cv2.connectedComponents(img)
    print(ret, np.unique(labels), labels[100, 100])
    # 1oids)

    cvWaitKeys()


if __name__ == '__main__':
    main()
