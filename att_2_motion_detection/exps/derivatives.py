import numpy as np
import cv2
from cv_named_window import CvNamedWindow, cvWaitKeys


def testImage():
    img = np.zeros((200, 400), np.uint8)
    cv2.circle(img, (100, 100), 50, 255, -1)
    # cv2.circle(img, (100, 100), 10, 0, -1)
    cv2.circle(img, (197, 100), 50, 255, -1)

    return img


def main():
    img = testImage()
    origWnd, dstWnd, centersWnd = CvNamedWindow.create('orig', 'dstTransform', 'centers')
    dst = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_5)

    dx = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    dy = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
    grad = cv2.addWeighted(cv2.convertScaleAbs(dx), 1, cv2.convertScaleAbs(dy), 1, 0)
    grad = cv2.convertScaleAbs(grad)

    origWnd.imshow(img)
    centersWnd.imshow(grad)
    condition = (grad == 0) & (dst > 0)
    print(np.where(condition))

    cvWaitKeys()

if __name__ == '__main__':
    main()
