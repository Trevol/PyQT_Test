import numpy as np
import cv2
from cv_named_window import CvNamedWindow, cvWaitKeys

import utils


# def testImage():
#     img = np.zeros((200, 400), np.uint8)
#     cv2.circle(img, (100, 100), 50, 255, -1)
#     # cv2.circle(img, (100, 100), 10, 0, -1)
#     cv2.circle(img, (197, 100), 50, 255, -1)
#
#     return img
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

def visDistTransformResult(dtResult):
    k = 255 / dtResult.max()
    return (dtResult * k).astype(np.uint8)


def main():
    oringWnd, dstWnd, centersWnd = CvNamedWindow.create('orig', {'dstTransform': cv2.WINDOW_NORMAL}, 'centers')

    img = testImage()
    # img = realImage()
    img = fillHoles(img)

    centers = img.copy()
    indexes, (distTransform, dx, dy) = findCentersIndexesSobel(img)
    centers[indexes] = 127

    oringWnd.imshow(img)
    centersWnd.imshow(centers)
    dstWnd.imshow(visDistTransformResult(distTransform))

    def mouse_callback(evt, x, y, flags, _):
        if evt == cv2.EVENT_LBUTTONDOWN:
            print(x, y, distTransform[y, x], dx[y, x], dy[y, x])

    dstWnd.mouse_callback = mouse_callback

    cvWaitKeys()


def findCentersIndexesSobel(img):
    dst = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_5)
    dx = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    dy = cv2.Sobel(dst, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    # cv2.Scharr()
    peakCondition = (dx == 0) & (dy == 0) & (img > 0)
    return np.where(peakCondition), (dst, dx, dy)


def findCentersIndexesLocalMaxima(img):
    pass


def fillHoles(img8u):
    cnt, labels = cv2.connectedComponents(img8u, connectivity=8)
    cv2.floodFill(labels, None, (0, 0), -1)
    dst = img8u.copy()
    # d[np.where(labels > -1)] = 255
    dst[np.where(labels == 0)] = 255
    return dst





def main():
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    import scipy.ndimage as ndi

    img = realImage()
    # img = testImage()
    img = fillHoles(img)

    thresh = img.copy()

    with utils.timeit_context():
        dst = ndi.distance_transform_edt(img)
        localMax = peak_local_max(dst, indices=False, min_distance=1, labels=thresh)
        markers = ndi.label(localMax)[0]
        labels = watershed(-dst, markers, mask=thresh)

    segmImg = (labels * (255 / labels.max())).astype(np.uint8)

    wnd = CvNamedWindow(flags=cv2.WINDOW_NORMAL)
    segmWnd = CvNamedWindow('segm', flags=cv2.WINDOW_NORMAL)

    wnd.imshow(img)
    segmWnd.imshow(segmImg)

    cvWaitKeys()


if __name__ == '__main__':
    main()
