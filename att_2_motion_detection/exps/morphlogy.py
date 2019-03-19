import cv2
import numpy as np
from cv_named_window import CvNamedWindow as Wnd, cvWaitKeys as waitKeys
import utils


# TODO:
#   - opening
#   - closing
def testImage():
    img = np.zeros((300, 400), np.uint8)
    cv2.ellipse(img, (100, 100), (50, 30), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (100, 100), (20, 10), 0, 0, 360, 0, -1)

    cv2.ellipse(img, (197, 100), (50, 30), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (197, 250), (5, 3), 0, 0, 360, 255, -1)
    return img


def realImage():
    imread = cv2.imread('230_47_121_35_1552309842.126161.png', flags=cv2.IMREAD_GRAYSCALE)
    return cv2.threshold(imread, 1, 255, cv2.THRESH_BINARY)[1]


class Segmenter:
    def __init__(self):
        self._labels = None
        self._mask = None
        self._distTransform = None
        self._thresh = None
        self._sureFg = None
        self._unknown = None
        self._bgr = None

    def segment(self, binaryImage):
        # fill holes
        if self._labels is None:
            self._labels = np.empty(binaryImage.shape, np.int32)
        cnt, labels = cv2.connectedComponents(binaryImage, ltype=cv2.CV_32S, labels=self._labels)

        cv2.floodFill(labels, None, (0, 0), cnt)  # fill main background
        mask = np.equal(labels, 0, out=self._mask)
        binaryImage[mask] = 255

        # prepare sure FG:
        distTransform = cv2.distanceTransform(binaryImage, cv2.DIST_L1, 3, dst=self._distTransform, dstType=cv2.CV_8U)
        _, sureFg = cv2.threshold(distTransform, 0.8 * distTransform.max(), 255, cv2.THRESH_BINARY, dst=self._thresh)
        # if self._sureFg is None:
        #     self._sureFg = np.empty(binaryImage.shape, np.uint8)
        # np.copyto(self._sureFg, sureFg, casting='unsafe')
        # sureFg = self._sureFg

        unknown = cv2.subtract(binaryImage, sureFg, dst=self._unknown)

        cnt, markers = cv2.connectedComponents(sureFg, labels=self._labels)

        cv2.add(markers, 1, dst=markers)  # background should be 1

        mask = np.equal(unknown, 255, out=self._mask)
        markers[mask] = 0  # label unknown regions with 0

        bgrImg = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR, dst=self._bgr)
        markers = cv2.watershed(bgrImg, markers)
        return markers

    def markersToDisplayImage(self, markers):
        disp = np.zeros([*markers.shape[:2], 3], np.uint8)
        for m in np.unique(markers):
            if m in (-1, 1):
                continue
            rr, cc = np.where(markers == m)
            disp[rr, cc] = utils.random_color()

        return disp


def main():
    # img = testImage()
    img = realImage()

    segmenter = Segmenter()

    # iters = 1000
    # images = [img.copy() for _ in range(iters)]
    # with utils.timeit_context():
    #     for i in range(1000):
    #         segmenter.segment2(images[i])
    #
    # images = [img.copy() for _ in range(iters)]
    # with utils.timeit_context():
    #     for i in range(1000):
    #         segmenter.segment(images[i])

    markers = segmenter.segment(img.copy())

    wnd = Wnd('labels')
    wnd.imshow(segmenter.markersToDisplayImage(markers))
    waitKeys()


if __name__ == '__main__':
    main()
