import cv2
import numpy as np
from cv_named_window import CvNamedWindow as Wnd, cvWaitKeys as waitKeys, imshow
import utils
import sklearn.cluster


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
    imread = cv2.imread('204_408_137_152_1552636918.9538898.png', flags=cv2.IMREAD_GRAYSCALE)
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

    def segment2(self, binaryImage):
        # fill holes
        if self._labels is None:
            self._labels = np.empty(binaryImage.shape, np.int32)
        cnt, labels = cv2.connectedComponents(binaryImage, ltype=cv2.CV_32S, labels=self._labels)

        cv2.floodFill(labels, None, (0, 0), cnt)  # fill main background
        mask = np.equal(labels, 0, out=self._mask)  # detect holes
        binaryImage[mask] = 255  # fill holes

        # prepare sure FG:
        distTransform = cv2.distanceTransform(binaryImage, cv2.DIST_L1, 3, dst=self._distTransform, dstType=cv2.CV_8U)

        from skimage.feature import peak_local_max

        _, sureFg = cv2.threshold(distTransform, 0.8 * distTransform.max(), 255, cv2.THRESH_BINARY, dst=self._thresh)

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


def main():
    # imgPeaksAndCenters = testImage()
    binaryImg = realImage()

    distTransformL1 = cv2.distanceTransform(binaryImg, cv2.DIST_L1, 5, dstType=cv2.CV_8U)
    distTransformL2 = cv2.distanceTransform(binaryImg, cv2.DIST_L2, 5, dstType=cv2.CV_8U)

    from skimage.feature import peak_local_max
    localPeaks = peak_local_max(distTransformL2, min_distance=2, indices=False)

    # binaryWnd, dstL1Wnd, dstL2Wnd, peaksWnd = Wnd.create('binaryImg', 'dstTransform_L1', 'dstTransform_L2', 'peaks')
    # binaryWnd.imshow(binaryImg)
    # dstL1Wnd.imshow(distTransformL1 / distTransformL1.max())
    # dstL2Wnd.imshow(distTransformL2 / distTransformL2.max())
    imshow(binaryImg=binaryImg, dstTransform_L1=distTransformL1 / distTransformL1.max(),
           dstTransform_L2=distTransformL2 / distTransformL2.max(), peaks=localPeaks / 1)

    localPeaksIndices = peak_local_max(distTransformL2, min_distance=4, indices=True)

    ms = sklearn.cluster.MeanShift()
    ms.fit(localPeaksIndices)
    print(np.unique(ms.labels_), ms.cluster_centers_)

    imgPeaksAndCenters = localPeaks / 1
    for center in ms.cluster_centers_:
        r, c = center
        r, c = int(round(r)), int(round(c))
        imgPeaksAndCenters[r - 1:r + 1, c - 1:c + 1] = 1.

    print('----------------------')

    # ms = sklearn.cluster.MeanShift()
    # indices = np.where(binaryImg == 255)
    # indices = np.dstack(indices).reshape([-1, 2])
    # ms.fit(indices)
    #
    # print(np.unique(ms.labels_), ms.cluster_centers_)
    # binaryImgWithClusters = binaryImg.copy()
    # for center in ms.cluster_centers_:
    #     r, c = center
    #     r, c = int(round(r)), int(round(c))
    #     binaryImgWithClusters[r - 1:r + 1, c - 1:c + 1] = 127
    #
    # imshow(peaks2=imgPeaksAndCenters, binaryImgWithClusters=binaryImgWithClusters)

    clusterContours(binaryImg)

    waitKeys()


def clusterContours(binaryImage):
    _, contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.empty([0, 2], dtype=np.int32)
    for contour in contours:
        contour = contour.reshape(-1, 2)
        pts = np.concatenate([pts, contour])

    ms = sklearn.cluster.MeanShift()
    ms.fit(pts)
    print('contour clustering', ms.labels_, ms.cluster_centers_)
    img = np.zeros_like(binaryImage)
    cv2.drawContours(img, contours, -1, 255, 1)

    for center in ms.cluster_centers_:
        r, c = center
        r, c = int(round(r)), int(round(c))
        img[r - 1:r + 1, c - 1:c + 1] = 127

    imshow(contours=img)


if __name__ == '__main__':
    main()
