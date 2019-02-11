import numpy as np
import cv2
from detection_attempts.att_1.detect_on_video import get_capture_and_calibration_image_video2, \
    get_capture_and_calibration_image_video6, visualize_calibration
from detection_attempts.att_1.calibrator import Calibrator


def mean():
    # a = np.ma.array([1, 2, 3], mask=[False, False, True])
    a = np.array([
        [[(200, 200, 200), (100, 200, 200)]],
        [[(200, 200, 200), (100, 200, 200)]]
    ])
    print(a.mean(axis=(0, 1, 2)))


def get_image_patch(ref_ellipse, image):
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, ref_ellipse.contour.points(), 255)

    rr, cc = mask.nonzero()
    print(np.mean(image[rr, cc], axis=0))

    mask_ = np.reshape(mask, [*mask.shape, 1])
    image_masked = np.bitwise_and(image, mask_)
    print(np.mean(image_masked[rr, cc], axis=0))
    print('---------------------')



    # gray2 = gray.astype(np.int16)
    # cv2.fillConvexPoly(gray2, ref_ellipse.contour.points(), -255)
    # pt_x, pt_y = ref_ellipse.contour.points()[0][0]
    # print(gray[pt_y, pt_x], gray[0, 0])

    # cv2.imshow('ddd', image_masked)
    #
    # cv2.waitKey()
    # print(image.min(), image.max())


def main():
    video, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    assert len(calibrator.reference_ellipses) >= 2
    for ref_ellipse in calibrator.reference_ellipses:
        get_image_patch(ref_ellipse, calibration_image)

    cv2.imshow('calib', calibration_image)

    def cb(evt, x, y, _, img):
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        print(x, y, img[y, x])
    cv2.setMouseCallback('calib', cb, param=calibration_image)
    cv2.waitKey()
    # visualize_calibration(calibrator, calibration_image)
    #


def main_perf():
    from utils import timeit
    _, image = get_capture_and_calibration_image_video6()

    def gray_cvt():
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gray_extract_channel():
        return cv2.extractChannel(image, 0)

    def gray_slice():
        return np.array(image[:, :, 1])

    def gray_slice2():
        return image[:, :, 1].copy()

    def gray_slice3():
        return np.zeros_like(image[:, :, 1], dtype=np.int8)

    def gray_slice4():
        return np.zeros(image.shape[0:2], dtype=np.int8)

    def gray_slice5():
        mask = np.zeros(image.shape[0:2], dtype=np.int8)
        return np.reshape(mask, [*mask.shape, 1])

    i = 500
    print(timeit(gray_cvt, i))
    print(timeit(gray_extract_channel, i))
    print(timeit(gray_slice, i))
    print(timeit(gray_slice2, i))
    print(timeit(gray_slice3, i))
    print(timeit(gray_slice4, i))
    print(timeit(gray_slice5, i))
    print('-----------------------------')
    print(timeit(gray_cvt, i))
    print(timeit(gray_extract_channel, i))
    print(timeit(gray_slice, i))
    print(timeit(gray_slice2, i))
    print(timeit(gray_slice3, i))
    print(timeit(gray_slice4, i))
    print(timeit(gray_slice5, i))


main()
