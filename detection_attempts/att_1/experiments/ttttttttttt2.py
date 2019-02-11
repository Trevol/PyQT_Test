import numpy as np
import cv2
import utils
from skimage.measure import compare_ssim as ssim
from detection_attempts.att_1.detect_on_video import (get_capture_and_calibration_image_video2, Calibrator, Detector,
                                                      CvNamedWindow, detect_and_show)

"""
- fast mean ellipse color
    - all frame -> label for each ellipse
    - bounding rect -> mask -> mean 
    - n points in m level ellipses
    - n random points??? 
- ssim (structured similarity index)
- key points matching???    
"""


def get_detector():
    video, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)

    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    return detector

    wnd = CvNamedWindow('detection', cv2.WINDOW_NORMAL)
    image = video.read_at_pos(820)
    cv2.imshow('111', image)
    detect_and_show(wnd, detector, image, video.frame_pos() - 1)


def calc_ssi(im1, im2):
    ind1 = ssim(im1, im2, multichannel=True, data_range=im2.max() - im2.min())
    ind2 = ssim(im1, im2, multichannel=True)
    return ind1, ind2


def test_ssim():
    video, calibration_image = get_capture_and_calibration_image_video2()
    frame = video.read_at_pos(286)

    true_ellipse_image1 = frame[551: 551 + 67, 418: 418 + 69]
    true_ellipse_image2 = video.read_at_pos(820)[269: 269 + 67, 659: 659 + 69]
    false_ellipse_image = frame[501: 501 + 67, 745: 745 + 69]
    # false_ellipse_image = cv2.resize(false_ellipse_image, (true_ellipse_image.shape[1], true_ellipse_image.shape[0]))

    detector = get_detector()

    # ind1, ind2 = calc_ssi(true_ellipse_image1, false_ellipse_image)
    # print('ssind', ind1, ind2)
    # ind1, ind2 = calc_ssi(true_ellipse_image1, true_ellipse_image2)
    # print('ssind', ind1, ind2)
    # ind1, ind2 = calc_ssi(true_ellipse_image2, false_ellipse_image)
    # print('ssind', ind1, ind2)
    # print('-------------------')
    # ind1, ind2 = calc_ssi(true_ellipse_image1, true_ellipse_image1)
    # print('ssind', ind1, ind2)
    # ind1, ind2 = calc_ssi(true_ellipse_image2, true_ellipse_image2)
    # print('ssind', ind1, ind2)
    # ind1, ind2 = calc_ssi(false_ellipse_image, false_ellipse_image)
    # print('ssind', ind1, ind2)

    i = 1000
    print(utils.timeit(lambda: calc_ssi(true_ellipse_image1, false_ellipse_image), i))
    print(utils.timeit(lambda: calc_ssi(true_ellipse_image1, true_ellipse_image2), i))
    print(utils.timeit(lambda: calc_ssi(true_ellipse_image2, false_ellipse_image), i))

    return

    true_ellipse_wnd1 = CvNamedWindow('true_ellipse1', cv2.WINDOW_NORMAL)
    detect_and_show(true_ellipse_wnd1, detector, true_ellipse_image1.copy(), None, wait=False)

    false_ellipse_wnd = CvNamedWindow('false_ellipse', cv2.WINDOW_NORMAL)
    detect_and_show(false_ellipse_wnd, detector, false_ellipse_image.copy(), None, wait=True)

    # ssim_const = ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min())
    # print(ssim_const)


def main():
    test_ssim()


def main__():
    pass


main()
