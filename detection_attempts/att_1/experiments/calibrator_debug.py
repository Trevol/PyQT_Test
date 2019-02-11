import numpy as np
from detection_attempts.att_1.detect_on_video import (get_capture_and_calibration_image_video2,
                                                      get_capture_and_calibration_image_video6, Detector, Calibrator)


def print_info(calibrator):
    for el in calibrator.reference_ellipses:
        angles = el.contour.measurements().approx_points_angles
        print(angles.min(), np.average(angles), angles.max())
        # print(angles)


def main():
    _, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    print_info(calibrator)
    print('---------------------')
    _, calibration_image = get_capture_and_calibration_image_video6()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    print_info(calibrator)


main()
