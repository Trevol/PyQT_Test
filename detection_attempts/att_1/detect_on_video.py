import cv2
import numpy as np
from detection_attempts.VideoCapture import VideoCapture
from detection_attempts.att_1.calibrator import Calibrator
from contour_visualize import draw_contours, draw_polylines
from detection_attempts.att_1.detector import Detector
from detection_attempts.timeit import timeit
import utils


def visualize_calibration(calibrator, calibration_image):
    visualize(calibrator.reference_ellipses, calibration_image, 'calibration')


def visualize(contours, image, winname):
    draw_contours(contours, image, (0, 255, 0), thickness=2, draw_measurements=False)

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, image)
    cv2.waitKey()
    cv2.destroyWindow(winname)


def draw_parts(parts, image):
    for part in parts:
        color = (0, 255, 0)  # utils.random_color()
        cv2.polylines(image, [part.points], False, color, 2)


def draw_ellipses(polygons, image):
    color = (0, 255, 0)
    pts = [p.points for p in polygons]
    cv2.polylines(image, pts, False, color, 2)
    # for poly in polygons:
    #     el = poly.fit_ellipse
    #     cv2.ellipse(image, utils.intt(el.center), utils.intt(el.axes), el.angle, 0, 360, color, thickness=2)


def visualize_contours(contours, image, winname):
    contours = sorted(contours, key=lambda c: c.len(), reverse=True)
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    for part in contours:
        cv2.drawContours(image, [part.points()], -1, utils.random_color(), 2)
        print('part', part.len())
        cv2.imshow(winname, image)
        cv2.waitKey()
    cv2.destroyWindow(winname)


def detect_and_draw(detector, image):
    ellipses, t = timeit(detector.detect, image)
    print(round(t, 4))
    draw_ellipses(ellipses, image)


def detect_and_show(winname, detector, image, frame_pos, wait=True):
    detect_and_draw(detector, image)
    if frame_pos is not None:
        utils.put_frame_pos(image, frame_pos)
    cv2.imshow(winname, image)
    if wait:
        cv2.waitKey()


def get_capture_and_calibration_image_video6():
    source = 'd:/DiskE/Computer_Vision_Task/Video_6.mp4'
    video = VideoCapture(source)
    calibration_image = video.read_at_pos(65)
    return video, calibration_image


def get_capture_and_calibration_image_video2():
    # (720, 1280, 3)
    source = 'd:/DiskE/Computer_Vision_Task/Video_2.mp4'
    video = VideoCapture(source)
    calibration_image = video.read_at_pos(225)
    # calibration_image[480:553, 650:720]  # strong ellipse
    return video, calibration_image[480:720, 0:1280]  # strong ellipse and parts (connected contacts)


def main_2():
    video, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    # visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)

    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    # detect_and_show('detection', detector, video.read_at_pos(442), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(464), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(471), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(833), video.frame_pos() - 1)

    detect_and_show('detection', detector, video.read_at_pos(497), video.frame_pos() - 1)

    # detect_and_show('detection', detector, video.read_at_pos(1511), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(1601), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(1602), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(1603), video.frame_pos() - 1)
    # detect_and_show('detection', detector, video.read_at_pos(1604), video.frame_pos() - 1)
    # 833

    video.release()
    cv2.destroyAllWindows()


def main_2_test():
    def test_region(frame_num, region):
        col, row, d_col, d_row = region
        detect_and_show('detection', detector, video.read_at_pos(frame_num)[row:row + d_row, col:col + d_col], None,
                        True)
        cv2.destroyAllWindows()

    video, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    # visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)

    # test_region(442, (474, 545, 134, 74))
    # test_region(442, (403, 545, 148, 74))
    # test_region(442, (403, 545, 204, 74))
    # test_region(442, (434, 542, 163, 76))
    # test_region(442, (472, 540, 78, 78))
    #
    # test_region(464, (596, 422, 187, 136))
    # test_region(464, (652, 422, 73, 136))
    # test_region(464, (652, 432, 73, 70))

    # test_region(833, (838, 485, 68, 73))
    # test_region(833, (770, 321, 68, 122))
    # test_region(833, (770, 256, 71, 132))

    # test_region(833, (598, 317, 68, 68))
    # test_region(833, (601, 317, 66, 122))
    # test_region(833, (561, 317, 106, 122))

    # test_region(1511, (721, 151, 131, 132))
    # test_region(1511, (723, 151, 127, 77))
    # test_region(1511, (658, 151, 250, 132))

    # test_region(1511, (779, 151, 71, 77))
    # test_region(1511, (780, 157, 66, 67))

    test_region(497, (958, 173, 74, 65))

    # test_region(1601, (308, 428, 70, 70))
    # test_region(1601, (308, 373, 76, 66))
    # test_region(1601, (308, 317, 136, 187))
    # test_region(1601, (318, 160, 76, 66))

    video.release()
    cv2.destroyAllWindows()


def main_2_video():
    video, calibration_image = get_capture_and_calibration_image_video2()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    # video.set_pos(480)
    for frame in video.frames():
        pos = video.frame_pos()
        print(pos)
        detect_and_draw(detector, frame)
        utils.put_frame_pos(frame, pos)
        cv2.imshow('detection', frame)
        cv2.waitKey(1)
    cv2.waitKey()

    video.release()
    cv2.destroyAllWindows()


def main_6():
    video, calibration_image = get_capture_and_calibration_image_video6()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    # visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    detect_and_show('detection', detector, video.read_at_pos(1660), video.frame_pos() - 1)

    video.release()
    cv2.destroyAllWindows()


def main_6_test():
    def test_region(frame_num, region):
        col, row, d_col, d_row = region
        detect_and_show('detection', detector, video.read_at_pos(frame_num)[row:row + d_row, col:col + d_col], None,
                        True)
        cv2.destroyAllWindows()

    video, calibration_image = get_capture_and_calibration_image_video6()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    # visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    # 1201 1218 3001
    test_region(1660, (1406, 444, 128, 120))

    video.release()
    cv2.destroyAllWindows()


def main_6_video():
    video, calibration_image = get_capture_and_calibration_image_video6()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    video.set_pos(0)
    for frame in video.frames():
        pos = video.frame_pos()
        print(pos)
        detect_and_draw(detector, frame)
        utils.put_frame_pos(frame, pos)
        cv2.imshow('detection', frame)
        cv2.waitKey(1)
    cv2.waitKey()

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    np.seterr(all='raise')
    main_6_video()
