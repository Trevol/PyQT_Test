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


def draw_ellipses(ellipses, image):
    color = (0, 255, 0)
    for el in ellipses:
        cv2.ellipse(image, utils.intt(el.center), utils.intt(el.axes), el.angle, 0, 360, color, thickness=2)


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


def detect_and_show(winname, detector, image, frame_pos):
    detect_and_draw(detector, image)
    utils.put_frame_pos(image, frame_pos)
    cv2.imshow(winname, image)
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
    detect_and_show('detection', detector, video.read_at_pos(442), video.frame_pos() - 1)
    detect_and_show('detection', detector, video.read_at_pos(464), video.frame_pos() - 1)
    detect_and_show('detection', detector, video.read_at_pos(471), video.frame_pos() - 1)

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


def main_video():
    video, calibration_image = get_capture_and_calibration_image_video6()
    calibrator = Calibrator.calibrate(calibration_image, 2)
    # visualize_calibration(calibrator, calibration_image)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return

    detector = Detector(calibrator)
    cv2.namedWindow('detection', cv2.WINDOW_NORMAL)
    video.set_pos(1600)
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
    main_video()
