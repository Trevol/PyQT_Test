import cv2
import numpy as np
from detection_attempts.att_1.detect_on_video import (get_capture_and_calibration_image_video2,
                                                      get_capture_and_calibration_image_video6, Detector, Calibrator)
from cv_named_window import CvNamedWindow


def get_detector(calibration_image):
    calibrator = Calibrator.calibrate(calibration_image, 2)
    if not calibrator.calibrated:
        print('System was not calibrated.')
        return
    return Detector(calibrator)


class RegionSelection:
    def __init__(self, start, end=None):
        self.start = start
        self.end = end

    def set_end(self, end):
        self.end = end
        self.__normalize_region()
        self.__print_selection()
        return self

    def __print_selection(self):
        (x_start, y_start), (x_end, y_end) = self.start, self.end
        print(f'Region Selection: image_slice ({y_start}:{y_end}, {x_start}:{x_end}). Points: {self.start}-{self.end}')

    def __normalize_region(self):
        (x_start, y_start), (x_end, y_end) = self.start, self.end
        self.start = (min(x_start, x_end), min(y_start, y_end))
        self.end = (max(x_start, x_end), max(y_start, y_end))

    def is_complete(self):
        return self.end is not None

    def point_test(self, pt):
        if self.end is None:
            return -1  # always outside of region
        (x_start, y_start), (x_end, y_end) = self.start, self.end
        x, y = pt
        if x < x_start or x > x_end or y < y_start or y > y_end:
            return -1  # outside region
        return 1

    def get_image_region(self, image):
        (x_start, y_start), (x_end, y_end) = self.start, self.end
        return image[y_start:y_end, x_start:x_end]

    def translate_image_pt(self, pt):
        if self.point_test(pt) < 0:  # outside of region
            return pt
        x, y = pt
        x_start, y_start = self.start
        return x - x_start, y - y_start

    def draw(self, image):
        white = (255, 255, 255)
        if self.end is None:
            x, y = self.start
            image[y - 1:y + 1, x - 1:x + 1] = white
        else:
            cv2.rectangle(image, self.start, self.end, white, 1)


class DetectionDebugger:
    def __init__(self, detector, image, region_selection=None):
        self.detector = detector
        self.base_image = image
        self.buffer_image = np.empty_like(self.base_image)
        self.region_selection = None
        self.set_region_selection(region_selection)
        self.ellipses = self.polygons = self.selected_ellipses = self.selected_polygons = []
        self.wnd = CvNamedWindow('test', mouse_callback=self.__mc)

    def set_region_selection(self, region_selection):
        if region_selection is None:
            return
        start, end = region_selection
        self.region_selection = RegionSelection(start=start).set_end(end)

    def __mc(self, evt, x, y, flags, param):
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        self.__handle_mouse_click((x, y), ctrl_key=flags & cv2.EVENT_FLAG_CTRLKEY)

    def __handle_mouse_click(self, pt, ctrl_key):
        if ctrl_key:
            self.make_region_selection(pt)
        elif self.region_selection and self.region_selection.point_test(pt) < 0:
            self.__clear_region_selection()
        else:
            pt = self.translate_image_pt_to_region_pt(pt)
            self.select_nearest(pt)

        self.__redraw()

    def __handle_key(self, key):
        if key == 27:
            return False
        should_redraw = False

        if key == ord('a'):
            self.__clear_region_selection()
            should_redraw = True
        elif key == ord('r'):
            self.__repeat_detect()
            should_redraw = True

        if should_redraw:
            self.__redraw()
        return True

    def translate_image_pt_to_region_pt(self, pt):
        if self.region_selection:
            return self.region_selection.translate_image_pt(pt)
        return pt

    def select_nearest(self, pt):
        nearest_polygons = self.find_nearest(pt, self.polygons)
        self.selected_polygons = nearest_polygons
        for p in nearest_polygons:
            print(f'POLY: {p.len}')
        nearest_ellipses = self.find_nearest(pt, self.ellipses)
        self.selected_ellipses = nearest_ellipses

    @staticmethod
    def find_nearest(pt, polygons):
        if len(polygons) == 0:
            return []
        polygon_distances = [(p, p.distance_from_point(pt)) for p in polygons]
        min_dist_polygon = min(polygon_distances, key=lambda item: item[1])[0]
        return [min_dist_polygon]

    def make_region_selection(self, pt):
        self.ellipses = self.polygons = self.selected_ellipses = self.selected_polygons = []
        if self.region_selection is None:
            self.region_selection = RegionSelection(start=pt)
        else:
            self.region_selection.set_end(pt)
            self.__detect()

    def __clear_region_selection(self):
        self.region_selection = None
        self.ellipses = self.polygons = self.selected_ellipses = self.selected_polygons = []
        self.__detect()

    def __repeat_detect(self):
        # self.ellipses = self.polygons = self.selected_ellipses = self.selected_polygons = []
        self.__detect()

    def __redraw(self):
        np.copyto(dst=self.buffer_image, src=self.base_image)
        if self.region_selection and self.region_selection.is_complete():
            region_buffer = self.region_selection.get_image_region(self.buffer_image)
        else:
            region_buffer = self.buffer_image

        # draw selection to region_buffer
        self.__draw_poly_for_presentation(self.selected_ellipses, region_buffer, (0, 255, 0), 2)
        self.__draw_poly_asis(self.selected_polygons, region_buffer, (0, 0, 255), 2)

        if self.region_selection:
            self.region_selection.draw(self.buffer_image)
        self.wnd.imshow(self.buffer_image)

    @staticmethod
    def __draw_poly_asis(polygons, image, color=(0, 255, 0), thickness=2):
        pts = [p.points for p in polygons]
        cv2.polylines(image, pts, False, color, thickness)

    @staticmethod
    def __draw_poly_for_presentation(polygons, image, color=(0, 255, 0), thickness=2):
        for p in polygons:
            p.draw_for_presentation(image, color, thickness)

    def __detect(self):
        if self.region_selection:
            image = self.region_selection.get_image_region(self.base_image).copy()
        else:
            image = self.base_image.copy()
        self.ellipses, self.polygons = self.detector.detect(image)
        print(f'__detect. ellipses: {len(self.ellipses)}. polygons: {len(self.polygons)}')
        self.selected_ellipses = list(self.ellipses)

    def start(self):
        self.__detect()
        self.__redraw()

        while self.__handle_key(cv2.waitKey()):
            pass


def main():
    video, calibration_image = get_capture_and_calibration_image_video2()
    detector = get_detector(calibration_image)
    # frame = video.read_at_pos(619)
    frame = video.read_at_pos(636)
    DetectionDebugger(detector, frame).start()


def main():
    video, calibration_image = get_capture_and_calibration_image_video6()
    detector = get_detector(calibration_image)
    frame = video.read_at_pos(1001)
    # region_selection = ((470, 757), (589, 888))
    # region_selection = ((470, 757), (571, 872))
    region_selection = ((520, 828), (580, 880))
    # ()-()
    DetectionDebugger(detector, frame, region_selection=region_selection).start()


if __name__ == '__main__':
    main()
