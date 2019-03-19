import cv2
import numpy as np
from detection_attempts.att_1.contour import Contour
from ellipse import Ellipse
import utils


class ReferenceEllipse:
    # masked image
    # or image patch
    # mean color
    def __init__(self, contour, calibration_image):
        self.contour = contour
        self.calibration_image = calibration_image
        self.mask, self.row_coords, self.column_coords, self.mean_color = \
            ReferenceEllipse.__mask_coords_stats(contour, calibration_image)

    @staticmethod
    def __mask_coords_stats(contour, image):
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, contour.points(), 255)
        row_coords, column_coords = mask.nonzero()
        mean_color = np.mean(image[row_coords, column_coords], axis=0, dtype=np.float32)
        return mask, row_coords, column_coords, mean_color


class Calibrator:
    @classmethod
    def calibrate(cls, calibration_image, contact_num):
        calibration_image = calibration_image.copy()
        reference_ellipses = Calibrator.detect_reference_ellipses(calibration_image)
        return cls(reference_ellipses)

    def __init__(self, reference_ellipses):
        self.reference_ellipses = reference_ellipses
        self.reference_ellipse = reference_ellipses[0]
        self.max_contour_angle = max(
            [el.contour.measurements().approx_points_angles.max() for el in reference_ellipses])

    def is_close_to_ref_ellipse(self, polygon):
        ref_area = self.reference_ellipse.contour.measurements().fitted_ellipse.area
        area_min, area_max = ref_area * 0.5, ref_area * 1.2

        ref_arc_len = self.reference_ellipse.contour.measurements().arc_len
        # TODO: если контур собран не полностью (что-то его закрывает) - то сущ. ограничения мешают его распознать
        #  Надо бы пропорционально "заполненности" контуров проверять на соответствие - cv2.matchShape???

        arc_len_min, arc_len_max = ref_arc_len * 0.5, ref_arc_len * 1.3

        ar = self.reference_ellipse.contour.measurements().fitted_ellipse.aspect_ratio
        ar_min, ar_max = ar * .8, ar * 1.2

        # cv2.matchShapes(self.reference_ellipse.contour.points(), polygon.points, 1, 0) <= 1.0
        # utils.polygon_polygon_test(polygon.points, self.reference_ellipse.contour.points(), -1) < 4.0
        try:
            return cv2.matchShapes(self.reference_ellipse.contour.points(), polygon.points, 1, 0) <= 1.0 and \
                   polygon.fit_ellipse and area_min <= polygon.fit_ellipse.area <= area_max and \
                   ar_min <= polygon.fit_ellipse.aspect_ratio <= ar_max
        except:
            raise

        return arc_len_min <= polygon.arc_len <= arc_len_max and \
               cv2.matchShapes(self.reference_ellipse.contour.points(), polygon.points, 1, 0) <= 1.0 and \
               polygon.fit_ellipse and area_min <= polygon.fit_ellipse.area <= area_max and \
               ar_min <= polygon.fit_ellipse.aspect_ratio <= ar_max

    @property
    def calibrated(self):
        return len(self.reference_ellipses) > 0

    @staticmethod
    def detect_reference_ellipses(bgr):
        # todo: assemble ellipses similar to strong ellipse
        #       - all ellipses should be of similar color (min distance in color space)
        #       - надо обработать "детект" внутренних эллипсов - по окружающему цвету??
        contours = Contour.find(bgr)
        all_strong_ellipses, not_ellipses = Calibrator.__extract_strong_ellipses(contours)
        if len(all_strong_ellipses) == 0:
            return []
        strong_ellipses = Calibrator.__skip_small_ellipses(all_strong_ellipses)
        ref_ellipses = [ReferenceEllipse(contour, bgr) for contour in strong_ellipses]
        # todo: assemble from parts using collected strong ellipses
        return ref_ellipses

    @staticmethod
    def __extract_strong_ellipses(contours):
        strong = []
        not_ellipses = []
        for c in contours:
            if Ellipse.is_strong_ellipse(c):
                strong.append(c)
            else:
                not_ellipses.append(c)
        return strong, not_ellipses

    @staticmethod
    def __skip_small_ellipses(ellipses):
        # остаются только близкие к наибольшему (по площади)
        max_ellipse = Calibrator.__ellipse_area(max(ellipses, key=Calibrator.__ellipse_area))
        area_threshold = 0.7 * max_ellipse
        return [el for el in ellipses if Calibrator.__ellipse_area(el) >= area_threshold]

    @staticmethod
    def __ellipse_area(e):
        return e.measurements().area
