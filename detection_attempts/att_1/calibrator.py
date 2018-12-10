import cv2
from contour import Contour
from ellipse import Ellipse
import utils


class Calibrator:
    @classmethod
    def calibrate(cls, calibration_image, contact_num):
        reference_ellipses = Calibrator.detect_reference_ellipses(calibration_image.copy())
        return cls(reference_ellipses)

    def __init__(self, reference_ellipses):
        self.reference_ellipses = reference_ellipses
        self.reference_ellipse = reference_ellipses[0]

    def is_close_to_ref_ellipse(self, polygon):
        ref_area = self.reference_ellipse.measurements().fitted_ellipse.area
        area_min, area_max = ref_area * 0.5, ref_area * 1.2

        ref_arc_len = self.reference_ellipse.measurements().arc_len
        # TODO: если контур собран не полностью (что-то его закрывает) - то сущ. ограничения мешают его распознать
        #  Надо бы пропорционально "заполненности" контуров проверять на соответствие - cv2.matchShape???

        arc_len_min, arc_len_max = ref_arc_len * 0.5, ref_arc_len * 1.3

        ar = self.reference_ellipse.measurements().fitted_ellipse.aspect_ratio
        ar_min, ar_max = ar * .8, ar * 1.2

        # cv2.matchShapes(self.reference_ellipse.points(), polygon.points, 1, 0) <= 1.0
        # utils.polygon_polygon_test(polygon.points, self.reference_ellipse.points(), -1) < 4.0
        return arc_len_min <= polygon.arc_len <= arc_len_max and \
               cv2.matchShapes(self.reference_ellipse.points(), polygon.points, 1, 0) <= 1.0 and \
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
        strong_ellipses, not_ellipses = Calibrator.extract_strong_ellipses(contours)
        if len(strong_ellipses) == 0:
            return []
        strong_ellipses = Calibrator.skip_small_ellipses(strong_ellipses)
        # todo: assemble from parts using collected strong ellipses
        return strong_ellipses


    @staticmethod
    def extract_strong_ellipses(contours):
        strong = []
        not_ellipses = []
        for c in contours:
            if Ellipse.is_strong_ellipse(c):
                strong.append(c)
            else:
                not_ellipses.append(c)
        return strong, not_ellipses


    @staticmethod
    def skip_small_ellipses(ellipses):
        # остаются только близкие к наибольшему (по площади)
        max_ellipse = Calibrator.ellipse_area(max(ellipses, key=Calibrator.ellipse_area))
        area_threshold = 0.7 * max_ellipse
        return [el for el in ellipses if Calibrator.ellipse_area(el) >= area_threshold]


    @staticmethod
    def ellipse_area(e):
        return e.measurements().area
