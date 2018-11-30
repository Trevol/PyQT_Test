from contour import Contour
from ellipse import Ellipse


class Calibrator:
    @classmethod
    def calibrate(cls, calibration_image, contact_num):
        reference_ellipses = Calibrator.detect_reference_ellipses(calibration_image.copy())
        return cls(reference_ellipses)

    def __init__(self, reference_ellipses):
        self.reference_ellipses = reference_ellipses

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
