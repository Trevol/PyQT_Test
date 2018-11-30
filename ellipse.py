import utils


class Ellipse:
    @staticmethod
    def is_strong_ellipse(contour):
        if contour.len() < 5:
            return False
        fitted_ellipse = contour.measurements().fitted_ellipse
        if not fitted_ellipse or fitted_ellipse.area == 0 or contour.measurements().area == 0:
            return False
        area_diff = abs(contour.measurements().area - fitted_ellipse.area) / contour.measurements().area
        return area_diff < 0.03  # 0.19 for approximated contour, 0.001 for NON-approximated
        # todo: analyze distance between contour and fitted ellipse centers

    @staticmethod
    def is_strong_garbage(contour):
        # to small
        if len(contour.measurements().tails) < 2 and contour.len() < 70:
            return True

        if len(contour.measurements().tails) == 2 and contour.len() < 50:
            return True

        fitted_ellipse = contour.measurements().fitted_ellipse
        if not fitted_ellipse:
            return True
        if fitted_ellipse.aspect_ratio < 0.2:
            return True

        # distance from fitted_ellipse
        dist = utils.polygon_polygon_test(contour.points(), fitted_ellipse.polygon(), -1)
        if dist > 1:
            return True

        return False
