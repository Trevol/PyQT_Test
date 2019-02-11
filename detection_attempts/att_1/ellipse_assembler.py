import cv2
import numpy as np
import utils
import sys


def arc_len(p):
    return p.arc_len


class EllipseAssembler:
    def __init__(self, calibrator):
        self.__calibrator = calibrator
        self.__init_thresholds()

    def __init_thresholds(self):
        min_radius = self.__calibrator.reference_ellipse.contour.measurements().fitted_ellipse.axes[0]
        self.ref_ellipse_poly = self.__calibrator.reference_ellipse.contour.points()
        self.dist_thr = min_radius / 1.3
        self.squared_dist_thr = self.dist_thr * self.dist_thr
        self.dist_thr_div_sqrt2 = self.dist_thr / 1.4142135623730951

    def assemble_ellipses(self, parts, bgr):
        ellipses = self.__assemble_ellipses_from_parts(parts, bgr)
        ellipses = self.remove_duplicates(ellipses)
        return ellipses

    def remove_duplicates(self, ellipses):
        if len(ellipses) == 0:
            return ellipses
        unique_ellipses = [ellipses[0]]
        for ell in ellipses[1:]:
            # if in result exist item close to ell
            if not self.exist_duplicate(ell, unique_ellipses):
                unique_ellipses.append(ell)
        return unique_ellipses

    def exist_duplicate(self, test_ellipse, unique_ellipses):
        return any([uq for uq in unique_ellipses if self.are_duplicates(test_ellipse, uq)])

    def are_duplicates(self, ellipse1, ellipse2):
        center1 = np.array(ellipse1.fit_ellipse.center)
        center2 = np.array(ellipse2.fit_ellipse.center)
        return self.__pts_are_close(center1, center2)

    def __assemble_ellipses_from_parts(self, src_parts, bgr):
        dst_ellipses = []

        # DEBUG
        # src_parts = [p for p in src_parts if p.len in (17, 19)]

        src_parts.sort(key=lambda p: p.len, reverse=True)
        while any(src_parts):
            part = src_parts.pop(0)
            if self.__polygon_is_complete_and_close_to_ref_ellipse(part):
                dst_ellipses.append(part)
                continue

            while True:
                # DEBUG.VIS_assemble_ellipses_from_parts(bgr, part, None, dst_ellipses)
                part, next_part, next_part_index = self.__find_next_ellipse_part(part, src_parts, bgr)
                # DEBUG.VIS_assemble_ellipses_from_parts(bgr, part, next_part, dst_ellipses)
                if next_part is None:  # если "след." составляющая не найдена
                    # проверяем - является ли накопленный полигон эллипсом
                    if self.__polygon_is_complete_and_close_to_ref_ellipse(part):
                        dst_ellipses.append(part)
                    break
                src_parts.pop(next_part_index)
        return dst_ellipses

    def __find_next_ellipse_part(self, part, src_parts, bgr):
        if self.__pts_are_close(part.last_pt, part.first_pt):
            return part, None, None
        ######
        # TODO: vectorize this loop
        #       - compute distance map (distances between all parts)
        #       - find nearest items using distance map
        next_part = (part, None, None)
        next_part_match = sys.float_info.max
        for i, src_part in enumerate(src_parts):
            parts_are_close, agg_part = self.__aggregate_if_close(part, src_part)
            if parts_are_close and self.__is_close_to_fit_ellipse(agg_part):
                match = self.__match_ref_ellipse(agg_part)
                if match < next_part_match and match <= 1.0:
                    next_part = (agg_part, src_part, i)
                    next_part_match = match

        return next_part

    #######################
    def __match_ref_ellipse(self, part):
        ref_ellipse_contour = self.__calibrator.reference_ellipse.contour.points()
        return cv2.matchShapes(part.points, ref_ellipse_contour, cv2.CONTOURS_MATCH_I1, 0)

    def __aggregate_if_close(self, poly, poly2):
        if self.__pts_are_close(poly.last_pt, poly2.first_pt):
            return True, poly.append(poly2)
        if self.__pts_are_close(poly.last_pt, poly2.last_pt):
            return True, poly.append(poly2.flip_points())
        if self.__pts_are_close(poly.first_pt, poly2.first_pt):
            return True, poly.flip_points().append(poly2)
        if self.__pts_are_close(poly.first_pt, poly2.last_pt):
            return True, poly2.append(poly)
        return False, None

    @staticmethod
    def __is_close_to_fit_ellipse(part):
        # todo: may be cv2.matchShapes(agg_part.points, agg_part_ellipse_poly)???
        dist = EllipseAssembler.__dist_to_fit_ellipse(part)
        return dist <= 1.2  # 1.35 # 1.2

    @staticmethod
    def __dist_to_fit_ellipse(part):
        if part.fit_ellipse is None or not part.fit_ellipse.valid:
            return sys.float_info.max
        return utils.polygon_polygon_test(part.points, part.fit_ellipse.poly, -1)

    def __polygon_is_complete_and_close_to_ref_ellipse(self, poly):
        return self.__poly_is_closed(poly) and self.__calibrator.is_close_to_ref_ellipse(poly)

    def __poly_is_closed(self, poly):
        tails_are_close = self.__pts_are_close(poly.first_pt, poly.last_pt)
        return tails_are_close or self.__poly_tails_are_overlap(poly)

    def __poly_tails_are_overlap(self, poly):
        # return False
        first_pt = poly.first_pt
        poly_without_first_pt = poly.points[1:]
        dist = abs(cv2.pointPolygonTest(poly_without_first_pt, tuple(first_pt), True))
        dist_thr = 2
        if dist < dist_thr:
            return True

        last_pt = poly.last_pt
        poly_without_last_pt = poly.points[0:len(poly.points) - 1]
        dist = abs(cv2.pointPolygonTest(poly_without_last_pt, tuple(last_pt), True))
        if dist < dist_thr:
            return True

        return False

    def __pts_are_close(self, pt1, pt2):
        return pts_are_close(pt1, pt2, self.dist_thr, self.squared_dist_thr, self.dist_thr_div_sqrt2)


from detection_attempts.att_1.cv_named_window import CvNamedWindow


class DEBUG:
    @staticmethod
    def VIS_assemble_ellipses_from_parts(bgr, part, next_part, ellipses):
        if bgr.shape[0] > 300 or bgr.shape[1] > 300:
            return
        green = (0, 255, 0)
        red = (0, 0, 255)
        blue = (255, 0, 0)

        im = bgr.copy()

        cv2.polylines(im, [part.points], False, green, 2)
        x, y = part.first_pt
        im[y - 3:y + 3, x - 3:x + 3] = [0, 255, 0]
        x, y = part.last_pt
        im[y - 3:y + 3, x - 3:x + 3] = [0, 127, 0]

        if next_part:
            cv2.polylines(im, [next_part.points], False, red, 2)
            x, y = next_part.first_pt
            im[y - 3:y + 3, x - 3:x + 3] = [0, 0, 255]
            x, y = next_part.last_pt
            im[y - 3:y + 3, x - 3:x + 3] = [0, 0, 127]

        for poly in ellipses:
            cv2.polylines(im, [poly.points], False, blue, 2)
        wnd = CvNamedWindow('DEBUG')
        wnd.imshow(im)
        cv2.waitKey()
        wnd.destroy()


def delete_items_by_indexes(items, indexes_in_desc_order):
    for i in indexes_in_desc_order:
        items.pop(i)


def pts_are_close(pt1, pt2, dist_threshold, squared_dist_threshold, dist_threshold_div_sqrt2):
    vec = pt1 - pt2
    delta_x = abs(vec[0])
    delta_y = abs(vec[1])
    if delta_x > dist_threshold or delta_y > dist_threshold:
        return False  # явно далеко - если хотя бы одна из составляющих больше порога
    # если обе координаты <= dist_threshold/sqrt(2) - то можно НЕ заниматься суммированием квадратов
    return delta_x <= dist_threshold_div_sqrt2 and delta_y <= dist_threshold_div_sqrt2 or \
           (delta_x * delta_x + delta_y * delta_y) <= squared_dist_threshold


def squared_distance_pt(pt1, pt2):
    vec = pt1 - pt2
    delta_x = vec[0]
    delta_y = vec[1]
    return delta_x * delta_x + delta_y * delta_y


def distance_pt(pt1, pt2):
    return np.sqrt(squared_distance_pt(pt1, pt2))


def __VIS__(bgr, part, src_part, track_prev=True):
    im_copy = bgr.copy()

    if track_prev:
        cv2.polylines(bgr, [part.points], False, (255, 0, 0), 2)
        if src_part:
            cv2.polylines(bgr, [src_part.points], False, (255, 0, 0), 2)

    cv2.polylines(im_copy, [part.points], False, (0, 255, 0), 2)
    cv2.circle(im_copy, tuple(part.last_pt), 3, (0, 255, 0), -1)

    if src_part:
        cv2.polylines(im_copy, [src_part.points], False, (0, 0, 255), 2)
        cv2.circle(im_copy, tuple(src_part.last_pt), 3, (0, 0, 255), -1)
        cv2.circle(im_copy, tuple(src_part.first_pt), 3, (0, 0, 0), -1)

    cv2.imshow('debug', im_copy)
    cv2.waitKey()
