import cv2
import numpy as np
import utils
import sys

def arc_len(p):
    return p.arc_len

class EllipseAssembler:
    def __init__(self, calibrator):
        self.__calibrator = calibrator
        self.__init_references()

    def __init_references(self):
        min_radius = self.__calibrator.reference_ellipse.measurements().fitted_ellipse.axes[0]
        self.ref_ellipse_poly = self.__calibrator.reference_ellipse.points()
        self.dist_thr = min_radius / 1.4
        self.squared_dist_thr = self.dist_thr * self.dist_thr
        self.dist_thr_div_sqrt2 = self.dist_thr / 1.4142135623730951


    def assemble_ellipses(self, src_parts:list, dst_ellipses, bgr):
        # src_parts.sort(key=arc_len, reverse=True)
        while any(src_parts):
            part = src_parts.pop(0)
            # print(self.dist_to_fit_ellipse(part))
            # __VIS__(bgr, part, None, True)
            if self.polygon_is_complete_and_close_to_ref_ellipse(part):
                dst_ellipses.append(part)
                continue

            while True:
                part, next_part, next_part_index = self.find_next_ellipse_part(part, src_parts, bgr)
                # print(self.dist_to_fit_ellipse(part))
                # __VIS__(bgr, part, next_part, True)
                if next_part is None:  # если "след." составляющая не найдена
                    # проверяем - является ли накопленный полигон эллипсом
                    if self.__calibrator.is_close_to_ref_ellipse(part):
                        dst_ellipses.append(part)
                    break
                src_parts.pop(next_part_index)

    def find_next_ellipse_part(self, part, src_parts, bgr):
        if self.pts_are_close(part.last_pt, part.first_pt):
            return part, None, None
        ######
        # TODO: vectorize this loop
        #       - compute distance map (distances between all parts)
        #       - find nearest items using distance map
        next_part = (part, None, None)
        next_part_match = sys.float_info.max
        for i, src_part in enumerate(src_parts):
            parts_are_close, agg_part = self.aggregate_if_close(part, src_part)
            if parts_are_close and self.is_close_to_fit_ellipse(agg_part):
                match = self.match_ref_ellipse(agg_part)
                if match < next_part_match and match <= 1.0:
                    next_part = (agg_part, src_part, i)
                    next_part_match = match

        return next_part

    #######################
    def match_ref_ellipse(self, part):
        return cv2.matchShapes(part.points, self.__calibrator.reference_ellipse.points(), cv2.CONTOURS_MATCH_I1, 0)

    def aggregate_if_close(self, poly, poly2):
        if self.pts_are_close(poly.last_pt, poly2.first_pt):
            return True, poly.append(poly2)
        if self.pts_are_close(poly.last_pt, poly2.last_pt):
            return True, poly.append(poly2.flip_points())
        if self.pts_are_close(poly.first_pt, poly2.first_pt):
            return True, poly.flip_points().append(poly2)
        if self.pts_are_close(poly.first_pt, poly2.last_pt):
            return True, poly2.append(poly)
        return False, None

    @staticmethod
    def is_close_to_fit_ellipse(part):
        # todo: may be cv2.matchShapes(agg_part.points, agg_part_ellipse_poly)???
        dist = EllipseAssembler.dist_to_fit_ellipse(part)
        return dist <= 1.2

    @staticmethod
    def dist_to_fit_ellipse(part):
        if part.fit_ellipse is None:
            return sys.float_info.max
        return utils.polygon_polygon_test(part.points, part.fit_ellipse.poly, -1)

    def polygon_is_complete_and_close_to_ref_ellipse(self, poly):
        return self.pts_are_close(poly.first_pt, poly.last_pt) and \
               self.__calibrator.is_close_to_ref_ellipse(poly)

    # def poly_are_close(self, poly1, poly2):
    #     # last_first_are_close = self.pts_are_close(poly1.last_pt, poly2.first_pt)
    #     # if last_first_are_close:
    #     #     return True, True
    #     # return (self.pts_are_close(poly1.last_pt, poly2.last_pt), False)
    #
    #     last_first_are_close = self.pts_are_close(poly1.last_pt, poly2.first_pt) or \
    #                            self.pts_are_close(poly1.first_pt, poly2.last_pt)
    #     if last_first_are_close:
    #         return True, True
    #     return (self.pts_are_close(poly1.first_pt, poly2.first_pt) or self.pts_are_close(poly1.last_pt, poly2.last_pt),
    #             False)

    def pts_are_close(self, pt1, pt2):
        return pts_are_close(pt1, pt2, self.dist_thr, self.squared_dist_thr, self.dist_thr_div_sqrt2)


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
