import cv2

from detection_attempts.att_1.contour import Contour
from detection_attempts.att_1.polygon import Polygon
import geometry_utils as geometry
import numpy as np
import utils


class Detector:
    def __init__(self, calibrator):
        if not calibrator.calibrated:
            raise Exception('not calibrator.calibrated')
        self.__calibrator = calibrator
        self.__ref_ellipse = self.__calibrator.reference_ellipses[0]

    def detect(self, bgr):
        contours = Contour.find(bgr)

        # skip short and straight (approx pts < 3) contours
        min_len = self.__ref_ellipse.len() // 3
        contours = [c for c in contours if c.measurements().approx_points.size != 0 and c.len() > min_len]

        all_parts = []
        for c in contours:
            parts = self.split(c)  # TODO: сразу фильтровать??
            all_parts.extend(parts)

        # убираем: слишком длинные и слишком короткие контуры
        arc_len_max = self.__ref_ellipse.measurements().arc_len * 1.2
        arc_len_min = self.__ref_ellipse.measurements().arc_len * 0.12
        all_parts = [p for p in all_parts if arc_len_min < p.arc_len < arc_len_max]

        all_parts = self.__detect_ellipses(all_parts)

        # убираем: не "фитятся" в эллипс
        return all_parts

    def __detect_ellipses(self, parts):
        # 1: part is ellipse: ref_ellipse.measurements
        ellipses = []
        for i, p in enumerate(parts):
            if self.is_close_to_ref_ellipse(p):
                ellipses.append(p.fit_ellipse)
                parts.pop(i)

        # self.__assemble_ellipses(parts, ellipses)

        return ellipses

    def __assemble_ellipses(self, parts, ellipses):
        #
        for i, p_i in enumerate(parts):
            for j in range(i + 1, len(parts)):
                if j >= len(parts):
                    break
                p_j = parts[j]
                p = p_i + p_j
                if self.is_close_to_ref_ellipse(p):
                    ellipses.append(p.fit_ellipse)
                    # parts.remove(p_i)
                    parts.remove(p_j)

    def is_close_to_ref_ellipse(self, polygon):
        ref_area = self.__ref_ellipse.measurements().fitted_ellipse.area
        area_min, area_max = ref_area * 0.8, ref_area * 1.2

        ref_arc_len = self.__ref_ellipse.measurements().arc_len
        arc_len_min, arc_len_max = ref_arc_len * 0.8, ref_arc_len * 1.2

        ar = self.__ref_ellipse.measurements().fitted_ellipse.aspect_ratio
        ar_min, ar_max = ar * .8, ar * 1.2
        return arc_len_min <= polygon.arc_len <= arc_len_max and \
               polygon.fit_ellipse and area_min <= polygon.fit_ellipse.area <= area_max and \
               ar_min <= polygon.fit_ellipse.aspect_ratio <= ar_max

    def split(self, contour):
        # сразу сравнивать с эталонным элипсом???
        pts = contour.measurements().approx_points
        angles, _ = geometry.compute_angles_vectorized(pts)

        # (45. < angles) & (angles <= 160.)
        # mask = angles > 160
        mask = 45 < angles
        indexes = np.where(mask)[0]
        if indexes.size == 0:
            return [Polygon(pts)]

        parts = []

        for i in range(len(indexes) - 1):
            # todo: м.б. формировать список уникальных контуров в этом цикле??
            index = indexes[i]
            next_index = indexes[i + 1]
            if (next_index - index) < 2:  # пропускаем фрагменты из 2 и менее точек
                continue
            part_pts = pts[index: next_index + 1]
            parts.append(Polygon(part_pts))

        part_pts = np.append(pts[indexes[-1]:], pts[:indexes[0] + 1], axis=0)
        if len(part_pts) > 2:  # пропускаем фрагменты из 2 и менее точек
            parts.append(Polygon(part_pts))

        if len(parts) == 0:
            return []

        ############################
        unique_parts = [parts[0]]
        for i in range(1, len(parts)):
            part = parts[i]
            uq_part, uq_part_index = \
                utils.first_or_default(unique_parts, lambda uniq_part: uniq_part.is_equivalent(part, 4))
            if uq_part is None:
                unique_parts.append(part)
            elif part.arc_len > uq_part.arc_len:  # compare sizes and use bigger part
                unique_parts[uq_part_index] = part

        return unique_parts

    @staticmethod
    def __visualize_contours_parts(bgr, contour, parts):
        cv2.namedWindow('debug', cv2.WINDOW_NORMAL)

        # im = bgr.copy()
        im = bgr
        cv2.drawContours(im, [contour.points()], -1, (0, 255, 0), 1)
        print(contour.len(), len(parts))
        cv2.imshow('debug', im)
        cv2.waitKey()

        for p in parts:
            # im = bgr.copy()
            cv2.polylines(im, [p], False, utils.random_color(), 2)
            cv2.imshow('debug', im)
            cv2.waitKey()

        cv2.drawContours(im, [contour.points()], -1, (255, 255, 255), 1)
        for p in parts:
            cv2.polylines(im, [p], False, (255, 255, 255), 2)

        while cv2.waitKey() != 27:
            pass


def print_parts(parts):
    for p in parts:
        pts = p.points[:, 0]
        print(pts[0], pts[len(pts) // 2], pts[-1], '   ||    ', *pts)
