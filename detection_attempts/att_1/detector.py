import cv2

from detection_attempts.att_1.contour import Contour
from detection_attempts.att_1.polygon import Polygon
from detection_attempts.att_1.ellipse_assembler import EllipseAssembler
import geometry_utils as geometry
import numpy as np
import utils


class Detector:
    def __init__(self, calibrator):
        if not calibrator.calibrated:
            raise Exception('not calibrator.calibrated')
        self.__calibrator = calibrator
        self.__ellipse_assembler = EllipseAssembler(calibrator)

    def __VIS_CONTOURS(self, bgr, contours):
        for c in contours:
            print(c.measurements().approx_points.size, c.len(), self.__calibrator.reference_ellipse.len() // 5)
            visualize_contours_parts(bgr, c, [])

    def detect(self, bgr):
        contours = Contour.find(bgr)

        # skip short and straight (approx pts < 3) contours
        min_len = self.__calibrator.reference_ellipse.len() // 5
        contours = [c for c in contours if c.measurements().approx_points.size != 0 and c.len() > min_len]

        all_parts = []
        for c in contours:
            parts = self.split(c)  # TODO: сразу фильтровать??
            # visualize_contours_parts(bgr, c, parts)
            all_parts.extend(parts)

        # убираем: слишком длинные и слишком короткие контуры
        arc_len_max = self.__calibrator.reference_ellipse.measurements().arc_len * 1.2
        arc_len_min = self.__calibrator.reference_ellipse.measurements().arc_len * 0.12
        all_parts = [p for p in all_parts if arc_len_min < p.arc_len < arc_len_max]

        ellipses = self.__detect_ellipses(all_parts, bgr)

        # убираем: не "фитятся" в эллипс
        return ellipses

    def __detect_ellipses(self, parts, bgr):
        ellipses = self.detect_complete_ellipses(parts)
        self.__ellipse_assembler.assemble_ellipses(parts, ellipses, bgr)
        return ellipses

    def detect_complete_ellipses(self, parts):
        ellipses = []
        for i, p in enumerate(parts):
            if self.__calibrator.is_close_to_ref_ellipse(p):
                ellipses.append(p)
                parts.pop(i)
        return ellipses

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


def visualize_contours_parts(bgr, contour, parts):
    cv2.namedWindow('debug', cv2.WINDOW_NORMAL)

    # im = bgr.copy()
    im = bgr
    cv2.drawContours(im, [contour.points()], -1, (0, 255, 0), 1)
    print(contour.len(), len(parts))
    cv2.imshow('debug', im)
    cv2.waitKey()

    for p in parts:
        # im = bgr.copy()
        cv2.polylines(im, [p.points], False, utils.random_color(), 2)
        cv2.imshow('debug', im)
        cv2.waitKey()

    cv2.drawContours(im, [contour.points()], -1, (255, 255, 255), 1)
    for p in parts:
        cv2.polylines(im, [p.points], False, (255, 255, 255), 2)

    while cv2.waitKey() != 27:
        pass


def visualize_parts(bgr, parts):
    cv2.polylines(bgr, [p.points for p in parts], False, (0, 0, 0), 2)
    cv2.waitKey()


def print_parts(parts):
    for p in parts:
        pts = p.points[:, 0]
        print(pts[0], pts[len(pts) // 2], pts[-1], '   ||    ', *pts)
