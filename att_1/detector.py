import cv2

from detection_attempts.att_1.contour import Contour
from detection_attempts.att_1.mean_color_filter import MeanColorFilter, ellipse_axes_mean_color
from detection_attempts.att_1.ellipse_assembler import EllipseAssembler
from detection_attempts.att_1.contour_splitter import ContourSplitter
from cv_named_window import CvNamedWindow
import geometry_utils as geometry
import numpy as np
import utils


class DEBUG:
    def VIS_CONTOURS(self, bgr, contours):
        for c in contours:
            print(c.measurements().approx_points.size, c.len(), self.__calibrator.reference_ellipse.contour.len() // 5)
            VISUALIZE_CONTOURS_PARTS(bgr, c, [])

    @staticmethod
    def separator(img):
        sh = list(img.shape)
        sh[1] = 5  # width
        return np.full(sh, 127, np.uint8)

    @staticmethod
    def VIS_ALL_CONTOURS(img, contours):
        if img.shape[0] > 300 or img.shape[1] > 300:
            return

        print('contours LEN:', len(contours))
        if len(contours) == 0:
            return
        separator = DEBUG.separator(img)
        images = []
        tails_colors = ((0, 127, 0), (0, 255, 0))
        points_color = (0, 0, 0)

        im = DEBUG.__draw_contours(img.copy(), contours, (0, 255, 0), 1, tails_colors, points_color)
        images.extend([im, separator])

        if len(contours) > 1:
            for c in contours:
                im = DEBUG.__draw_contours(img.copy(), [c], (0, 255, 0), 1, tails_colors, points_color)
                images.extend([im, separator])
                print(f'VIS_ALL_CONTOURS: contour len {c.len()}')

        wnd = CvNamedWindow('contours', cv2.WINDOW_NORMAL)
        wnd.imshow(np.hstack(images))
        cv2.waitKey()
        wnd.destroy()

    @staticmethod
    def VIS_POLYGONS(img, polygons):
        if img.shape[0] > 300 or img.shape[1] > 300:
            return
        print('VIS_POLYGONS: polygons count:', len(polygons), [p.len for p in polygons])
        if len(polygons) == 0:
            return

        tails_colors = ((0, 127, 0), (0, 255, 0))
        points_color = (0, 0, 0)
        separator = DEBUG.separator(img)
        images = []

        im = DEBUG.__draw_polygons(img.copy(), polygons, (0, 255, 0), 1, tails_colors, points_color)
        images.extend([im, separator])

        if len(polygons) > 1:
            for p in polygons:
                im = DEBUG.__draw_polygons(img.copy(), [p], (0, 255, 0), 1, tails_colors, points_color)
                images.extend([im, separator])

        wnd = CvNamedWindow('polygons', cv2.WINDOW_NORMAL)
        wnd.imshow(np.hstack(images))
        cv2.waitKey()
        wnd.destroy()

    @staticmethod
    def __draw_polygons(img, polygons, color, thickness, tails_colors, points_color):
        cv2.polylines(img, [p.points for p in polygons], False, color, thickness)
        if tails_colors:
            first_color, last_color = tails_colors
            for p in polygons:
                DEBUG.__put_point(img, p.first_pt, first_color, r=thickness+3)
                DEBUG.__put_point(img, p.last_pt, last_color, r=thickness+2)
        if points_color:
            for p in polygons:
                for pt in p.points:
                    DEBUG.__put_point(img, pt[0], points_color, r=thickness)
        return img

    @staticmethod
    def __draw_contours(img, contours, color, thickness, tails_colors, points_color):
        cv2.drawContours(img, [c.points() for c in contours], -1, color, thickness)

        if tails_colors:
            first_color, last_color = tails_colors
            for c in contours:
                approx_pts = c.measurements().approx_points
                DEBUG.__put_point(img, approx_pts[0, 0], first_color, r=thickness+3)
                DEBUG.__put_point(img, approx_pts[-1, 0], last_color, r=thickness+2)
        if points_color:
            for c in contours:
                approx_pts = c.measurements().approx_points
                for pt in approx_pts:
                    DEBUG.__put_point(img, pt[0], points_color, r=0)
        return img

    @staticmethod
    def __put_point(img, pt, color, r):
        x, y = pt
        if r == 0:
            img[y, x] = color
        else:
            img[y - r:y + r, x - r:x + r] = color


class Detector:
    def __init__(self, calibrator):
        if not calibrator.calibrated:
            raise Exception('not calibrator.calibrated')
        self.__calibrator = calibrator
        self.__ellipse_assembler = EllipseAssembler(calibrator)
        self.__mean_color_filter = MeanColorFilter(calibrator)

    def detect(self, bgr):
        contours = Contour.find(bgr.copy())
        # DEBUG.VIS_ALL_CONTOURS(bgr, contours)
        # skip short and straight (approx pts < 3) contours
        min_len = self.__calibrator.reference_ellipse.contour.len() // 5
        contours = [c for c in contours if c.measurements().approx_points.size != 0 and c.len() > min_len]

        all_parts = []
        for c in contours:
            DEBUG.VIS_ALL_CONTOURS(bgr, [c])
            parts = ContourSplitter.split_contour(c, self.__calibrator.max_contour_angle + 5)
            DEBUG.VIS_POLYGONS(bgr, parts)
            all_parts.extend(parts)

        all_parts = self.clean_polygons(all_parts)

        # DEBUG
        # all_parts = [p for p in all_parts if p.len in (11, 3)]
        # all_parts = [p for p in all_parts if p.within_rectangle((717 - 710, 436 - 370), (778 - 710, 501 - 370))]
        # DEBUG.VIS_POLYGONS(bgr, all_parts)

        all_parts_ = list(all_parts)

        ellipses = self.__ellipse_assembler.assemble_ellipses(all_parts, bgr)
        # ellipses = self.__mean_color_filter.filter_ellipses(ellipses, bgr)
        # ellipses = self.__debug_color_filtering(ellipses, bgr)
        return ellipses, all_parts_

    def clean_polygons(self, polygons):
        # убираем: слишком длинные и слишком короткие контуры
        arc_len_max = self.__calibrator.reference_ellipse.contour.measurements().arc_len * 1.2
        arc_len_min = self.__calibrator.reference_ellipse.contour.measurements().arc_len * 0.1
        def is_valid(poly):
            return poly.len > 2 and arc_len_min < poly.arc_len < arc_len_max and poly.fit_ellipse and poly.fit_ellipse.valid
        return [p for p in polygons if is_valid(p)]

    def __debug_color_filtering(self, ellipses, frame):
        # return ellipses

        print(self.__calibrator.reference_ellipse.mean_color.round(1))

        for (x1, y1), (x2, y2) in [((751, 501), (818, 562)), ((512, 238), (572, 292)), ((267, 532), (329, 601)),
                                   ((1043, 276), (1128, 372)), ((1043, 485), (1148, 635))]:
            mean = np.mean(frame[y1:y2, x1:x2], axis=(0, 1), dtype=np.float32)
            squared_color_distance = geometry.squared_color_distance(self.__calibrator.reference_ellipse.mean_color,
                                                                     mean)
            print(mean.round(1), squared_color_distance.round(1), np.sqrt(squared_color_distance).round(1))
        print('-----')

        mean_color_n_distances = []
        result = []
        for i, el in enumerate(ellipses):
            mean_color = ellipse_axes_mean_color(el, frame).round(1)
            squared_dist = geometry.squared_color_distance(self.__calibrator.reference_ellipse.mean_color, mean_color)
            if 12000 < squared_dist < 13000:
                result.append(el)
            mean_color_n_distances.append((mean_color, squared_dist, np.sqrt(squared_dist)))

        mean_color_n_distances.sort(key=lambda t: t[1])
        for item in mean_color_n_distances:
            print(item)

        return result


def VISUALIZE_CONTOURS_PARTS(bgr, contour, parts):
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
