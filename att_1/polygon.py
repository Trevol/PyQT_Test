import numpy as np
import math
import cv2
import utils
import geometry_utils as geometry


class Polygon:
    def __init__(self, points):
        self.points = points
        self.len = len(points)
        self.__arc_len = None
        self.__fit_ellipse = None

    def draw_for_presentation(self, image, color, thickness=2):
        ellipse = self.fit_ellipse
        if ellipse is None or not ellipse.valid:
            return
        axis_delta = 3
        center = (round(ellipse.center[0]), round(ellipse.center[1]))
        axes = (round(ellipse.axes[0]) - axis_delta, round(ellipse.axes[1]) - axis_delta)
        cv2.ellipse(image, center, axes, ellipse.angle, 0, 360, color, thickness=thickness, lineType=cv2.LINE_AA)

    def merge(self, other, flip_other):
        other_points = np.flip(other.points, axis=0) if flip_other else other.points
        return Polygon(np.vstack((self.points, other_points)))

    def append(self, other):
        return Polygon(np.vstack((self.points, other.points)))

    def flip_points(self):
        return Polygon(np.flip(self.points, axis=0))

    def distance_from_point(self, pt):
        return abs(cv2.pointPolygonTest(self.points, pt, True))

    @property
    def first_pt(self):
        return self.points[0, 0]

    @property
    def last_pt(self):
        return self.points[-1, 0]

    @property
    def arc_len(self):
        if not self.__arc_len:
            self.__arc_len = cv2.arcLength(self.points, False)
        return self.__arc_len

    @property
    def fit_ellipse(self):
        if self.len < 5:
            return None
        if self.__fit_ellipse is None:
            self.__fit_ellipse = Polygon.FitEllipse(self.points)

        return self.__fit_ellipse

    def is_equivalent(self, other, distance_threshold):
        # "источником точек" д.б. полигон меньшего размера
        if self.arc_len > other.arc_len:
            other, self = self, other

        self_first_pt = self.points[0, 0]
        dist0 = cv2.pointPolygonTest(other.points, (self_first_pt[0], self_first_pt[1]), True)
        if abs(dist0) >= distance_threshold:
            return False
        self_last_pt = self.points[-1, 0]
        dist_last = cv2.pointPolygonTest(other.points, (self_last_pt[0], self_last_pt[1]), True)
        if abs(dist_last) >= distance_threshold:
            return False
        self_middle_pt = self.points[len(self.points) // 2, 0]
        dist_middle = cv2.pointPolygonTest(other.points, (self_middle_pt[0], self_middle_pt[1]), True)
        return abs(dist_middle) < distance_threshold

    def within_rectangle(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2

        xx = self.points[..., 0]
        pts_rect_x1 = xx.min()
        if x1 > pts_rect_x1:
            return False
        pts_rect_x2 = xx.max()
        if x2 < pts_rect_x2:
            return False

        yy = self.points[..., 1]
        pts_rect_y1 = yy.min()
        if y1 > pts_rect_y1:
            return False
        pts_rect_y2 = yy.max()
        if y2 < pts_rect_y2:
            return False
        return True

    class FitEllipse:
        def __init__(self, points):
            fit_ellipse = cv2.fitEllipseDirect(points)
            self.valid = self.validate_ellipse(fit_ellipse)
            if not self.valid:
                return

            self.center, self.axes, self.angle = utils.box_to_ellipse(*fit_ellipse)

            self.axis_a, self.axis_b = self.axes
            self.area = self.axis_a * self.axis_b * np.pi
            self.aspect_ratio = min(self.axes[0], self.axes[1]) / max(self.axes[0], self.axes[1])
            self.__poly = self.__min_rect = self.__main_axes_pts = None

        @staticmethod
        def validate_ellipse(ellipse):
            (x, y), (a, b), _ = ellipse
            if math.isnan(a) or math.isinf(a) or math.isnan(b) or math.isinf(b):
                return False
            # TODO: compare axes a/b with ref. ellipse
            if a < 5 or b < 5:
                return False
            # TODO: compare with polygon rect (or frame dimensions)
            if x > 2000 or y > 2000:
                return False
            if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
                return False
            return True

        class DEBUG:
            @staticmethod
            def show_points(points):
                w = points[..., 0].max() + 5
                h = points[..., 1].max() + 5
                im = np.zeros((h, w), np.uint8)
                cv2.polylines(im, [points], False, 255, 1)
                cv2.namedWindow('DEBUG_POINTS', flags=cv2.WINDOW_NORMAL)
                cv2.imshow('DEBUG_POINTS', im)
                cv2.waitKey()
                cv2.destroyWindow('DEBUG_POINTS')

        @property
        def min_rect(self):
            pass

        @property
        def main_axes_pts(self):
            if self.__main_axes_pts is None:
                self.__main_axes_pts = geometry.ellipse_main_axes_pts(self.center, self.axes, self.angle)
            return self.__main_axes_pts

        @property
        def poly(self):
            if self.__poly is None:
                self.__poly = utils.ellipseF2Poly(self.center, self.axes, self.angle, 0, 360, 1)
            return self.__poly
