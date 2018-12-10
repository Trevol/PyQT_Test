import numpy as np
import cv2
import utils


class Polygon:
    def __init__(self, points):
        self.points = points
        self.len = len(points)
        self.__arc_len = None
        self.__fit_ellipse = None

    def merge(self, other, flip_other):
        other_points = np.flip(other.points, axis=0) if flip_other else other.points
        return Polygon(np.vstack((self.points, other_points)))

    def append(self, other):
        return Polygon(np.vstack((self.points, other.points)))

    def flip_points(self):
        return Polygon(np.flip(self.points, axis=0))

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

    class FitEllipse:
        def __init__(self, points):
            self.center, self.axes, self.angle = utils.box_to_ellipse(*cv2.fitEllipseDirect(points))
            # self.center, self.axes, self.angle = utils.box_to_ellipse(*cv2.fitEllipse(points))
            # self.center, self.axes, self.angle = utils.box_to_ellipse(*cv2.fitEllipseAMS(points))
            self.axis_a, self.axis_b = self.axes
            self.area = self.axis_a * self.axis_b * np.pi
            self.aspect_ratio = min(self.axes[0], self.axes[1]) / max(self.axes[0], self.axes[1])
            self.__poly = None

        @property
        def poly(self):
            if self.__poly is None:
                self.__poly = utils.ellipseF2Poly(self.center, self.axes, self.angle, 0, 360, 1)
            return self.__poly
