# import cv2
# import numpy as np
# import math
# import utils
# import geometry_utils as geometry
#
#
# class Contour:
#     _points = None
#     _measurements = None
#
#     @classmethod
#     def find(cls, bgr):
#         # cv2.pyrMeanShiftFiltering(calibration_image, 4, 40, maxLevel=2)
#         bgr = cv2.GaussianBlur(bgr, (3, 3), 0, dst=bgr)
#         gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 0, 255, edges=gray)
#         _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         return cls.from_cv_contours(contours)
#
#     @classmethod
#     def from_cv_contours(cls, cv_contours):
#         return [cls(cv_contour) for cv_contour in cv_contours]
#
#     def __init__(self, points):
#         super(Contour, self).__init__()
#         self._points = points
#
#     def points(self):
#         return self._points
#
#     def len(self):
#         return len(self._points)
#
#     def measurements(self):
#         if not self._measurements:
#             self._measurements = ContourMeasurements(self)
#         return self._measurements
#
#     def draw(self, dst_image, color, thickness=-2, draw_measurements=True):
#         cv2.drawContours(dst_image, [self._points], 0, color, thickness=thickness)
#         if draw_measurements:
#             self.measurements().draw(dst_image, color)
#
#     def point_test(self, x, y):
#         dist_from_contour = cv2.pointPolygonTest(self._points, (x, y), measureDist=True)
#         if dist_from_contour >= -10:  # accepted if inside closed contour (> 0) or on edge (==0) or outside but near contour (distance from contour <= -10)
#             return True
#         fitted_ellipse = self.measurements().fitted_ellipse
#         return fitted_ellipse and fitted_ellipse.point_test(x, y)
#
#     def __str__(self):
#         return f'Len: {self.len()}. Measurements: {self.measurements()}'
#
#
# class FittedEllipse:
#     center = None
#     axes = None
#     aspect_ratio = None
#     angle = None
#     area = None
#     _polygon = None
#
#     @classmethod
#     def fromContour(cls, contour):
#         if contour.len() < 5:
#             return None
#         center, axes, angle = cv2.fitEllipseDirect(contour.points())
#         if math.isnan(axes[0]) or math.isnan(axes[1]) or axes[0] == 0 or axes[1] == 0:
#             return None
#         return cls(center, axes, angle)
#
#     def __init__(self, center, axes, angle):
#         super(FittedEllipse, self).__init__()
#         (self.center, self.axes, self.angle) = utils.box_to_ellipse(center, axes, angle)
#         self.area = self.axes[0] * self.axes[1] * math.pi
#         self.aspect_ratio = min(self.axes[0], self.axes[1]) / max(self.axes[0], self.axes[1])
#         self.center_i, self.axes_i, self.angle_i = utils.intt(self.center), utils.intt(self.axes), round(
#             self.angle)  # cache values rounded to int
#
#     def draw(self, im, color):
#         cv2.ellipse(im, self.center_i, self.axes_i, self.angle, 0, 360, color, thickness=2)
#         cv2.circle(im, utils.intt(self.center), 2, color=(255, 255, 255), thickness=-1)
#
#     def polygon(self):
#         if self._polygon is None:
#             # self._polygon = utils.ellipseF2PolyRounded(self.center, self.axes, self.angle, 0, 360, delta=1)
#             self._polygon = cv2.ellipse2Poly(self.center_i, self.axes_i, self.angle_i, 0, 360, delta=1)
#         return self._polygon
#
#     def point_test(self, x, y):
#         return cv2.pointPolygonTest(self.polygon(), (x, y), measureDist=False) >= 0
#
#     def __str__(self):
#         return f'Axes: {self.axes}, AspectRatio: {self.aspect_ratio:.2f}, Angle: {self.angle:2f}, Area: {self.area:.2f}'
#
#
# class ContourMeasurements:
#     area = None
#     centroid = None
#     contour_len = None
#     arc_len = None
#     fitted_ellipse: FittedEllipse = None
#     approx_points = None
#     tails = None  # tails - where approx contour turns on ~180deg
#     extreme_points = None  # extreme points - where approx contour turns on (45 < ang < 160)
#
#     def __init__(self, contour: Contour):
#         super(ContourMeasurements, self).__init__()
#         self._contour = contour
#         self.area = cv2.contourArea(contour.points())
#         self.centroid = utils.centroid(contour.points())
#         self.contour_len = contour.len()
#         self.arc_len = cv2.arcLength(contour.points(), closed=True)
#         self.fitted_ellipse = FittedEllipse.fromContour(contour)
#
#         if contour.len() >= 3:
#             self.approx_points, self.tails, self.extreme_points = ContourMeasurements._approximate(contour.points())
#         else:
#             self.approx_points = ContourMeasurements.empty_points()
#             self.tails = ContourMeasurements.empty_points()  # tails - where approx contour turns on ~180deg
#             self.extreme_points = ContourMeasurements.empty_points()
#
#     @staticmethod
#     def empty_points():
#         return np.empty((0, 2), dtype=np.uint32)
#
#     @staticmethod
#     def _approximate(points):
#         approx_points = cv2.approxPolyDP(points, 1, True)
#
#         angles, approx_points_unwrapped = geometry.compute_angles_vectorized(approx_points)
#         extreme_points = approx_points_unwrapped[(45. < angles) & (angles <= 160.)]
#         tails = approx_points_unwrapped[angles > 160.]
#
#         return approx_points, tails, extreme_points
#
#     def draw(self, im, color):
#         # draw centroid
#         # if self.centroid:
#         #     cv2.circle(im, utils.intt(self.centroid), radius=2, color=color, thickness=-1)
#         # if self.fitted_ellipse:
#         #     self.fitted_ellipse.draw(im, color)
#         for t in self.tails:
#             cv2.circle(im, tuple(t), radius=2, color=(255, 255, 0), thickness=-1)
#
#     def __str__(self):
#         return f'Area: {self.area:.2f}, ArcLen: {self.arc_len:.2f}, FittedEllipse: {self.fitted_ellipse}'
