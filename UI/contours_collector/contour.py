import cv2, math


class Contour:
    _points = None
    _measurements = None

    def __init__(self, points):
        super(Contour, self).__init__()
        self._points = points

    def points(self):
        return self._points

    def len(self):
        return len(self._points)

    def measurements(self):
        if not self._measurements:
            self._measurements = ContourMeasurements(self)
        return self._measurements

    def draw(self, dst_image, color, thickness=-2):
        cv2.drawContours(dst_image, [self._points], 0, color, thickness=thickness)
        self.measurements().draw(dst_image, color)

    def __str__(self):
        return str(self.measurements().area)

    def point_test(self, x, y):
        dist_from_contour = cv2.pointPolygonTest(self._points, (x, y), measureDist=True)
        if dist_from_contour >= -10: #accepted if inside closed contour (> 0) or on edge (==0) or outside but near contour (distance from contour <= -10)
            return True
        fitted_ellipse = self.measurements().fitted_ellipse
        return fitted_ellipse and fitted_ellipse.point_test(x, y)


class FittedEllipse:
    center = None
    axes = None
    angle = None
    area = None
    polygon = None

    def __init__(self, contour):
        super(FittedEllipse, self).__init__()
        (self.center, self.axes, self.angle) = box_to_ellipse(*cv2.fitEllipseDirect(contour.points()))
        self.center_i, self.axes_i = intt(self.center), intt(self.axes)  # values rounded to int
        self.area = self.axes[0] * self.axes[1] * math.pi

    def draw(self, im, color):
        cv2.ellipse(im, self.center_i, self.axes_i, self.angle, 0, 360, color, thickness=2)
        cv2.circle(im, intt(self.center), 2, color=(255, 255, 255), thickness=-1)

    def point_test(self, x, y):
        if self.polygon is None:
            self.polygon = cv2.ellipse2Poly(self.center_i, self.axes_i, int(self.angle), 0, 360, delta=1)
        return cv2.pointPolygonTest(self.polygon, (x, y), measureDist=False) >= 0


class ContourMeasurements:
    area = None
    centroid = None
    contour_len = None
    arc_len_closed = None
    arc_len_open = None
    fitted_ellipse: FittedEllipse = None

    def __init__(self, contour: Contour):
        super(ContourMeasurements, self).__init__()
        self._contour = contour
        self.area = cv2.contourArea(contour.points())
        self.centroid = centroid(contour.points())
        self.contour_len = contour.len()
        self.arc_len_closed = cv2.arcLength(contour.points(), closed=True)
        self.arc_len_open = cv2.arcLength(contour.points(), closed=False)
        if contour.len() >= 5:
            self.fitted_ellipse = FittedEllipse(contour)

    def draw(self, im, color):
        # draw centroid
        if self.centroid:
            cv2.circle(im, intt(self.centroid), radius=2, color=color, thickness=-1)
        if self.fitted_ellipse:
            self.fitted_ellipse.draw(im, color)


def centroid(points):
    m = cv2.moments(points)
    m00 = m["m00"]
    if not m00:
        return None
    cx = int(m["m10"] / m00)
    cy = int(m["m01"] / m00)
    return cx, cy


def intt(iterable):
    '''
    returns tuple with items rounded to int
    '''
    return tuple(round(i) for i in iterable)


def box_to_ellipse(center, axes, angle):
    return center, (axes[0] / 2, axes[1] / 2), angle
