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

    def __str__(self):
        return str(self.measurements().area)


class FittedEllipse:
    center = None
    axes = None
    angle = None
    area = None

    def __init__(self, contour):
        super(FittedEllipse, self).__init__()
        (self.center, self.axes, self.angle) = box_to_ellipse(*cv2.fitEllipseDirect(contour.points()))
        self.area = self.axes[0] * self.axes[1] * math.pi

    def draw(self, im, color):
        pass


class ContourMeasurements:
    area = None
    centroid = None
    fittedEllipse: FittedEllipse = None

    def __init__(self, contour: Contour):
        super(ContourMeasurements, self).__init__()
        self._contour = contour
        self.area = cv2.contourArea(contour.points())
        self.centroid = centroid(contour.points())
        if contour.len() >= 5:
            self.fittedEllipse = FittedEllipse(contour)

    def draw(self, im, color):
        # draw centroid
        cv2.circle(im, intt(self.centroid), radius=1, color=color, thickness=-1)
        self.fittedEllipse.draw(im, color)


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
