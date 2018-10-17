import cv2


class Contour:
    points = None
    _area = None

    def __init__(self, points):
        super(Contour, self).__init__()
        self.points = points

    def area(self):
        if self._area is None:
            self._area = cv2.contourArea(self.points)
        return self._area

    def draw(self, dst_image, color, thickness=-2):
        cv2.drawContours(dst_image, [self.points], 0, color, thickness=thickness)
