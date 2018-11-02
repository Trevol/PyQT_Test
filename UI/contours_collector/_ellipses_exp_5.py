from UI.contours_collector.contours_collector import ContoursCollector
from UI.contours_collector.contour import intt, box_to_ellipse
import cv2
import numpy as np


def create_test_image():
    img = np.zeros((300, 300), dtype=np.uint8)

    contour_closed = False
    cv2.ellipse(img, (100, 100), (60, 80), 15, 0, 70, 255, 1)
    cv2.ellipse(img, (100, 100), (60, 80), 15, 90, 170, 255, 1)
    # cv2.rectangle(img, (50, 50), (100, 100), 255, -1)

    return img, contour_closed


def normalize_contour(contour):
    if len(contour.shape) == 3:
        return contour.reshape((contour.shape[0], contour.shape[2]))  # (x, 1, y) -> (x, y)
    return contour


def normalize_contours(contours):
    return [normalize_contour(contour) for contour in contours]


def take_n_points(points, n):
    pts_count = len(points)
    if pts_count <= n:
        return points
    if n == 1:
        return [points[round(pts_count/2)]]
    step = (pts_count - 1) / (n - 1)
    indexes = [round(i * step) for i in range(n)]
    return points[indexes]

def polygon_polygon_test(from_poly1, to_poly2, numberOfPoints=4):
    pts = np.unique(from_poly1, axis=1)
    test_pts = take_n_points(pts, numberOfPoints)
    distances = [abs(cv2.pointPolygonTest(to_poly2, tuple(pt), measureDist=True)) for pt in test_pts]
    return sum(distances) / len(distances) #average distance


def centroid(points):
    m = cv2.moments(points)
    m00 = m["m00"]
    if not m00:
        return None
    cx = round(m["m10"] / m00)
    cy = round(m["m01"] / m00)
    return cx, cy

def point(img, pt, color):
    cv2.circle(img, pt, 2, color, -1)

def main():
    img, contour_closed = create_test_image()

    _, contours, _ = cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)

    contours = normalize_contours(contours)
    concatenated_contours = np.vstack(contours)

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    center, axes, angle = box_to_ellipse(*cv2.fitEllipseDirect(concatenated_contours))
    cv2.ellipse(bgr, intt(center), intt(axes), angle, 0, 360, (255, 0, 0), 1)

    for c in contours:
        point(bgr, centroid(c), (0, 255, 0))
        center, axes, angle = box_to_ellipse(*cv2.fitEllipseDirect(c))
        cv2.ellipse(bgr, intt(center), intt(axes), angle, 0, 360, (0, 255, 0), 1)

    cv2.imshow('dd', bgr)
    cv2.waitKey()

main()
