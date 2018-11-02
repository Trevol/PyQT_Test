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


def polygon_polygon_test22(from_poly1, to_poly2, numberOfPoints=4):
    pts = np.unique(from_poly1, axis=1)
    # todo: may be taking by distance in array will be faster then random indexes
    rnd_indexes = np.random.randint(0, pts.shape[0], size=numberOfPoints)
    test_pts = pts[rnd_indexes]

    distances = [abs(cv2.pointPolygonTest(to_poly2, tuple(pt), measureDist=True)) for pt in test_pts]
    return sum(distances) / len(distances)


def polygon_polygon_test(from_poly1, to_poly2, numberOfPoints=4):
    pts = np.unique(from_poly1, axis=1)
    pts_count = len(pts)
    if pts_count <= numberOfPoints:
        test_pts = pts
    else:
        step = (pts_count-1) / (numberOfPoints - 1)
        indexes = [round(i * step) for i in range(numberOfPoints)]
        test_pts = pts[indexes]

    distances = [abs(cv2.pointPolygonTest(to_poly2, tuple(pt), measureDist=True)) for pt in test_pts]
    return sum(distances) / len(distances)

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

    stacked = np.vstack(contours)

    center, axes, angle = box_to_ellipse(*cv2.fitEllipseDirect(stacked))

    ellipse_poly = cv2.ellipse2Poly(intt(center), intt(axes), int(angle), 0, 360, 1)

    # compute and show centroids of each contours, staked contour, fitted ellipse and ellipse_poly
    for i, c in enumerate(contours):
        #print('c', i, centroid(c))
        point()

    print('stacked', centroid(stacked))
    print('fitted', center)
    print('ellipse_poly', centroid(ellipse_poly))

main()
