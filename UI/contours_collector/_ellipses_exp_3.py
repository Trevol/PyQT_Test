from UI.contours_collector.contours_collector import ContoursCollector
import cv2
import numpy as np


def main():
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), 255, -1)
    cv2.rectangle(img, (150, 50), (250, 150), 255, -1)

    _, contours, _ = cv2.findContours(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    print(len(contour))
    contour = contour.reshape((contour.shape[0], contour.shape[2]))  # (x, 1, y) -> (x, y)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_bgr, [contour], -1, (255, 0, 0), 1)
    cv2.circle(img_bgr, tuple(contour[0]), 2, (0, 0, 255), -1)
    cv2.circle(img_bgr, tuple(contour[-1]), 2, (0, 255, 0), -1)

    # animate point from 0 to -1
    for point in contour:
        im = img_bgr.copy()
        cv2.circle(im, tuple(point), 3, (0, 0, 255), 1)
        cv2.imshow('ccc', im)
        if cv2.waitKey() == 27:
            return

    while cv2.waitKey() != 27:
        pass

    return


    cc.find_contours(method=cv2.CHAIN_APPROX_SIMPLE)

    contour_points = contours[0].points()
    contour_points = contour_points.reshape((contour_points.shape[0], contour_points.shape[2]))  # (x, 1, y) -> (x, y)

    cv2.drawContours(edges, [contour_points], 0, (255, 0, 0), 1)
    cv2.circle(edges, tuple(contour_points[0]), 2, (0, 0, 255), -1)
    cv2.circle(edges, tuple(contour_points[-1]), 2, (0, 255, 0), -1)

    # animate point from 0 to -1
    for point in contour_points:
        im = edges.copy()
        cv2.circle(im, tuple(point), 3, (0, 0, 255), 1)
        cv2.imshow('ccc', im)
        if cv2.waitKey() == 27:
            return

    while cv2.waitKey() != 27:
        pass


def is_groud_truth_ellipse(contour):
    fitted_ellipse = contour.measurements().fitted_ellipse
    if not fitted_ellipse or fitted_ellipse.area == 0:
        return False
    area_diff = abs(contour.measurements().area - fitted_ellipse.area) / contour.measurements().area
    return area_diff < 0.01
    # todo: analize distance between contour and fitted ellipse centers


main()
