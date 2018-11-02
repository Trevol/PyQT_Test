from UI.contours_collector.contours_collector import ContoursCollector
import cv2
import numpy as np


def imread(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"

    # 806-288  1099 379
    cc = ContoursCollector(imread(frame)[288:379, 806:1099])

    cc.find_contours(method=cv2.CHAIN_APPROX_SIMPLE)

    contours = list(cc.contoursList.items)
    edges = cc.image_edges.copy()

    # find contours which already are ground truth ellipses
    gt_ellipses = [c for c in contours if is_groud_truth_ellipse(c)]
    contours = [c for c in contours if c not in gt_ellipses]  # remaining contours
    # cv2.drawContours(edges, [c.points() for c in gt_ellipses], -1, (0, 0, 255), 2)

    contour_points = contours[0].points()
    contour_points = contour_points.reshape((contour_points.shape[0], contour_points.shape[2]))  # (x, 1, y) -> (x, y)

    cv2.drawContours(edges, [contour_points], 0, (255, 0, 0), -1)
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
