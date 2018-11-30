from UI.contours_collector.contours_collector import ContoursCollector
import cv2
import numpy as np
import utils


def imread(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"

    # 806-288  1099 379
    # 478-651 694-751
    # cc = ContoursCollector(imread(frame)[288:379, 806:1099])
    cc = ContoursCollector(imread(frame)[651:755, 478:694])

    cc.find_contours(method=cv2.CHAIN_APPROX_SIMPLE)

    contours = list(cc.contoursList.items)

    # find contours which already are ground truth ellipses
    gt_ellipses = [c for c in contours if c.is_ground_truth_ellipse()]
    contours = [c for c in contours if c not in gt_ellipses]  # remaining contours
    # cv2.drawContours(edges, [c.points() for c in gt_ellipses], -1, (0, 0, 255), 2)

    for contour in sorted(contours, key=lambda c: len(c.points()), reverse=True):
        contour_points = contour.points()
        contour_points = cv2.approxPolyDP(contour_points, 0.001*cv2.arcLength(contour_points, True), True)
        contour_points = utils.normalize_contour(contour_points)

        edges = cc.image_edges.copy()
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





main()
