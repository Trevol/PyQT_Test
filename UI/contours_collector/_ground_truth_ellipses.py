from UI.contours_collector.contours_collector import ContoursCollector
import cv2


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
    cc = ContoursCollector.from_file(frame)
    cc.find_contours()
    initial_contours = cc.contoursList.items
    edges = cc.image_edges.copy()

    # find contours which already are ellipses
    ellipses = [c for c in initial_contours if c.is_ground_truth_ellipse()]

    cv2.drawContours(edges, [c.points() for c in ellipses], -1, (0, 0, 255), 2)
    cv2.imshow('ccc', edges)
    cv2.waitKey()

main()
