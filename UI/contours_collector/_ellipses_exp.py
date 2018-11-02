from UI.contours_collector.contours_collector import ContoursCollector
import cv2


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
    cc = ContoursCollector.from_file(frame)
    cc.find_contours()
    initial_contours = cc.contoursList.items
    edges = cc.image_edges.copy()

    # find contours which already are ellipses
    ellipses = [c for c in initial_contours if is_groud_truth_ellipse(c)]

    cv2.drawContours(edges, [c.points() for c in ellipses], -1, (0, 0, 255), 2)
    cv2.imshow('ccc', edges)
    cv2.waitKey()




def is_groud_truth_ellipse(contour):
    fitted_ellipse = contour.measurements().fitted_ellipse
    if not fitted_ellipse or fitted_ellipse.area == 0:
        return False
    area_diff = abs(contour.measurements().area - fitted_ellipse.area) / contour.measurements().area
    return area_diff < 0.01
    #todo: analize distance between contour and fitted ellipse centers

main()
