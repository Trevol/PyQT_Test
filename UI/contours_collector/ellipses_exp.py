from UI.contours_collector.contours_collector import ContoursCollector


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
    cc = ContoursCollector(frame)
    cc.find_contours()
    initial_contours = cc.contoursList.items
    edges = cc.image_edges
    # find contours which already are ellipses
    el = [c for c in initial_contours if is_groud_truth_ellipse(c)]
    print(len(el))


def is_groud_truth_ellipse(contour):
    fitted_ellipse = contour.measurements().fitted_ellipse
    if not fitted_ellipse or fitted_ellipse.area == 0:
        return False
    area_diff = abs(contour.measurements().area - fitted_ellipse.area) / contour.measurements().area
    return area_diff < 0.01
    #todo: compare centers (distance)

main()
