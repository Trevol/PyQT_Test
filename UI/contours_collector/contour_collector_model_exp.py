import cv2
import time
from UI.contours_collector.contours_collector_model import ContoursCollectorModel


def intt(iterable):
    '''
    returns tuple with items rounded to int
    '''
    return tuple(round(i) for i in iterable)


def box_to_ellipse(center, axes, angle):
    return center, (axes[0] / 2, axes[1] / 2), angle


def main():
    frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
    #frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_0068_4533.33_4.53.jpg"
    model = ContoursCollectorModel(frame)

    t0 = time.time()
    contours, edges = model.find_contours()
    print('contours', time.time()-t0)
    contours = [contour for contour in contours if len(contour.points) >= 5]

    t0 = time.time()
    ellipses = [box_to_ellipse(*cv2.fitEllipseDirect(contour.points)) for contour in contours]
    print('fitEllipseDirect', time.time() - t0)

    t0 = time.time()
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    for center, axes, angle in ellipses:
        cv2.ellipse(edges, intt(center), intt(axes), angle, 0, 360, (255, 0, 0), 2)
    print('draw ellipses', time.time() - t0)

    for contour in contours:
        contour.draw(edges, (0,255,0), 1)

    cv2.imshow('ee', cv2.cvtColor(edges, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

    ## area, number of points
    # cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) → img
    # cv2.ellipse(img, box, color[, thickness[, lineType]]) → img


main()
