import cv2
import utils
import geometry_utils as geometry


def walk_cv_contour_points_by_kbd(contour, base_img, delta=5):
    def point(img, pt, color, radius=2, thickness=-1):
        cv2.circle(img, tuple(pt), radius, color, thickness=thickness)

    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    outliers = []

    base_img = base_img.copy()
    point(base_img, contour[0, 0], red, 3, 1)  # start point
    point(base_img, contour[-1, 0], green, 3, 1)  # end point

    last_index = len(contour) - 1
    for i in range(0, len(contour)):
        prev_pt = contour[i - 1, 0]
        pt = contour[i, 0]
        next_pt = contour[i + 1, 0] if i < last_index else contour[0, 0]

        ang = geometry.angle(prev_pt, pt, next_pt)
        if ang > 80:
            outliers.append(pt)
        print(round(ang, 1))

        img = base_img.copy()
        cv2.drawContours(img, [contour], -1, blue, thickness=2)
        point(img, pt, red)

        for out in outliers:
            point(img, out, green)

        utils.imshow(dd=img)
        if cv2.waitKey() == 27:
            return

    while cv2.waitKey() != 27:
        pass


def walk_cv_contours_by_kbd(polylines, base_img):
    for poly in polylines:
        img = base_img.copy()
        cv2.polylines(img, [poly], False, utils.random_color(), thickness=2)
        utils.imshow(dd=img)
        if cv2.waitKey() == 27:
            return

    if len(polylines):
        while cv2.waitKey() != 27:
            pass


def walk_contours_by_kbd(contours, base_img, color=(0, 0, 255), thickness=1, draw_measurements=True, draw_visited=False,
                         on_next_contour=None):
    if draw_visited:
        img = base_img.copy()
    for contour in contours:
        if not draw_visited:
            img = base_img.copy()
        contour.draw(img, color, thickness=thickness, draw_measurements=draw_measurements)

        on_next_contour and on_next_contour(contour)

        utils.imshow(contour=img)
        if cv2.waitKey() == 27:
            return

    if len(contours):
        while cv2.waitKey() != 27:
            pass


def draw_contours(contours, base_img, color=None, thickness=1, draw_measurements=True):
    if len(contours) == 0:
        return
    colors = [color for c in range(len(contours))] if color else utils.make_n_colors(len(contours))

    for i, contour in enumerate(contours):
        contour.draw(base_img, colors[i], thickness=thickness, draw_measurements=draw_measurements)


def draw_polylines(polylines, base_img, color=None, thickness=1, draw_measurements=True):
    if len(polylines) == 0:
        return
    colors = [color for c in range(len(polylines))] if color else utils.make_n_colors(len(polylines))

    for i, polyline in enumerate(polylines):
        cv2.polylines(base_img, [polyline], False, colors[i], thickness=thickness)
