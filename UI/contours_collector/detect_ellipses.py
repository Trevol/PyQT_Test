import cv2
import numpy as np
import utils
from UI.contours_collector.contour import Contour
import geometry_utils as geometry
from ellipse import Ellipse
from contour_visualize import walk_contours_by_kbd, draw_contours, walk_cv_contour_points_by_kbd


# def read():
#     frame = "D:/DiskE/Computer_Vision_Task/frames_6/f_2770_184666.67_184.67.jpg"
#     imread = cv2.imread  # utils.imread_rgb
#     # return imread(frame)[651:755, 478:694]
#     # return imread(frame)[461:554, 718:822]  # 613 456  926 664
#     # return imread(frame)[456:664, 613:926]  # 613 456  926 664
#     # return imread(frame)[288: 379, 806: 1099] # Этот
#     # return imread(frame)[648:877, 1132:1586]
#     # return imread(frame)[654:870, 795:917]
#     return imread(frame)

def read():
    frame = "D:/DiskE/Computer_Vision_Task/frames_2/f_62_4133.33_4.13.jpg"
    imread = cv2.imread  # utils.imread_rgb
    # return imread(frame)[651:755, 478:694]
    # return imread(frame)[542:613, 800:903, :]
    return imread(frame)


def find_contours(image):
    _, cv_contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = Contour.from_cv_contours(cv_contours)
    return contours


def find_edges(gray):
    return cv2.Canny(gray, 0, 150) #214


def extract_ellipses(contours, extract_garbage=True):
    ellipses = []
    garbage = []
    other_contours = []
    for c in contours:
        if extract_garbage and Ellipse.is_strong_garbage(c):
            garbage.append(c)
        elif Ellipse.is_strong_ellipse(c):
            ellipses.append(c)
        else:
            other_contours.append(c)

    return ellipses, other_contours, garbage


def hide_contours(contours, edges):
    if len(contours) == 0:
        return
    cv2.drawContours(edges, [c.points() for c in contours], -1, 0, -1)  # need to fill entire area of contour
    # pts = np.vstack([c.points()[:, 0] for c in contours])
    # edges[pts[:, 1:2], pts[:, 0:1]] = 0


def find_ellipses(edges):
    def hide_bad_corners(contours, edges):
        xps = np.vstack(c.measurements().extreme_points for c in contours)
        edges[xps[:, 1:2], xps[:, 0:1]] = 0

    edges = edges.copy()

    contours = find_contours(edges)
    ellipses, contours, garbage = extract_ellipses(contours, extract_garbage=False)
    hide_contours(ellipses, edges)

    hide_bad_corners(contours, edges)

    contours = find_contours(edges)
    ellipses_2, contours, garbage_2 = extract_ellipses(contours, extract_garbage=False)
    hide_contours(ellipses_2, edges)

    hide_bad_corners(contours, edges)

    ellipses.extend(ellipses_2)
    garbage.extend(garbage_2)

    contours = find_contours(edges)
    ellipses_2, contours, garbage_2 = extract_ellipses(contours, extract_garbage=True)
    hide_contours(ellipses_2, edges)

    ellipses.extend(ellipses_2)
    garbage.extend(garbage_2)

    return ellipses, edges, contours, garbage


##############################
def assemble_ellipses(parts):
    ellipses = []
    parts = sorted(parts, key=lambda c: c.len(), reverse=True)

    # start from contours closest to its fitted ellipse
    for p in filter(is_close_to_ellipse, parts):
        ellipses.append([p])
        parts.remove(p)

    # continue find parts for closest contours
    for p in ellipses:
        search_ellipse_parts(p, parts)

    for p in list(parts):
        el = [p]
        search = list(parts)

        if p in search:
            search.remove(p)

        search_ellipse_parts(el, search)
        if len(el) > 1 and is_complete_ellipse_parts(el):  # todo: check 'el' for "completeness". arcLen??
            ellipses.append(el)
            for pp in el[1:]:
                if pp in parts:
                    parts.remove(pp)

    return ellipses, []


def search_ellipse_parts(ellipse_parts, other_contours):
    # для каждого контура найти контуры (среди остальных):
    #   - с миним. расстоянием между "хвостами" (< 10)
    #   - все вместе должны помещаться в эллипс (cv2.fitEllipse) с миним. расстоянием
    parts_found = False
    for sc in list(other_contours):
        for cc in list(ellipse_parts):
            if is_close_by_tails(sc, cc, 0.4) and dist_from_fitted_ellipse([*ellipse_parts, sc]) < 1:
                parts_found = True
                ellipse_parts.append(sc)
                if sc in other_contours:
                    other_contours.remove(sc)
    if parts_found:  # если за текущий проход что-то нашли, то выполняем еще одну попытку поска - м.б. что-то еще найдем для данного эллипса
        search_ellipse_parts(ellipse_parts, other_contours)


def is_complete_ellipse_parts(parts):
    arc_len = sum([p.measurements().arc_len for p in parts])
    # find max dist
    max_dist = -1
    for p1 in parts:
        for p2 in parts:
            if p1 == p2:
                continue
            tails1 = p1.measurements().tails
            tails2 = p2.measurements().tails
            max_dist = max(max_dist,
                           distance_pt(tails1[0], tails2[0]), distance_pt(tails1[0], tails2[1]),
                           distance_pt(tails1[1], tails2[0]), distance_pt(tails1[1], tails2[1]))
    if max_dist == -1:
        raise Exception('max_dist == -1')
    is_complete = max_dist == 0 or (max_dist / arc_len) < 0.2
    # if is_complete:
    #     print(max_dist, arc_len, max_dist / arc_len)
    return is_complete


def is_close_by_tails(contour1, contour2, threshold):
    # todo: threshold сделать относительным. Только относительно чего??? arcLen??
    tails1 = contour1.measurements().tails
    tails2 = contour2.measurements().tails
    if len(tails1) < 2 or len(tails2) < 2:
        return False
    arc_len = contour1.measurements().arc_len + contour2.measurements().arc_len
    return distance_pt(tails1[0], tails2[0]) / arc_len <= threshold or \
           distance_pt(tails1[0], tails2[1]) / arc_len <= threshold or \
           distance_pt(tails1[1], tails2[0]) / arc_len <= threshold or \
           distance_pt(tails1[1], tails2[1]) / arc_len <= threshold


def is_close_to_ellipse(contour: Contour):
    tails = contour.measurements().tails
    if len(tails) == 1:
        return False

    if len(tails) == 0:
        dist_tails = 0
    else:
        dist_tails = distance(tails[0:1], tails[1:2])[0]

    arc_len = contour.measurements().arc_len
    tail_to_arc_ratio = dist_tails / arc_len
    is_close = tail_to_arc_ratio < 0.2 and dist_from_fitted_ellipse([contour]) < 1
    # if is_close:
    #     print(f'arc_len={arc_len} dist_tails={dist_tails} ratio={tail_to_arc_ratio} dist_from_ellipse={dist_from_fitted_ellipse([contour])}')
    return is_close


def dist_from_fitted_ellipse(contours):
    points = np.vstack([c.points() for c in contours])
    center, exes, angle = utils.box_to_ellipse(*(cv2.fitEllipseDirect(points)))
    ellipse_poly = utils.ellipseF2Poly(center, exes, angle, 0, 360, 1)
    return utils.polygon_polygon_test(points, ellipse_poly, -1)


def squared_distance(pts1, pts2):
    vec = pts1 - pts2
    vec_x = vec[:, 0]
    vec_y = vec[:, 1]
    return vec_x * vec_x + vec_y * vec_y


def distance(pts1, pts2):
    return np.sqrt(squared_distance(pts1, pts2))


def distance_pt(pt1, pt2):
    vec = pt1 - pt2
    x = vec[0]
    y = vec[1]
    return np.sqrt(x * x + y * y)


##################################
def agg_length(contours):
    return sum([c.measurements().arc_len for c in contours])


max_area = 0
max_area_thr = 0


def calibrate_or_apply_calibrated(ellipses, assembled_ellipses):
    global max_area, max_area_thr

    def ellipse_area(ell):
        return ell.measurements().fitted_ellipse.area

    def assembled_area(parts):
        pts = np.vstack([p.points() for p in parts])
        center, axes, angle = utils.box_to_ellipse(*cv2.fitEllipseDirect(pts))
        return axes[0] * axes[1] * np.pi

    if len(ellipses) > 0:
        max_area_ellipses = ellipse_area(max(ellipses, key=ellipse_area))
        max_area = max(max_area, max_area_ellipses)
        max_area_thr = max_area * .7

    if len(assembled_ellipses) > 0:
        max_area_assembled = assembled_area(max(assembled_ellipses, key=assembled_area))
        max_area = max(max_area, max_area_assembled)
        max_area_thr = max_area * .7

    if max_area:
        ellipses = filter(lambda el: ellipse_area(el) >= max_area_thr, ellipses)
        assembled_ellipses = filter(lambda el: assembled_area(el) >= max_area_thr, assembled_ellipses)
        return list(ellipses), list(assembled_ellipses)

    return ellipses, assembled_ellipses


def detect_and_visualize(bgr, visualize_edges=True):
    bgr = cv2.GaussianBlur(bgr, (3, 3), 0, dst=bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = find_edges(gray)

    ellipses, edges_2, rest_of_contours, garbage_contours = find_ellipses(edges)
    #assembled_ellipses, unused_contours = assemble_ellipses(rest_of_contours)

    #ellipses, assembled_ellipses = calibrate_or_apply_calibrated(ellipses, assembled_ellipses)

    #########
    im = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if visualize_edges else bgr
    draw_contours(ellipses, im, (0, 255, 0), thickness=2, draw_measurements=False)

    # for ellipse_parts in assembled_ellipses:
    #     # color = utils.random_color()
    #     for part in ellipse_parts:
    #         part.draw(im, (0, 0, 255), thickness=2, draw_measurements=False)
    #     # draw_fitted_ellipse(im, ellipse_parts, color)
    #     # print(dist_from_fitted_ellipse(ellipse_parts))
    return im


def main():
    bgr = read()
    im = detect_and_visualize(bgr, visualize_edges=True)
    utils.imshow(cv2.WINDOW_NORMAL, im=im)
    cv2.waitKey()


def draw_fitted_ellipse(img, contours, color):
    pts = np.vstack([c.points() for c in contours])
    center, axes, angle = utils.box_to_ellipse(*cv2.fitEllipseDirect(pts))
    cv2.ellipse(img, utils.intt(center), utils.intt(axes), angle, 0, 360, color, 1)


np.seterr(all='raise')
if __name__ == '__main__':
    main()
