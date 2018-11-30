from threading import Timer
import colorsys
import math
import cv2
import numpy as np
import random
import time


def int_to_str(val):
    return f'{val}'


def timeit(fn, iterations, *args, **kwargs):
    t0 = time.time()
    for i in range(iterations):
        fn(*args, **kwargs)
    return round(time.time() - t0, 3)


def debounce(wait):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """

    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it():
                fn(*args, **kwargs)

            try:
                debounced.t.cancel()
            except(AttributeError):
                pass
            debounced.t = Timer(wait, call_it)
            debounced.t.start()

        return debounced

    return decorator


def first_or_default(iterable, criteria):
    for i, item in enumerate(iterable):
        if criteria(item):
            return item, i
    return None, None


def put_frame_pos(frame, pos):
    cv2.putText(frame, str(pos), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


def imshow(flag=cv2.WINDOW_NORMAL, **name_img):
    for name in name_img:
        cv2.namedWindow(name, flag)
        cv2.imshow(name, name_img[name])


def make_n_colors(N, bright=True):
    """
    Generate random colors. To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    return [
        [int(a * 255) for a in colorsys.hsv_to_rgb(i / N, 1, brightness)]
        for i in range(N)]


def random_color():
    return rand255(), rand255(), rand255()


def rand255():
    return round(random.random() * 255)


def imread_rgb(file):
    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)


def rotate_around_point(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def take_n_points(points, n):
    pts_count = len(points)
    if pts_count <= n:
        return points
    if n == 1:
        return [points[round(pts_count / 2)]]
    step = (pts_count - 1) / (n - 1)
    indexes = [round(i * step) for i in range(n)]
    return points[indexes]


def polygon_polygon_test(from_poly1, to_poly2, numberOfPoints=4):
    if numberOfPoints < 1:  # use all points
        test_pts = from_poly1
    else:
        pts = np.unique(from_poly1, axis=1)
        test_pts = take_n_points(pts, numberOfPoints)

    distances = [abs(cv2.pointPolygonTest(to_poly2, tuple(pt[0]), measureDist=True)) for pt in test_pts]
    return sum(distances) / len(distances)  # average distance


def centroid(points):
    m = cv2.moments(points)
    m00 = m["m00"]
    if not m00:
        return None
    cx = round(m["m10"] / m00)
    cy = round(m["m01"] / m00)
    return cx, cy


def intt(iterable):
    '''
    returns tuple with items rounded to int
    '''
    return tuple(round(i) for i in iterable)


def box_to_ellipse(center, axes, angle):
    return center, (axes[0] / 2, axes[1] / 2), angle


def normalize_contour(contour):
    if len(contour.shape) == 3:
        return contour.reshape((contour.shape[0], contour.shape[2]))  # (x, 1, y) -> (x, y)
    return contour


def normalize_contours(contours):
    return [normalize_contour(contour) for contour in contours]


def ellipseF2PolyRounded(center, axes, angle, arc_start, arc_end, delta):
    pts = ellipseF2Poly(center, axes, angle, arc_start, arc_end, delta)
    return np.round(pts, decimals=0, out=pts)  # .astype(np.int32)


def ellipseF2Poly(center, axes, angle, arc_start, arc_end, delta):
    # todo: verify params - like in original ellipse2Poly

    width = axes[0]
    height = axes[1]

    angles = np.arange(arc_start, arc_end + 1, delta, dtype=np.float32)
    pts = np.empty((angles.shape[0], 2), dtype=np.float32)
    angles = np.radians(angles, out=angles)

    pts[:, 0] = width * np.cos(angles)
    pts[:, 1] = height * np.sin(angles)

    # move to center and rotate
    return np.add(rotate(pts, angle, out=pts), center, out=pts)


def rotate(pts, deg, out=None):
    rad = np.radians(deg)
    cos = np.cos(rad)
    sin = np.sin(rad)
    rot_matrix = np.array([
        [cos, sin],
        [-sin, cos]
    ], dtype=np.float32)
    return np.dot(pts, rot_matrix, out=out)

# def ellipseF2Poly_numpy_rounded_slow(center, axes, angle, arc_start, arc_end, delta):
#     pts = ellipseF2Poly_numpy(center, axes, angle, arc_start, arc_end, delta)
#     if pts.size == 0:
#         return pts
#     pts = np.round(pts, decimals=0, out=pts)
#
#     return _dedublicate(pts.astype(np.int32))
#
#
# def _dedublicate(pts):
#     result = np.empty_like(pts)
#
#     i = 1
#     prev_item = pts[0]
#     result[0] = prev_item
#     for item in pts[1:]:
#         if not np.array_equal(item, prev_item):
#             result[i] = item
#             prev_item = item
#             i = i + 1
#     # resize to i-length
#     new_shape = tuple([i, *result.shape[1:]])
#     result.resize(new_shape)
#     return result

# def ellipseF2PolyRounded(center, axes, angle, arc_start, arc_end, delta):
#     pts = []
#     rad = math.radians(angle)
#     alpha = math.cos(rad)
#     beta = math.sin(rad)
#
#     center_x = center[0]
#     center_y = center[1]
#     width = axes[0]
#     height = axes[1]
#
#     prev_pt = None
#     angle = arc_start
#     while angle < arc_end + delta:
#         if angle > arc_end:
#             angle = arc_end
#
#         rad = math.radians(angle)
#         x = width * math.cos(rad)
#         y = height * math.sin(rad)
#         pt_x = round(center_x + x * alpha - y * beta)
#         pt_y = round(center_y + x * beta + y * alpha)
#         pt = [pt_x, pt_y]
#         if pt != prev_pt:
#             pts.append(pt)
#             prev_pt = pt
#
#         angle = angle + delta
#
#     return np.array(pts, dtype=np.int32)
#
#
# def ellipseF2Poly(center, axes, angle, arc_start, arc_end, delta):
#     # todo: verify params - like in original ellipse2Poly
#     pts = []
#     rad = math.radians(angle)
#     alpha = math.cos(rad)
#     beta = math.sin(rad)
#
#     center_x = center[0]
#     center_y = center[1]
#     width = axes[0]
#     height = axes[1]
#
#     angle = arc_start
#     while angle < arc_end + delta:
#         if angle > arc_end:
#             angle = arc_end
#
#         rad = math.radians(angle)
#         x = width * math.cos(rad)
#         y = height * math.sin(rad)
#         pt_x = center_x + x * alpha - y * beta
#         pt_y = center_y + x * beta + y * alpha
#         pt = [pt_x, pt_y]
#         pts.append(pt)
#
#         angle = angle + delta
#
#     return np.array(pts, dtype=np.float32)
