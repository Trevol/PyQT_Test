import math
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __eq__(self, other):
        return self is other or (isinstance(other, Point) and self.x == other.x and self.y == other.y)


class Vector:
    DegreesInRadians = 180 / math.pi

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.x = end.x - start.x
        self.y = end.y - start.y

    def __str__(self):
        return f'{self.start}->{self.end}'

    @classmethod
    def from_coords(cls, x_start, y_start, x_end, y_end):
        return cls(Point(x_start, y_start), Point(x_end, y_end))

    def axis_X_angle_deg(self):
        return self.axis_X_angle_rad() * Vector.DegreesInRadians

    def axis_X_angle_rad(self):
        return math.atan(self.axis_X_angle_tg())

    def axis_X_angle_tg(self):
        delta_x = self.end.x - self.start.x
        if not delta_x:
            return math.inf
        return (self.end.y - self.start.y) / delta_x

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def scalar_product(self, vec):
        return self.x * vec.x + self.y * vec.y


def angle(prev_pt, pt, next_pt):
    # acos(scalar product divide len(vec1)*len()
    vec_prev = Vector(Point(*prev_pt), Point(*pt))
    vec_next = Vector(Point(*pt), Point(*next_pt))
    len_mult = (vec_prev.length() * vec_next.length())
    if len_mult == 0:
        return math.nan
    cos = vec_prev.scalar_product(vec_next) / len_mult
    cos = np.clip(cos, -1, 1)  # round(cos, 5)
    degrees = math.degrees(math.acos(cos))
    return degrees


def enumerate_angles(contour):
    # can we vectorize it with numpy??
    last_index = len(contour) - 1
    for i in range(0, len(contour)):
        prev_pt = contour[i - 1, 0]
        pt = contour[i, 0]
        next_pt = contour[i + 1, 0] if i < last_index else contour[0, 0]
        yield pt, angle(prev_pt, pt, next_pt)


def compute_angles_vectorized(pts):
    if len(pts.shape) == 3:
        pts = pts[:, 0]  # [ [[x, y]]...[[x, y]] ] -> [ [x,y]...[x,y] ]
    next_pts = np.roll(pts, -1, axis=0)
    prev_pts = np.roll(pts, 1, axis=0)

    prev_vecs = pts - prev_pts
    next_vecs = next_pts - pts

    prev_vec_x = prev_vecs[:, 0]
    prev_vec_y = prev_vecs[:, 1]

    next_vec_x = next_vecs[:, 0]
    next_vec_y = next_vecs[:, 1]

    scalar_product = prev_vec_x * next_vec_x + prev_vec_y * next_vec_y

    prev_vec_norm = vector_norm(prev_vec_x, prev_vec_y)
    next_vec_norm = vector_norm(next_vec_x, next_vec_y)

    cos = np.clip(scalar_product / (prev_vec_norm * next_vec_norm), -1, 1)

    angles = np.degrees(np.arccos(cos))
    return angles, pts


def vector_norm(vector_x, vector_y):
    return np.sqrt(vector_x * vector_x + vector_y * vector_y)


def squared_color_distance(c1, c2):
    vec = np.subtract(c1, c2, dtype=np.float32)
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]


def squared_color_distances(color, colors):
    vectors = np.subtract(colors, color, dtype=np.float32)
    v0 = vectors[:, 0]
    v1 = vectors[:, 1]
    v2 = vectors[:, 2]
    squared_distances = v0 * v0 + v1 * v1 + v2 * v2
    return squared_distances


def int32(a):
    return a.round(0).astype(np.int32)


def concat(*args):
    return np.concatenate(args)


def ellipse_main_axes_pts(center, axes, angle):
    x0, y0 = center
    axis_a, axis_b = np.array(axes)  # / 2

    if angle >= 90:
        axis_a, axis_b, angle = axis_b, axis_a, angle - 90
    # TODO: rewrite with matrix multiplication (one or two operation(s))
    radians = np.radians(angle)
    cos = np.cos(radians)
    sin = np.sin(radians)
    axis_a_range = np.arange(-axis_a, axis_a, 1)
    axis_b_range = np.arange(-axis_b, axis_b, 1)

    axis_a_x_points = x0 + axis_a_range * cos
    axis_a_y_points = y0 + axis_a_range * sin

    axis_b_x_points = x0 - axis_b_range * sin
    axis_b_y_points = y0 + axis_b_range * cos

    return int32(concat(axis_a_x_points, axis_b_x_points)), int32(concat(axis_a_y_points, axis_b_y_points))
