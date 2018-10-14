import numpy as np
from skimage import draw, io
from utils import rotate_around_point
import math


# def rc_to_xy(rc):
#     y, x = rc
#     return x, y
# def draw_rotated_line_old(im, start, end, origin, orientation, color):
#     (c0, r0) = rotate_around_point(rc_to_xy(start), -orientation, origin)
#     (c1, r1) = rotate_around_point(rc_to_xy(end), -orientation, origin)
#     (r0, c0), (r1, c1) = (int(round(r0)), int(round(c0))), (int(round(r1)), int(round(c1)))
#     rr, cc = draw.line(r0, c0, r1, c1)
#     im[rr, cc] = color

def draw_rotated_line(im, start, end, origin, orientation, color):
    r0, c0 = rotate_around_point(start, orientation, origin)
    r1, c1 = rotate_around_point(end, orientation, origin)
    r0, c0, r1, c1 = int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))
    rr, cc = draw.line(r0, c0, r1, c1)
    im[rr, cc] = color


def make_image():
    im = (np.ones((201, 201, 3)) * 255).astype(np.uint8)
    rr, cc = draw.ellipse_perimeter(100, 100, 30, 70, orientation=math.radians(17))
    im[rr, cc] = (0, 0, 0)
    return im


def imshow(im):
    io.imshow(im), io.show()


def main():
    im = make_image()
    draw_rotated_line(im, (100, 170), (100, 30), (100, 100), math.radians(0), (0, 0, 255))
    draw_rotated_line(im, (100, 170), (100, 30), (100, 100), math.radians(15), (0, 255, 0))
    draw_rotated_line(im, (100, 170), (100, 30), (100, 100), math.radians(17), (255, 0, 0))
    imshow(im)


main()
