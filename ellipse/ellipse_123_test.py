import time
from skimage import data, color, img_as_ubyte, io, draw
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import matplotlib.pyplot as plt
import utils


def imread():
    # return data.coffee()[0:220, 160:420]
    return io.imread('images/contacts/3contacts.jpg')#[15:129, 29:146]


def detect_ellipses(image_rgb):
    edges = canny(color.rgb2gray(image_rgb), sigma=1.0, low_threshold=0.55, high_threshold=0.8)
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    # ellipses = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    ellipses = hough_ellipse(edges, accuracy=17, threshold=264, min_size=51)
    # sorted by accumulator
    return sorted(ellipses, key=lambda i: i[0], reverse=True), edges


def print_info(ellipse_items):
    print(len(ellipse_items))
    for r in ellipse_items:
        accumulator, yc, xc, a, b, orientation = r
        print(accumulator, yc, xc, a, b, orientation)


def draw_rotated_line(im, start, end, origin, orientation, color):
    r0, c0 = utils.rotate_around_point(start, orientation, origin)
    r1, c1 = utils.rotate_around_point(end, orientation, origin)
    r0, c0, r1, c1 = to_int(r0), to_int(c0), to_int(r1), to_int(c1)
    rr, cc = draw.line(r0, c0, r1, c1)
    im[rr, cc] = color


def draw_ellipse_axes(im, yc, xc, a, b, orientation, color):
    draw_rotated_line(im, (yc, xc + b), (yc, xc - b), (yc, xc), orientation, color)
    draw_rotated_line(im, (yc + a, xc), (yc - a, xc), (yc, xc), orientation, color)


def draw_ellipse(ellipse, on_img, with_color=None):
    with_color = with_color or utils.make_n_colors(1)[0]
    _, yc, xc, a, b, orientation = ellipse
    py, px = ellipse_perimeter(to_int(yc), to_int(xc), to_int(a), to_int(b), orientation)
    on_img[py, px] = with_color

    # draw axes
    draw_ellipse_axes(on_img, yc, xc, a, b, orientation, with_color)


def to_int(f):
    return int(round(f))


def do():
    image_rgb = imread()

    t0 = time.time()
    ellipse_items, edges = detect_ellipses(image_rgb)
    print(f'Done. {time.time() - t0} sec')

    print_info(ellipse_items)
    if not len(ellipse_items):
        print('No ellipses detected')
        return;

    edges = color.gray2rgb(img_as_ubyte(edges))
    # ellipse = ellipse_items[0]
    colors = utils.make_n_colors(len(ellipse_items))
    for i in range(len(ellipse_items)):
        draw_ellipse(ellipse_items[i], image_rgb, colors[i])
        draw_ellipse(ellipse_items[i], edges, colors[i])

    _, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(image_rgb)
    ax2.imshow(edges)
    plt.show()


do()
