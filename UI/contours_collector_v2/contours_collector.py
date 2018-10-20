from atom.api import Atom, Int, Str, Typed, ContainerList
import cv2
import numpy as np
import utils
from contour import Contour

blur_kernel_size_options = [1, 3, 5, 7, 9]


class ContoursCollector(Atom):
    blur_kernel_size = Int(3)

    area_filter_lo = Int(-1)
    area_filter_hi = Int(40000)

    canny_thr_1 = Int(0)
    canny_thr_2 = Int(255)

    image_rgb = Typed(np.ndarray)
    contoursImage = Typed(np.ndarray)
    contours = ContainerList()

    def __init__(self, imageFile):
        self.image_rgb = cv2.cvtColor(cv2.imread(imageFile), cv2.COLOR_BGR2RGB)

    def make_contours(self, autoCannyThresholds=False):
        contours, edges = self.find_contours()
        self.contours.clear()
        self.contours.extend(contours)
        self.contoursImage = draw_contours(contours, edges)
        # return draw_contours(contours, self.image_rgb.copy())

    def find_contours(self):
        image_rgb = self._blur_original_image()

        channels = self._get_single_channel_images(image_rgb)

        edges = self._combined_edges(channels)
        _, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        contours = [Contour(points) for points in contours]
        contours = [cont for cont in contours if self.area_filter_accept(cont)]
        return sorted(contours, key=lambda c: c.area(), reverse=True), edges

    # def _find_contours_in_channels(self, channels):
    #     edges = self._combined_edges(channels)
    #     _, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    #     r = [Contour(points) for points in contours]
    #     return sorted(r, key=lambda c: c.area(), reverse=True), edges

    def area_filter_accept(self, cont):
        return self.area_filter_lo <= cont.area() <= self.area_filter_hi

    def _blur_original_image(self):
        if self.blur_kernel_size == 1:
            return self.image_rgb
        # return cv2.GaussianBlur(self.image_rgb, ksize=(self.blur_kernel_size, self.blur_kernel_size), sigmaX=2)
        return cv2.GaussianBlur(self.image_rgb, (self.blur_kernel_size, self.blur_kernel_size), 0)
        # return cv2.blur(self.image_rgb, ksize=(self.blur_kernel_size, self.blur_kernel_size))
        return cv2.medianBlur(self.image_rgb, ksize=self.blur_kernel_size)
        # return cv2.bilateralFilter(self.image_rgb, d=0, sigmaColor=7, sigmaSpace=7)

    def _get_single_channel_images(self, rgb):
        # gray, R G B, A B
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        # hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return [
            gray,
            # rgb[..., 0],
            # rgb[..., 1],
            # rgb[..., 2],
            # lab[..., 0],
            lab[..., 1],
            lab[..., 2],
            # hsv[..., 0],
            # hsv[..., 1],
            # hsv[..., 2],
        ]

    def _edges(self, image):
        # return cv2.Canny(im_gray, 255 * 0.55, 255 * 0.8, edges=None, apertureSize=3, L2gradient=True)
        return cv2.Canny(image, self.canny_thr_1, self.canny_thr_2, edges=None, apertureSize=3, L2gradient=False)

    def _combined_edges(self, channels):
        if not len(channels):
            return None
        accum = np.zeros_like(channels[0])
        for ch in channels:
            edges = self._edges(ch)
            accum = cv2.bitwise_or(accum, edges)
        return accum


def draw_contours(contours, dst):
    colors = utils.make_n_colors(len(contours))
    # np.random.shuffle(colors)
    for i, contour in enumerate(contours):
        contour.draw(dst, colors[i])
    return dst


def auto_canny_example(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
