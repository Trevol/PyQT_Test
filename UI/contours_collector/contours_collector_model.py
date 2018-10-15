from atom.api import Atom, Int, Str, Typed
import cv2
import numpy as np
import utils


class ContoursCollectorModel(Atom):
    blur_kernel_size = Int(3)

    image_rgb = Typed(np.ndarray)

    def __init__(self, imageFile):
        self.image_rgb = cv2.cvtColor(cv2.imread(imageFile), cv2.COLOR_BGR2RGB)

    def collectContours(self):
        image_rgb = cv2.blur(self.image_rgb, ksize=(self.blur_kernel_size, self.blur_kernel_size))
        # image_rgb = cv2.medianBlur(image_rgb, ksize=5)

        imSingleChannel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # image_rgb[:, :, 1]

        # edges = cv2.Canny(im_gray, 255 * 0.55, 255 * 0.8, edges=None, apertureSize=3, L2gradient=True)
        edges = cv2.Canny(imSingleChannel, 255 * 0.00, 255 * 1.00, edges=None, apertureSize=3, L2gradient=False)

        _, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        im = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        #im = np.zeros_like(im)
        colors = utils.make_n_colors(len(contours))
        for i, contour in enumerate(contours):
            drawContour(im, contour, colors[i])

        return im

def drawContour(im, contour, color):
    # contour has shape (N, 1, 2) -> reshape to (N, 2)
    # contour = contour.reshape(contour.shape[0], 2)
    # rows, cols = contour[:, 1], contour[:, 0]
    # im[rows, cols] = color
    cv2.drawContours(im, [contour], 0, color, thickness=1)

