import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import canny
import cv2
import utils


def imread():
    # return data.coffee()[0:220, 160:420]
    return io.imread('../images/contacts/big.jpg')  # [15:129, 29:146]
    # return io.imread('../images/contacts/3contacts.jpg')  # [15:129, 29:146]


def drawContour(im, contour, color):
    # # contour has shape (N, 1, 2) -> reshape to (N, 2)
    # contour = contour.reshape(contour.shape[0], 2)
    # rows, cols = contour[:, 1], contour[:, 0]
    # im[rows, cols] = color

    cv2.drawContours(im, [contour], 0, color, thickness=3)


def main():
    image_rgb = imread()
    image_rgb = cv2.blur(image_rgb, ksize=(3, 3))
    #image_rgb = cv2.medianBlur(image_rgb, ksize=5)
    im_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) # image_rgb[:, :, 1]



    # edges = cv2.Canny(im_gray, 255 * 0.55, 255 * 0.8, edges=None, apertureSize=3, L2gradient=True)
    edges = cv2.Canny(im_gray, 255 * 0.00, 255 * 1.00, edges=None, apertureSize=3, L2gradient=False)

    _, contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    im = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    im = np.zeros_like(im)
    colors = utils.make_n_colors(len(contours))
    for i, contour in enumerate(contours):
        drawContour(im, contour, colors[i])

    # f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    # ax1.imshow(edges, cmap=plt.cm.gray)
    # ax2.imshow(im)
    #
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    f, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.imshow(im)
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.show()


main()
