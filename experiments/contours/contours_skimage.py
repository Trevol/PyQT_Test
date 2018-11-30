import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw, color, img_as_ubyte
from skimage.feature import canny

from skimage import measure

def imread():
    # return data.coffee()[0:220, 160:420]
    return io.imread('../images/contacts/3contacts.jpg')#[15:129, 29:146]

def drawShape(img, coordinates, color):
    # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)
    img[coordinates[:, 0], coordinates[:, 1]] = color
    return img

def main():

    image_rgb = imread()
    edges = canny(color.rgb2gray(image_rgb), sigma=1.0, low_threshold=0.55, high_threshold=0.8)
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(edges, 0.8, fully_connected='high', positive_orientation='high')

    # Display the image and plot all contours found
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.imshow(edges, interpolation='nearest', cmap=plt.cm.gray)

    im = color.gray2rgb(img_as_ubyte(edges))
    for n, contour in enumerate(contours):
        drawShape(im, contour, (255, 0, 0))
        #ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax2.imshow(im)

    ax1.axis('image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.axis('image')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()

main()