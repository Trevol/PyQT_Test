from skimage import data, color, img_as_ubyte, io
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import matplotlib.pyplot as plt

import time

# Load picture, convert to grayscale and detect edges
#image_rgb = data.coffee()[0:220, 160:420]
image_rgb = io.imread('images/contacts/3contacts.jpg')
image_gray = color.rgb2gray(image_rgb)

t0 = time.time()
edges = canny(image_gray, sigma=2.0,
              low_threshold=0.55, high_threshold=0.8)
print(f'canny: {time.time()-t0}')
# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
t_ellipse = time.time()
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=100, max_size=120)
print(f'hough_ellipse: {time.time()-t_ellipse}')
print(f'Total: {time.time()-t0}')

result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
image_rgb[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                sharey=True)

ax1.set_title('Original picture')
ax1.imshow(image_rgb)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
