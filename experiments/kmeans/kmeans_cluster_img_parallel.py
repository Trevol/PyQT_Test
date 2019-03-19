from skimage import io, color
import numpy as np
from time import time
from kmeans.kmeans_utils import cluster_img
from multiprocessing import Pool

if __name__ == '__main__':
    n_colors = 16

    img_rgb = io.imread("../images/contacts/3contacts.jpg")
    img_lab = color.rgb2lab(img_rgb)  # [:, :, 0:3]
    img_hsv = color.rgb2hsv(img_rgb)  # [:, :, 0:3]

    pool = Pool()

    t0 = time()
    clustered_img_rgb, clustered_img_lab, clustered_img_hsv = \
        pool.starmap(cluster_img, [(img_lab, n_colors), (img_rgb, n_colors), (img_hsv, n_colors)])
    print(f'Done. {time()-t0} sec')

    t0 = time()
    clustered_img_rgb, clustered_img_lab, clustered_img_hsv = \
        pool.starmap(cluster_img, [(img_lab, n_colors), (img_rgb, n_colors), (img_hsv, n_colors)])
    print(f'Done. {time()-t0} sec')

    img_to_show = np.vstack((img_rgb, clustered_img_rgb, clustered_img_lab, clustered_img_hsv))
    io.imshow(img_to_show)
    io.show()
