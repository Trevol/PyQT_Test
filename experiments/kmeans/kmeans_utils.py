import utils
from sklearn.cluster import KMeans
import numpy as np


def cluster_img(img, n_clusters):
    if len(img.shape) == 3:
        h, w, d = img.shape
    else:
        (h, w), d = img.shape, 1
    # take ab values
    img_array = np.reshape(img, (h * w, d))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_array)
    labels = kmeans.labels_
    img_labels = np.reshape(labels, (h, w))

    lut = utils.make_n_colors(n_clusters)
    img_lut_labels = np.zeros((h, w, 3), dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            img_lut_labels[r, c] = lut[img_labels[r, c]]

    return img_lut_labels
