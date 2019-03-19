from atom.api import observe, Int, Atom, Typed, Bool
import numpy as np
from kmeans.kmeans_utils import cluster_img
from utils import debounce


def some_func():
    print('some_func')


class ImageClustersModel(Atom):
    busy = Bool(False)
    n_clusters = Int(2)
    image = Typed(np.ndarray)
    clustered_image = Typed(np.ndarray)

    @observe('n_clusters')
    @debounce(1)
    def update_clusters(self, change):
        try:
            self.busy = True
            self.clustered_image = cluster_img(self.image, self.n_clusters)
        finally:
            self.busy = False
