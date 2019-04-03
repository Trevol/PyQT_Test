import numpy as np
import sklearn.cluster
import skimage.draw
from sklearn.datasets.samples_generator import make_blobs


def main():
    # X, _ = make_blobs(n_samples=100, centers=[[1, 1], [5, 5]], cluster_std=0.3)
    # print(X)

    rr, cc = skimage.draw.circle(20, 20, 10)
    X = np.dstack([rr, cc]).reshape([-1, 2]).astype(np.float64)
    # rr, cc = skimage.draw.circle(20, 45, 10)
    # X2 = np.dstack([rr, cc]).reshape([-1, 2]).astype(np.float64)
    # X = np.concatenate([X, X2])



    ms = sklearn.cluster.MeanShift()
    ms.fit(X)

    print(np.unique(ms.labels_), ms.cluster_centers_)

    # dbscan = sklearn.cluster.DBSCAN()
    # dbscan.fit(X)
    # lb = dbscan.fit_predict(X)
    # print(np.unique(lb))


if __name__ == '__main__':
    main()
