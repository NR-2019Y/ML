import numpy as np
from scipy.spatial.distance import cdist


def kmeans_plus_plus(X: np.ndarray, k: int):
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features), dtype=np.float64)
    centroids[0] = X[np.random.choice(n_samples, 1)[0]]
    for i in range(1, k):
        dist = cdist(X, centroids[:i, :])
        center_index = dist.min(axis=1).argmax()
        centroids[i] = X[center_index]
    return centroids


class MyKMeans(object):
    def __init__(self, *, n_clusters=8):
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        k = self.n_clusters
        # K-Means++算法得到初始化聚类中心
        centroids = kmeans_plus_plus(X, k)

        labels = np.full(n_samples, -1)
        while True:
            # new_labels = np.empty_like(labels)
            # for i in range(n_samples):
            #     dist = np.linalg.norm(centroids - X[i], ord=2, axis=1)
            #     new_labels[i] = dist.argmin()
            dist = cdist(X, centroids)
            new_labels = dist.argmin(axis=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            for i in range(k):
                centroids[i] = np.mean(X[labels == i, :], axis=0)

        sse_val = 0
        for i in range(k):
            sse_val += np.sum((X[labels == i, :] - centroids[i]) ** 2)
        self.inertia_ = sse_val
        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).labels_

    def predict(self, X):
        dist = cdist(X, self.cluster_centers_)
        return dist.argmin(axis=1)
