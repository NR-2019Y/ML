import numpy as np
from scipy.spatial.distance import cdist


class MyDBSCAN(object):
    def __init__(self, *, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        n_samples, n_features = X.shape
        dist_mat = cdist(X, X)
        is_neighbors = dist_mat <= self.eps
        n_neighbors = np.sum(is_neighbors, axis=1)  # (n_samples, )
        core_samples_mask = n_neighbors >= self.min_samples
        self.core_sample_indices_ = np.argwhere(core_samples_mask).ravel()
        self.labels_ = np.full(n_samples, -1, dtype=np.int64)

        neighbors_indices_cache = [None] * n_samples

        def get_neighbors_indices(index):
            if neighbors_indices_cache[index] is None:
                neighbors_indices_cache[index] = np.argwhere(is_neighbors[index]).ravel()
            return neighbors_indices_cache[index]

        q = []
        current_label = 0
        for i in self.core_sample_indices_:
            if self.labels_[i] == -1:
                # BFS
                self.labels_[i] = current_label
                q.append(i)
                while q:
                    idx = q.pop(0)
                    for u in get_neighbors_indices(idx):
                        if self.labels_[u] == -1:
                            self.labels_[u] = current_label
                            q.append(u)
                current_label += 1
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_
