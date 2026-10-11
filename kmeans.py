import numpy as np

EPSILON = 1e-4
MAX_ITER = 300


def _closest_centroid(point, centroids):
    """Return the index of the nearest centroid, breaking ties by index."""
    distances = np.sum((centroids - point) ** 2, axis=1)
    return int(np.argmin(distances))


def kmeans(points, k, max_iter=MAX_ITER, epsilon=EPSILON):
    """Return labels for points using k first-point initial centroids.

    Empty clusters preserve their old centroid; stop at the movement
    threshold or max_iter, then assign final labels.
    """
    data = np.asarray(points, dtype=float)
    centroids = data[:k].copy()
    labels = np.zeros(len(data), dtype=int)
    for _ in range(max_iter):
        labels = np.array([_closest_centroid(point, centroids) for point in data])
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if len(members) > 0:
                new_centroids[cluster] = np.mean(members, axis=0)
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < epsilon:
            centroids = new_centroids
            break
        centroids = new_centroids
    labels = np.array([_closest_centroid(point, centroids) for point in data])
    return labels.tolist()
