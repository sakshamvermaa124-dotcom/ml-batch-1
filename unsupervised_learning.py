from sklearn.cluster import KMeans
import numpy as np


def perform_kmeans(data, n_clusters=2, random_state=42):
    """
    Perform K-Means clustering on numerical data.

    Args:
        data: Numerical dataset.
        n_clusters: Number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        cluster_labels: Cluster assigned to each data point.
        cluster_centers: Coordinates of cluster centers.
    """

    if data is None:
        raise ValueError("Data cannot be None.")

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("Data cannot be empty.")

    if data.ndim != 2:
        raise ValueError("Data must be a 2D numerical array.")

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain only numerical values.")

    if n_clusters < 1:
        raise ValueError("Number of clusters must be at least 1.")

    if n_clusters > len(data):
        raise ValueError(
            "Number of clusters cannot exceed the number of data points."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    cluster_labels = model.fit_predict(data)

    return cluster_labels, model.cluster_centers_


if __name__ == "__main__":
    sample_data = [
        [1, 2],
        [1, 3],
        [2, 2],
        [8, 8],
        [9, 8],
        [8, 9]
    ]

    labels, centers = perform_kmeans(sample_data, n_clusters=2)

    print("Cluster Labels:", labels)
    print("Cluster Centers:")
    print(centers)
