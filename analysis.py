import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
import symnmfmodule
from kmeans import kmeans
from symnmf import ERROR_MESSAGE, initialize_h, read_points


def nmf_labels(points, k):
    """Run SymNMF and convert association rows to hard cluster labels

    Arguments: points: data points, k: number of clusters
    Returns: for each point- the index of highest association score.
    """
    w = symnmfmodule.norm(points.tolist())
    h = symnmfmodule.symnmf(initialize_h(w, k), w)
    return np.argmax(np.asarray(h), axis=1)


def valid_k(k, n):
    """Check if k is in the size boundary. Returns a boolean answer"""
    return 1 < k < n


def main():
    """Compare symnmf and kmeans with silhouette score and ARI.

    Gets user input: k, input file
    Prints silhouette scores and ARI of the methods, or prints and error.
    """
    try:
        if len(sys.argv) != 3:
            raise ValueError("wrong number of arguments")
        k = int(sys.argv[1])
        points = read_points(sys.argv[2])
        if not valid_k(k, len(points)):
            print("Incorrect number of clusters!")
            return
        nmf = nmf_labels(points, k)
        km = np.asarray(kmeans(points, k))
        print(f"nmf: {silhouette_score(points, nmf):.4f}")
        print(f"kmeans: {silhouette_score(points, km):.4f}")
        print(f"ari: {adjusted_rand_score(nmf, km):.4f}")
    except Exception:
        print(ERROR_MESSAGE)


if __name__ == "__main__":
    main()
