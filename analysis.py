import sys
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
import symnmfmodule as symnmf
from kmeans import kmeans

np.random.seed(1234)

ERROR_MESSAGE = "An Error Has Occurred"


def read_points(file_name):
    """Read a comma-separated dataset."""
    return np.loadtxt(file_name, delimiter=",", ndmin=2)


def initialize_h(w, k):
    """Initialize H according to the SymNMF specification."""
    m = float(np.mean(w))
    upper = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper, size=(len(w), k)).tolist()


def nmf_labels(points, k):
    """Run SymNMF and convert association rows to hard cluster labels."""
    w = symnmf.norm(points.tolist())
    h = symnmf.symnmf(initialize_h(w, k), w)
    return np.argmax(np.asarray(h), axis=1)


def main():
    """Compare SymNMF and K-means using silhouette score and ARI."""
    try:
        if len(sys.argv) != 3:
            raise ValueError("wrong number of arguments")
        k = int(sys.argv[1])
        points = read_points(sys.argv[2])
        nmf = nmf_labels(points, k)
        km = np.asarray(kmeans(points, k))
        print(f"nmf: {silhouette_score(points, nmf):.4f}")
        print(f"kmeans: {silhouette_score(points, km):.4f}")
        print(f"ari: {adjusted_rand_score(nmf, km):.4f}")
    except Exception:
        print(ERROR_MESSAGE)


if __name__ == "__main__":
    main()
