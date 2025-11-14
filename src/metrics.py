from sklearn.metrics import adjusted_rand_score, pairwise_distances
import numpy as np

def hausdorff_distance(centers_est, centers_true):
    """
    Symmetric Hausdorff distance between two finite point sets in R^d.
    """
    centers_est = np.asarray(centers_est)
    centers_true = np.asarray(centers_true)

    dmat = pairwise_distances(centers_est, centers_true, metric="euclidean")

    # directed distances
    d1 = dmat.min(axis=1).max()  # max over est of min true
    d2 = dmat.min(axis=0).max()  # max over true of min est

    return max(d1, d2)

def distortion(X, centers, labels):
    dists_sq = np.sum((X - centers[labels]) ** 2, axis=1)
    return float(dists_sq.sum())


def clustering_metrics(X, y_true, centers_true, labels_est, centers_est, inertia):
    """
    Convenience wrapper returning a dict with ARI, Hausdorff, and distortion.
    """
    ari = adjusted_rand_score(y_true, labels_est)
    hd = hausdorff_distance(centers_est, centers_true)
    distortion_var = inertia / X.shape[0]   # average SSE per point (optional)

    return {
        "ARI": ari,
        "Hausdorff": hd,
        "Distortion": distortion_var,
        "SSE": inertia
    }

