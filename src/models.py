import numpy as np
from sklearn.metrics import pairwise_distances
from numpy.random import default_rng


def balanced_kmeans(X, k, gamma, max_iter=100, random_state=None, verbose=False):
    """
    Balanced k-means with a *minimum* cluster size constraint:
        |C_j| >= floor(gamma * n) for all j.

    Heuristic algorithm:
      1. Initialize centers with k-means++ style seeding.
      2. Assignment step: assign each point to nearest center.
      3. Repair step: if any cluster is below min_size, steal points from larger clusters
         that cause the smallest increase in distortion.
      4. Update centers and repeat until convergence or max_iter.

    Returns
    -------
    labels : (n,) int array
    centers : (k, d) array
    inertia : float (sum of squared distances)
    n_iter : int
    converged : bool
    """
    rng = default_rng(random_state)
    X = np.asarray(X)
    n, d = X.shape

    min_size = int(np.floor(gamma * n))
    if min_size < 1:
        min_size = 1

    if k * min_size > n:
        raise ValueError(f"Infeasible gamma={gamma:.3g}: k * floor(gamma*n) = {k*min_size} > n={n}")

    # --- k-means++ initialization ---
    centers = np.empty((k, d), dtype=float)
    # pick first center at random
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    closest_dist_sq = pairwise_distances(X, centers[0:1], metric="sqeuclidean").ravel()

    for j in range(1, k):
        # prob proportional to distance^2
        probs = closest_dist_sq / closest_dist_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[j] = X[idx]
        new_dist_sq = pairwise_distances(X, centers[j:j+1], metric="sqeuclidean").ravel()
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    labels = np.zeros(n, dtype=int)
    prev_centers = centers.copy()
    converged = False

    for it in range(max_iter):
        # --- unconstrained assignment ---
        dists = pairwise_distances(X, centers, metric="sqeuclidean")
        labels = np.argmin(dists, axis=1)

        # --- repair step: enforce |C_j| >= min_size ---
        sizes = np.bincount(labels, minlength=k)

        # While some cluster j is too small, steal points from larger clusters
        while True:
            small_clusters = np.where(sizes < min_size)[0]
            if small_clusters.size == 0:
                break

            j = small_clusters[0]   # fix one underfull cluster at a time

            # clusters that can donate
            donor_clusters = np.where(sizes > min_size)[0]
            if donor_clusters.size == 0:
                # Can't fix; break and accept violation
                if verbose:
                    print("No donor clusters available to fix small cluster.")
                break

            best_i = None
            best_from = None
            best_cost = np.inf

            # For each donor cluster c, consider reassigning each point i in c to j
            for c in donor_clusters:
                mask = (labels == c)
                if not np.any(mask):
                    continue
                idx_c = np.where(mask)[0]

                # cost of sending each candidate point i from c to j
                # current distance to its own center c
                d_i_c = dists[idx_c, c]
                # distance to small cluster center j
                d_i_j = dists[idx_c, j]
                cost_inc = d_i_j - d_i_c

                # find the best (smallest cost) candidate in this donor cluster
                loc_best_idx = np.argmin(cost_inc)
                if cost_inc[loc_best_idx] < best_cost:
                    best_cost = cost_inc[loc_best_idx]
                    global_idx = idx_c[loc_best_idx]
                    best_i = global_idx
                    best_from = c

            if best_i is None:
                if verbose:
                    print("No feasible reassignments found.")
                break

            # Perform move: best_from -> j
            labels[best_i] = j
            sizes[best_from] -= 1
            sizes[j] += 1

        # Now all clusters respect the min_size constraint (or we gave up).
        # Update centers
        for c in range(k):
            pts_c = X[labels == c]
            if pts_c.shape[0] > 0:
                centers[c] = pts_c.mean(axis=0)
            else:
                # If somehow a cluster became empty, re-seed it randomly
                centers[c] = X[rng.integers(0, n)]

        # convergence check
        shift = np.linalg.norm(centers - prev_centers)
        if shift < 1e-6:
            converged = True
            break
        prev_centers = centers.copy()

    # compute inertia
    final_dists = pairwise_distances(X, centers, metric="sqeuclidean")
    inertia = np.sum(final_dists[np.arange(n), labels])

    return labels, centers, inertia, (it + 1), converged

def kmeans_plus_plus_init(X, k, rng=None):
    rng = rng or default_rng()
    n, d = X.shape
    centers = np.empty((k, d), dtype=float)

    # choose first center uniformly
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    # remaining centers
    closest_dist_sq = pairwise_distances(X, centers[0:1], metric="sqeuclidean").ravel()
    for j in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[j] = X[idx]
        new_dist_sq = pairwise_distances(X, centers[j:j+1], metric="sqeuclidean").ravel()
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers
