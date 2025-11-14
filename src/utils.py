import pandas as pd
import math
import numpy as np
from numpy.random import default_rng
from src.metrics import clustering_metrics
from src.data_generation import generate_two_gaussians, generate_two_paretos
from src.models import balanced_kmeans


def gamma_grid(n, powers=(1, 2, 3, 4), k=2):
    """
    γ_n = (log n)^p / n for p in powers, filtered so that k * floor(γ_n n) <= n.
    """
    gammas = []
    for p in powers:
        g = (math.log(n) ** p) / n
        min_size = int(np.floor(g * n))
        if min_size < 1:
            min_size = 1
        if k * min_size <= n:
            gammas.append(g)
    return np.array(gammas)


def run_gamma_sweep(
    dist_type="gaussian",
    n=20000,
    k=2,
    n_reps=20,
    powers=(0.01, 0.25, 0.5, 0.75, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    random_state=0,
):
    """
    Run balanced k-means over a grid of γ_n values and collect ARI, Hausdorff,
    and distortion. Returns a pandas DataFrame.
    """
    rng = default_rng(random_state)
    gammas = gamma_grid(n, powers=powers, k=k)

    records = []

    for rep in range(n_reps):
        seed = int(rng.integers(0, 10**9))

        if dist_type == "gaussian":
            X, y_true, centers_true = generate_two_gaussians(n=n, seed=seed)
        elif dist_type == "pareto":
            X, y_true, centers_true = generate_two_paretos(n=n, seed=seed)
        else:
            raise ValueError(f"Unknown dist_type={dist_type}")

        for g in gammas:
            try:
                labels, centers, inertia, n_iter, converged = balanced_kmeans(
                    X, k=k, gamma=g, random_state=seed
                )
            except ValueError:
                # infeasible gamma (should be rare due to filtering)
                continue

            mets = clustering_metrics(X, y_true, centers_true, labels, centers, inertia)
            records.append({
                "dist_type": dist_type,
                "n": n,
                "rep": rep,
                "gamma": g,
                "min_size": int(np.floor(g * n)),
                "ARI": mets["ARI"],
                "Hausdorff": mets["Hausdorff"],
                "Distortion": mets["Distortion"],
                "SSE": mets["SSE"],
                "n_iter": n_iter,
                "converged": converged,
            })

    return pd.DataFrame.from_records(records)

