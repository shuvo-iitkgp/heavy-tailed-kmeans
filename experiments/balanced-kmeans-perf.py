import os
import sys
import numpy as np 
from sklearn.cluster import KMeans
from numpy.random import default_rng
import time
# Add the project root (one level up from this file) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from src import *


logging.basicConfig(
    level=logging.INFO,              # INFO | DEBUG | WARNING | ERROR
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./experiments/experiment-k-means-vs-balanced-kmeans.log"),  # writes to file
        logging.StreamHandler()                 # prints to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting experiment...")

######################variable controls########################
n=90000
n_runs=20
alpha=1.2
gamma=0.1
sigma_gauss=1.0
seed=0

######################variable controls########################

rng = default_rng(seed)
centers_true = np.array([[-3.0, 0.0],
                             [ 3.0, 0.0]])

records = []

for run in range(n_runs):
    rs = int(rng.integers(0, 10_000_000))

    # --- Gaussian mixture ---
    Xg, y_true_g = sample_gaussian_mixture(n, centers_true,
                                            sigma=sigma_gauss,
                                            rng=default_rng(rs))

    # standard k-means
    km = KMeans(n_clusters=2, n_init=10, random_state=rs)
    labels_km_g = km.fit_predict(Xg)
    centers_km_g = km.cluster_centers_
    ari_km_g = adjusted_rand_score(y_true_g, labels_km_g)
    haus_km_g = hausdorff_distance(centers_km_g, centers_true)
    dist_km_g = distortion(Xg, centers_km_g, labels_km_g)

    records.append({
        "dist": "Gaussian",
        "algo": "KMeans",
        "ARI": ari_km_g,
        "Hausdorff": haus_km_g,
        "Distortion": dist_km_g,
        "Failure": float(ari_km_g < 0.8),
    })

    # balanced k-means
    labels_bal_g, centers_bal_g, inertia_bal_g, _, _ = balanced_kmeans(
        Xg, k=2, gamma=gamma, max_iter=100, random_state=rs
    )
    ari_bal_g = adjusted_rand_score(y_true_g, labels_bal_g)
    haus_bal_g = hausdorff_distance(centers_bal_g, centers_true)
    dist_bal_g = distortion(Xg, centers_bal_g, labels_bal_g)

    records.append({
        "dist": "Gaussian",
        "algo": "Balanced",
        "ARI": ari_bal_g,
        "Hausdorff": haus_bal_g,
        "Distortion": dist_bal_g,
        "Failure": float(ari_bal_g < 0.8),
    })

    # --- Pareto mixture ---
    Xp, y_true_p = sample_pareto_mixture(n, centers_true,
                                            alpha=alpha,
                                            rng=default_rng(rs + 1))

    # standard k-means
    km_p = KMeans(n_clusters=2, n_init=10, random_state=rs + 1)
    labels_km_p = km_p.fit_predict(Xp)
    centers_km_p = km_p.cluster_centers_
    ari_km_p = adjusted_rand_score(y_true_p, labels_km_p)
    haus_km_p = hausdorff_distance(centers_km_p, centers_true)
    dist_km_p = distortion(Xp, centers_km_p, labels_km_p)

    records.append({
        "dist": "Pareto",
        "algo": "KMeans",
        "ARI": ari_km_p,
        "Hausdorff": haus_km_p,
        "Distortion": dist_km_p,
        "Failure": float(ari_km_p < 0.8),
    })

    # balanced k-means
    labels_bal_p, centers_bal_p, inertia_bal_p, _, _ = balanced_kmeans(
        Xp, k=2, gamma=gamma, max_iter=100, random_state=rs + 1
    )
    ari_bal_p = adjusted_rand_score(y_true_p, labels_bal_p)
    haus_bal_p = hausdorff_distance(centers_bal_p, centers_true)
    dist_bal_p = distortion(Xp, centers_bal_p, labels_bal_p)

    records.append({
        "dist": "Pareto",
        "algo": "Balanced",
        "ARI": ari_bal_p,
        "Hausdorff": haus_bal_p,
        "Distortion": dist_bal_p,
        "Failure": float(ari_bal_p < 0.8),
    })

df = pd.DataFrame.from_records(records)


logger.info("Completed the df calculation")

plot_boxplots(df, save_path="./experiments/balanced-kmeans-perf-boxplots-alpha-1_2.png")

logger.info("Completed the box plots calculation")

print_summary_table(df, save_path="./experiments/balanced-kmeans-perf-summary-table-alpha-1_2.png")
logger.info("saved the summary table")

center_convergence_plot(alpha=1.2, gamma=0.1, path_save="./experiments/center-convergence-alpha-1_2.png")

logger.info("All done.")
