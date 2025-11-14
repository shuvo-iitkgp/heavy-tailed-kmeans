import os
import sys
import time
# Add the project root (one level up from this file) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualization import plot_metric_vs_gamma
from src.utils import run_gamma_sweep
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,              # INFO | DEBUG | WARNING | ERROR
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./experiments/experiment.log"),  # writes to file
        logging.StreamHandler()                 # prints to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting experiment...")


# Full gamma sweep
# Example: n = 200000, k = 2
start = time.time()
df_gauss = run_gamma_sweep(dist_type="gaussian", n=90_000, k=2, n_reps=20)
logger.info("Gamma sweep complete in %.2f seconds.", time.time() - start)
start = time.time()
df_pareto = run_gamma_sweep(dist_type="pareto",  n=90_000, k=2, n_reps=20)
logger.info("Gamma sweep complete in %.2f seconds.", time.time() - start)


results = pd.concat([df_gauss, df_pareto], ignore_index=True)
logger.info("Writing CSV...")
results.to_csv("./results/results_sharpness_gamma.csv", index=False)

plot_metric_vs_gamma(results, "ARI", "Balanced k-means: ARI vs γ_n", save_path="./images/ARI_vs_gamma.png")
plot_metric_vs_gamma(results, "Hausdorff", "Balanced k-means: Hausdorff vs γ_n", save_path="./images/Hausdorff_vs_gamma.png")
plot_metric_vs_gamma(results, "Distortion", "Balanced k-means: Distortion vs γ_n", save_path="./images/Distortion_vs_gamma.png")
logger.info("All done.")
