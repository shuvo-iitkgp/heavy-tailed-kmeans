import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from src.models import balanced_kmeans
from src.metrics import hausdorff_distance
from src.data_generation import sample_gaussian_mixture, sample_pareto_mixture

from numpy.random import default_rng

def plot_metric_vs_gamma(df, metric, title, save_path="./images/mock.png"):
    """
    Plot mean ± std of a metric as a function of gamma,
    separately for each distribution type.
    """
    plt.figure(figsize=(6, 4))
    for dist_type, group in df.groupby("dist_type"):
        g_sorted = group.sort_values("gamma")
        gamma_unique = np.sort(g_sorted["gamma"].unique())
        means = []
        stds = []
        for g in gamma_unique:
            vals = g_sorted.loc[g_sorted["gamma"] == g, metric].values
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means)
        stds = np.array(stds)
        plt.errorbar(gamma_unique, means, yerr=stds, label=dist_type, capsize=3)

    plt.xscale("log")
    plt.xlabel(r"$\gamma_n$")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    # plt.savefig(save_path)

def plot_boxplots(df, path_save = "./images/balanced-kmeans-convergence.png"):
    # Create a combined label for the 4 combos
    df = df.copy()
    df["Group"] = df["dist"] + " - " + df["algo"]

    groups = df["Group"].unique()
    order = ["Gaussian - KMeans",
             "Gaussian - Balanced",
             "Pareto - KMeans",
             "Pareto - Balanced"]
    groups = [g for g in order if g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ARI boxplot
    data_ari = [df.loc[df["Group"] == g, "ARI"] for g in groups]
    axes[0].boxplot(data_ari, labels=groups, showmeans=True)
    axes[0].set_title("ARI distribution")
    axes[0].set_ylabel("ARI")
    axes[0].set_xticklabels(groups, rotation=20)

    # Hausdorff boxplot
    data_haus = [df.loc[df["Group"] == g, "Hausdorff"] for g in groups]
    axes[1].boxplot(data_haus, labels=groups, showmeans=True)
    axes[1].set_title("Hausdorff distance to true centers")
    axes[1].set_ylabel("Hausdorff distance")
    axes[1].set_xticklabels(groups, rotation=20)

    plt.tight_layout()
    plt.savefig(path_save)
    # plt.show()


def print_summary_table(df, save_path = "./results/summary-table.png"):
    agg = df.groupby(["dist", "algo"]).agg(
        ARI_mean=("ARI", "mean"),
        ARI_std=("ARI", "std"),
        Hausdorff_mean=("Hausdorff", "mean"),
        Hausdorff_std=("Hausdorff", "std"),
        Distortion_mean=("Distortion", "mean"),
        Distortion_std=("Distortion", "std"),
        Failure_rate=("Failure", "mean"),
    )

    # pretty print as mean ± std
    table = pd.DataFrame(index=agg.index)
    for metric in ["ARI", "Hausdorff", "Distortion"]:
        m = agg[f"{metric}_mean"]
        s = agg[f"{metric}_std"]
        table[metric] = m.map("{:.3f}".format) + " ± " + s.map("{:.3f}".format)

    table["Failure prob (ARI<0.8)"] = agg["Failure_rate"].map("{:.3f}".format)
    table.to_csv(save_path , index = None)
    print("\nSummary (mean ± std over runs):\n")
    print(table.to_string())

def center_convergence_plot(alpha=1.5, gamma=0.1,
                            n_grid=None, seed=123, path_save = "./images/center-convergence.png"):
    if n_grid is None:
        n_grid = np.linspace(200, 10_000, 40, dtype=int)

    rng = default_rng(seed)
    centers_true = np.array([[-3.0, 0.0],
                             [ 3.0, 0.0]])

    err_gauss_km = []
    err_gauss_bal = []
    err_par_km = []
    err_par_bal = []

    for n in n_grid:
        rs = int(rng.integers(0, 10_000_000))

        # Gaussian
        Xg, _ = sample_gaussian_mixture(n, centers_true, sigma=1.0,
                                        rng=default_rng(rs))
        km = KMeans(n_clusters=2, n_init=10, random_state=rs)
        labels_km_g = km.fit_predict(Xg)
        centers_km_g = km.cluster_centers_
        err_gauss_km.append(hausdorff_distance(centers_km_g, centers_true))

        labels_bal_g, centers_bal_g, _, _, _ = balanced_kmeans(
            Xg, k=2, gamma=gamma, random_state=rs + 1
        )
        err_gauss_bal.append(hausdorff_distance(centers_bal_g, centers_true))

        # Pareto
        Xp, _ = sample_pareto_mixture(n, centers_true, alpha=alpha,
                                      rng=default_rng(rs + 2))
        km_p = KMeans(n_clusters=2, n_init=10, random_state=rs + 2)
        labels_km_p = km_p.fit_predict(Xp)
        centers_km_p = km_p.cluster_centers_
        err_par_km.append(hausdorff_distance(centers_km_p, centers_true))

        labels_bal_p, centers_bal_p, _, _, _ = balanced_kmeans(
            Xp, k=2, gamma=gamma, random_state=rs + 3
        )
        err_par_bal.append(hausdorff_distance(centers_bal_p, centers_true))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].plot(n_grid, err_gauss_km, label="KMeans")
    axes[0].plot(n_grid, err_gauss_bal, label="Balanced")
    axes[0].set_title("Gaussian mixture")
    axes[0].set_xlabel("n (samples)")
    axes[0].set_ylabel("Hausdorff error to true centers")
    axes[0].legend()

    axes[1].plot(n_grid, err_par_km, label="KMeans")
    axes[1].plot(n_grid, err_par_bal, label="Balanced")
    axes[1].set_title(f"Pareto mixture (alpha={alpha})")
    axes[1].set_xlabel("n (samples)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path_save)
    # plt.show()

