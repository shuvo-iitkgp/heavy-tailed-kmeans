import numpy as np
from numpy.random import default_rng
def generate_two_gaussians(n=2000, delta=4.0, seed=0):
    """
    Two symmetric 2D Gaussians at (+delta, 0) and (-delta, 0).
    """
    rng = default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    mu1 = np.array([delta, 0.0])
    mu2 = np.array([-delta, 0.0])

    X1 = rng.normal(loc=mu1, scale=1.0, size=(n1, 2))
    X2 = rng.normal(loc=mu2, scale=1.0, size=(n2, 2))

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    centers_true = np.stack([mu1, mu2], axis=0)
    return X, y, centers_true


def generate_two_paretos(n=2000, alpha=2.0, delta=4.0, xm=1.0, seed=0):
    """
    Very simple 'two cluster' Pareto: sample Pareto radius,
    then center around (+delta, 0) and (-delta, 0) on the x-axis.
    """
    rng = default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    def pareto_r(size):
        U = rng.random(size=size)
        return xm * (U ** (-1.0 / alpha))

    # Unit directions on circle
    theta1 = rng.uniform(0.0, 2*np.pi, size=n1)
    theta2 = rng.uniform(0.0, 2*np.pi, size=n2)
    U1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    U2 = np.stack([np.cos(theta2), np.sin(theta2)], axis=1)

    R1 = pareto_r(n1)
    R2 = pareto_r(n2)

    X1 = U1 * R1[:, None] + np.array([delta, 0.0])
    X2 = U2 * R2[:, None] + np.array([-delta, 0.0])

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])
    centers_true = np.stack([np.array([delta, 0.0]), np.array([-delta, 0.0])], axis=0)
    return X, y, centers_true
def sample_gaussian_mixture(n, centers, sigma=1.0, rng=None):
    """
    2-component Gaussian mixture with equal weights, spherical cov sigma^2 I.
    centers: (2, d) array of true centers.
    Returns X, y_true.
    """
    rng = rng or default_rng()
    k, d = centers.shape
    assert k == 2, "This helper assumes k=2."
    n_per = n // 2
    n_extra = n - 2 * n_per

    X_list = []
    y_list = []
    for j, c in enumerate(centers):
        m = n_per + (1 if j < n_extra else 0)
        Xj = rng.normal(loc=c, scale=sigma, size=(m, d))
        yj = np.full(m, j, dtype=int)
        X_list.append(Xj)
        y_list.append(yj)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def sample_pareto_mixture(n, centers, alpha=1.5, rng=None):
    """
    Heavy-tailed 2D mixture around the same centers.
    We sample a radial Pareto(α) perturbation with random direction.
    centers: (2, d) array.
    Returns X, y_true.
    """
    rng = rng or default_rng()
    k, d = centers.shape
    assert k == 2 and d == 2, "This helper assumes 2 clusters in 2D."

    n_per = n // 2
    n_extra = n - 2 * n_per

    X_list = []
    y_list = []

    # Standard Pareto with xm = 1, shape alpha: r ~ xm * (1 + U)^(-1/alpha) etc.
    for j, c in enumerate(centers):
        m = n_per + (1 if j < n_extra else 0)

        # sample radius ~ Pareto(alpha)
        u = rng.uniform(size=m)
        r = (1 - u) ** (-1.0 / alpha)  # xm = 1

        # random directions on unit circle
        theta = rng.uniform(0, 2 * np.pi, size=m)
        dir_vec = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        Xj = c + (r[:, None] * dir_vec)
        yj = np.full(m, j, dtype=int)

        X_list.append(Xj)
        y_list.append(yj)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

