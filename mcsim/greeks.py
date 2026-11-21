import numpy as np


def _discounted_payoffs(S0, K, r, sigma, T, Z):
    """Discounted European call payoffs for given standard normal draws Z."""
    ST = S0 * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.maximum(ST - K, 0.0)


def mc_delta(S0, K, r, sigma, T, n_paths, h=0.5, seed=None, common_random_numbers=True):
    """Delta by central finite difference: (C(S0 + h) - C(S0 - h)) / (2h).

    With common_random_numbers=True, both prices use the same draws Z.
    Returns (delta, standard_error).
    """
    rng = np.random.default_rng(seed)
    Z_up = rng.standard_normal(n_paths)
    Z_down = Z_up if common_random_numbers else rng.standard_normal(n_paths)

    up = _discounted_payoffs(S0 + h, K, r, sigma, T, Z_up)
    down = _discounted_payoffs(S0 - h, K, r, sigma, T, Z_down)
    return _finite_difference(up, down, h, common_random_numbers)


def mc_vega(S0, K, r, sigma, T, n_paths, h=0.01, seed=None, common_random_numbers=True):
    """Vega by central finite difference: (C(sigma + h) - C(sigma - h)) / (2h).

    Returns (vega, standard_error).
    """
    rng = np.random.default_rng(seed)
    Z_up = rng.standard_normal(n_paths)
    Z_down = Z_up if common_random_numbers else rng.standard_normal(n_paths)

    up = _discounted_payoffs(S0, K, r, sigma + h, T, Z_up)
    down = _discounted_payoffs(S0, K, r, sigma - h, T, Z_down)
    return _finite_difference(up, down, h, common_random_numbers)


def _finite_difference(up, down, h, common_random_numbers):
    n = len(up)
    estimate = (up.mean() - down.mean()) / (2 * h)
    if common_random_numbers:
        # paired samples: the error comes from the per-path differences
        std_error = ((up - down) / (2 * h)).std(ddof=1) / np.sqrt(n)
    else:
        # independent samples: the two errors add up
        std_error = np.sqrt(up.var(ddof=1) + down.var(ddof=1)) / (2 * h) / np.sqrt(n)
    return estimate, std_error


def mc_delta_pathwise(S0, K, r, sigma, T, n_paths, seed=None):
    """Delta by the pathwise method: average of exp(-rT) * 1{S_T > K} * S_T / S0.

    No bump needed: differentiate the payoff of each path directly.
    Returns (delta, standard_error).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    samples = np.exp(-r * T) * (ST > K) * ST / S0
    return samples.mean(), samples.std(ddof=1) / np.sqrt(n_paths)