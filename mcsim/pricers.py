import numpy as np
from scipy.stats import norm

from mcsim.paths import simulate_gbm


def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def mc_call(S0, K, r, sigma, T, n_paths, seed=None):
    paths = simulate_gbm(S0, r, sigma, T, 1, n_paths, seed=seed)
    ST = paths[:, -1]

    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci