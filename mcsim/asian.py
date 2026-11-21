import numpy as np
from scipy.stats import norm

from mcsim.paths import simulate_gbm


def geometric_asian_call(S0, K, r, sigma, T, n_steps):
    n = n_steps
    mean_log = np.log(S0) + (r - sigma**2 / 2) * T * (n + 1) / (2 * n)
    var_log = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)

    d1 = (mean_log - np.log(K) + var_log) / np.sqrt(var_log)
    d2 = d1 - np.sqrt(var_log)
    expected_G = np.exp(mean_log + var_log / 2)
    return np.exp(-r * T) * (expected_G * norm.cdf(d1) - K * norm.cdf(d2))


def mc_asian_call(S0, K, r, sigma, T, n_steps, n_paths, seed=None, control_variate=True):
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    prices = paths[:, 1:]                      # drop S0: average over the n_steps fixings
    discount = np.exp(-r * T)

    arithmetic_avg = prices.mean(axis=1)
    Y = discount * np.maximum(arithmetic_avg - K, 0.0)

    if control_variate:
        geometric_avg = np.exp(np.log(prices).mean(axis=1))
        X = discount * np.maximum(geometric_avg - K, 0.0)
        X_exact = geometric_asian_call(S0, K, r, sigma, T, n_steps)

        cov = np.cov(Y, X, ddof=1)
        beta = cov[0, 1] / cov[1, 1]
        Y = Y - beta * (X - X_exact)

    price = Y.mean()
    std_error = Y.std(ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci