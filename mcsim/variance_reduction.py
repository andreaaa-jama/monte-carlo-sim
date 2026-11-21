import numpy as np


def mc_call_antithetic(S0, K, r, sigma, T, n_paths, seed=None):
    n_pairs = n_paths // 2
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_pairs)

    drift = (r - sigma**2 / 2) * T
    vol = sigma * np.sqrt(T)
    ST_up = S0 * np.exp(drift + vol * Z)
    ST_down = S0 * np.exp(drift - vol * Z)

    payoff_up = np.maximum(ST_up - K, 0.0)
    payoff_down = np.maximum(ST_down - K, 0.0)

    pair_average = np.exp(-r * T) * (payoff_up + payoff_down) / 2

    price = pair_average.mean()
    std_error = pair_average.std(ddof=1) / np.sqrt(n_pairs)
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci

def mc_call_control_variate(S0, K, r, sigma, T, n_paths, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    ST = S0 * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)
    discount = np.exp(-r * T)

    Y = discount * np.maximum(ST - K, 0.0)   # what we want to estimate
    X = discount * ST                        # control: E[X] = S0 exactly

    cov = np.cov(Y, X, ddof=1)
    beta = cov[0, 1] / cov[1, 1]

    adjusted = Y - beta * (X - S0)

    price = adjusted.mean()
    std_error = adjusted.std(ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci, beta