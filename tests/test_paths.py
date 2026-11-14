import numpy as np
from mcsim.paths import simulate_gbm

S0, mu, sigma, T = 100, 0.05, 0.2, 1
n_steps, n_paths = 252, 100_000


def get_paths():
    return simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=1)


def test_shape():
    assert get_paths().shape == (n_paths, n_steps + 1)


def test_starts_at_S0():
    assert np.all(get_paths()[:, 0] == S0)


def test_mean_final_price():
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_paths = 100_000
    paths = simulate_gbm(S0, mu, sigma, T, 252, n_paths, seed=1)
    final = paths[:, -1]

    expected = S0 * np.exp(mu * T)
    std_error = final.std() / np.sqrt(n_paths)
    assert abs(final.mean() - expected) < 3 * std_error


def test_log_return_stats():
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_paths = 100_000
    paths = simulate_gbm(S0, mu, sigma, T, 252, n_paths, seed=1)
    log_ret = np.log(paths[:, -1] / S0)

    expected_mean = (mu - sigma**2 / 2) * T
    expected_std = sigma * np.sqrt(T)
    std_error = log_ret.std() / np.sqrt(n_paths)
    assert abs(log_ret.mean() - expected_mean) < 3 * std_error
    assert abs(log_ret.std() - expected_std) < 0.002