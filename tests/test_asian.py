import numpy as np

from mcsim.asian import geometric_asian_call, mc_asian_call
from mcsim.paths import simulate_gbm
from mcsim.pricers import bs_call

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
n_steps = 52


def test_geometric_formula_matches_simulation():
    n_paths = 200_000
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed=3)
    G = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    payoffs = np.exp(-r * T) * np.maximum(G - K, 0.0)
    std_error = payoffs.std(ddof=1) / np.sqrt(n_paths)
    assert abs(payoffs.mean() - geometric_asian_call(S0, K, r, sigma, T, n_steps)) < 3 * std_error


def test_one_fixing_reduces_to_black_scholes():
    assert abs(geometric_asian_call(S0, K, r, sigma, T, 1) - bs_call(S0, K, r, sigma, T)) < 1e-10


def test_control_variate_agrees_with_plain_mc():
    p_plain, se_plain, _ = mc_asian_call(S0, K, r, sigma, T, n_steps, 200_000, seed=1,
                                         control_variate=False)
    p_cv, se_cv, _ = mc_asian_call(S0, K, r, sigma, T, n_steps, 200_000, seed=2)
    assert abs(p_plain - p_cv) < 3 * np.sqrt(se_plain**2 + se_cv**2)


def test_control_variate_reduces_error_a_lot():
    _, se_plain, _ = mc_asian_call(S0, K, r, sigma, T, n_steps, 100_000, seed=1,
                                   control_variate=False)
    _, se_cv, _ = mc_asian_call(S0, K, r, sigma, T, n_steps, 100_000, seed=1)
    assert se_cv < se_plain / 10


def test_price_ordering():
    arithmetic, _, _ = mc_asian_call(S0, K, r, sigma, T, n_steps, 100_000, seed=1)
    geometric = geometric_asian_call(S0, K, r, sigma, T, n_steps)
    european = bs_call(S0, K, r, sigma, T)
    assert geometric < arithmetic < european