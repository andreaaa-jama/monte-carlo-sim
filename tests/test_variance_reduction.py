import numpy as np

from mcsim.pricers import bs_call, mc_call
from mcsim.variance_reduction import mc_call_antithetic

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1


def test_antithetic_matches_black_scholes():
    price, std_error, _ = mc_call_antithetic(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert abs(price - bs_call(S0, K, r, sigma, T)) < 3 * std_error


def test_antithetic_has_smaller_standard_error():
    _, se_plain, _ = mc_call(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    _, se_anti, _ = mc_call_antithetic(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert se_anti < se_plain


def test_antithetic_confidence_interval_contains_exact_price():
    _, _, (low, high) = mc_call_antithetic(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert low < bs_call(S0, K, r, sigma, T) < high

def test_control_variate_matches_black_scholes():
    from mcsim.variance_reduction import mc_call_control_variate
    price, std_error, _, _ = mc_call_control_variate(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert abs(price - bs_call(S0, K, r, sigma, T)) < 3 * std_error


def test_control_variate_beats_plain_and_antithetic():
    from mcsim.variance_reduction import mc_call_control_variate
    _, se_plain, _ = mc_call(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    _, se_anti, _ = mc_call_antithetic(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    _, se_cv, _, _ = mc_call_control_variate(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert se_cv < se_anti < se_plain