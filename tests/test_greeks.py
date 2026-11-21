from mcsim.greeks import mc_delta, mc_delta_pathwise, mc_vega
from mcsim.pricers import bs_delta, bs_vega

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
N = 200_000


def test_bs_greeks_known_values():
    assert abs(bs_delta(S0, K, r, sigma, T) - 0.6368) < 1e-4
    assert abs(bs_vega(S0, K, r, sigma, T) - 37.524) < 1e-3


def test_delta_crn_matches_black_scholes():
    delta, se = mc_delta(S0, K, r, sigma, T, N, seed=1)
    assert abs(delta - bs_delta(S0, K, r, sigma, T)) < 3 * se + 1e-3


def test_vega_crn_matches_black_scholes():
    vega, se = mc_vega(S0, K, r, sigma, T, N, seed=1)
    assert abs(vega - bs_vega(S0, K, r, sigma, T)) < 3 * se + 1e-2


def test_pathwise_delta_matches_black_scholes():
    delta, se = mc_delta_pathwise(S0, K, r, sigma, T, N, seed=1)
    assert abs(delta - bs_delta(S0, K, r, sigma, T)) < 3 * se


def test_common_random_numbers_shrink_the_error():
    _, se_crn = mc_delta(S0, K, r, sigma, T, N, seed=1, common_random_numbers=True)
    _, se_ind = mc_delta(S0, K, r, sigma, T, N, seed=1, common_random_numbers=False)
    assert se_crn < se_ind / 10
    