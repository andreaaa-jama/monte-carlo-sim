from mcsim.pricers import bs_call, mc_call

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1


def test_bs_call_known_value():
    assert abs(bs_call(S0, K, r, sigma, T) - 10.4506) < 1e-3


def test_mc_call_matches_black_scholes():
    price, std_error, _ = mc_call(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert abs(price - bs_call(S0, K, r, sigma, T)) < 3 * std_error


def test_bs_price_inside_confidence_interval():
    _, _, (low, high) = mc_call(S0, K, r, sigma, T, n_paths=200_000, seed=1)
    assert low < bs_call(S0, K, r, sigma, T) < high