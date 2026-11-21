from mcsim.barrier import down_and_out_call, down_and_out_call_discrete, mc_down_and_out_call
from mcsim.pricers import bs_call

S0, K, H, r, sigma, T = 100, 100, 90, 0.05, 0.2, 1


def test_far_barrier_gives_vanilla_call():
    assert abs(down_and_out_call(S0, K, 1e-6, r, sigma, T) - bs_call(S0, K, r, sigma, T)) < 1e-8


def test_knocked_out_from_the_start_is_worth_zero():
    assert down_and_out_call(85, K, H, r, sigma, T) == 0.0


def test_barrier_option_is_cheaper_than_vanilla():
    assert down_and_out_call(S0, K, H, r, sigma, T) < bs_call(S0, K, r, sigma, T)


def test_mc_matches_discrete_monitoring_correction():
    price, std_error, _ = mc_down_and_out_call(S0, K, H, r, sigma, T, 52, 200_000, seed=1)
    assert abs(price - down_and_out_call_discrete(S0, K, H, r, sigma, T, 52)) < 3 * std_error


def test_discrete_monitoring_is_biased_upward():
    price, std_error, _ = mc_down_and_out_call(S0, K, H, r, sigma, T, 52, 200_000, seed=1)
    assert price - down_and_out_call(S0, K, H, r, sigma, T) > 5 * std_error
    