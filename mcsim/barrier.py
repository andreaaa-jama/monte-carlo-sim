import numpy as np

from mcsim.paths import simulate_gbm
from mcsim.pricers import bs_call

BGK_CONSTANT = 0.5826  # -zeta(1/2) / sqrt(2*pi), Broadie-Glasserman-Kou (1997)


def down_and_out_call(S0, K, H, r, sigma, T):
    """Exact price of a down-and-out call, barrier watched continuously (H <= K)."""
    if S0 <= H:
        return 0.0
    exponent = 2 * r / sigma**2 - 1
    return bs_call(S0, K, r, sigma, T) - (H / S0) ** exponent * bs_call(H**2 / S0, K, r, sigma, T)


def down_and_out_call_discrete(S0, K, H, r, sigma, T, n_steps):
    """Approximate price when the barrier is only checked at n_steps dates.

    Broadie-Glasserman-Kou correction: shift the barrier down by
    exp(-0.5826 * sigma * sqrt(dt)) and use the continuous formula.
    """
    dt = T / n_steps
    H_shifted = H * np.exp(-BGK_CONSTANT * sigma * np.sqrt(dt))
    return down_and_out_call(S0, K, H_shifted, r, sigma, T)


def mc_down_and_out_call(S0, K, H, r, sigma, T, n_steps, n_paths, seed=None, batch_size=20_000):
    """Monte Carlo price of a down-and-out call, barrier checked at each time step.

    Paths are simulated in batches to keep memory use small.
    Returns (price, standard_error, (ci_low, ci_high)).
    """
    seeds = np.random.SeedSequence(seed).spawn(int(np.ceil(n_paths / batch_size)))
    discount = np.exp(-r * T)
    payoffs = []

    remaining = n_paths
    for batch_seed in seeds:
        n = min(batch_size, remaining)
        paths = simulate_gbm(S0, r, sigma, T, n_steps, n, seed=batch_seed)
        alive = paths.min(axis=1) > H
        payoffs.append(discount * np.maximum(paths[:, -1] - K, 0.0) * alive)
        remaining -= n

    payoffs = np.concatenate(payoffs)
    price = payoffs.mean()
    std_error = payoffs.std(ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96 * std_error, price + 1.96 * std_error)
    return price, std_error, ci