from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcsim.pricers import bs_call, mc_call
from mcsim.variance_reduction import mc_call_antithetic, mc_call_control_variate

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
exact = bs_call(S0, K, r, sigma, T)
N = 10_000
n_reps = 2_000

methods = {
    "Plain Monte Carlo": lambda s: mc_call(S0, K, r, sigma, T, N, seed=s)[0],
    "Antithetic variates": lambda s: mc_call_antithetic(S0, K, r, sigma, T, N, seed=s)[0],
    "Control variate (stock price)": lambda s: mc_call_control_variate(S0, K, r, sigma, T, N, seed=s)[0],
}

estimates = {name: np.array([f(s) for s in range(n_reps)]) for name, f in methods.items()}
plain_var = estimates["Plain Monte Carlo"].var()

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
bins = np.linspace(exact - 0.6, exact + 0.6, 80)

for i, (ax, (name, prices)) in enumerate(zip(axes, estimates.items())):
    ratio = plain_var / prices.var()
    ax.hist(prices, bins=bins, color=f"C{i}", alpha=0.85)
    ax.axvline(exact, color="black", linestyle="--")
    ax.set_yticks([])
    ax.text(0.01, 0.78, f"{name}\nspread ±{prices.std():.3f}   variance reduction ×{ratio:.1f}",
            transform=ax.transAxes, fontsize=10)

axes[0].set_title(f"Call price estimates, {N:,} paths each, {n_reps:,} repeated runs")
axes[-1].set_xlabel("Estimated call price (dashed line = Black-Scholes 10.4506)")

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/variance_reduction.png", dpi=150)

for name, prices in estimates.items():
    print(f"{name:32s} mean={prices.mean():.4f}  std={prices.std():.4f}  "
          f"variance reduction x{plain_var / prices.var():.1f}")