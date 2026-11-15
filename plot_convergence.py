from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcsim.paths import simulate_gbm
from mcsim.pricers import bs_call, mc_call

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
exact = bs_call(S0, K, r, sigma, T)
discount = np.exp(-r * T)

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1])

# ---- Left: running estimate of the price as paths are added (the "funnel") ----
ax = fig.add_subplot(gs[:, 0])
n_max, n_runs = 100_000, 12
n = np.arange(1, n_max + 1)

all_payoffs = []
for run in range(n_runs):
    ST = simulate_gbm(S0, r, sigma, T, 1, n_max, seed=run)[:, -1]
    payoffs = discount * np.maximum(ST - K, 0.0)
    all_payoffs.append(payoffs)
    running_mean = np.cumsum(payoffs) / n
    ax.plot(n, running_mean, linewidth=1, alpha=0.8)

payoff_std = np.concatenate(all_payoffs).std()
band = 1.96 * payoff_std / np.sqrt(n)
ax.fill_between(n, exact - band, exact + band, color="grey", alpha=0.2,
                label="95% band (±1.96 σ/√N)")
ax.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes = {exact:.2f}")
ax.set_xscale("log")
ax.set_ylim(exact - 6, exact + 6)
ax.set_xlabel("Number of paths used (log scale)")
ax.set_ylabel("Monte Carlo price estimate")
ax.set_title(f"{n_runs} independent runs: more paths, tighter estimate")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# ---- Right: spread of the estimate for different numbers of paths ----
Ns = [100, 1_000, 10_000, 100_000]
n_reps = 1_000
bins = np.linspace(exact - 6, exact + 6, 80)
axes = []
for i, N in enumerate(Ns):
    ax_h = fig.add_subplot(gs[i, 1], sharex=axes[0] if axes else None)
    axes.append(ax_h)
    prices = np.array([mc_call(S0, K, r, sigma, T, N, seed=s)[0] for s in range(n_reps)])
    ax_h.hist(prices, bins=bins, color=f"C{i}", alpha=0.85)
    ax_h.axvline(exact, color="black", linestyle="--")
    ax_h.set_yticks([])
    ax_h.text(0.02, 0.8, f"N = {N:,}   (spread ±{prices.std():.2f})",
              transform=ax_h.transAxes, fontsize=10)
    if i < len(Ns) - 1:
        plt.setp(ax_h.get_xticklabels(), visible=False)
axes[0].set_title(f"Price estimates over {n_reps:,} repeated runs")
axes[-1].set_xlabel("Estimated call price (dashed line = Black-Scholes)")

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/convergence.png", dpi=150)
print("Saved figures/convergence.png")