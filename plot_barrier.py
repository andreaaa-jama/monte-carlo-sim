from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcsim.barrier import down_and_out_call, down_and_out_call_discrete, mc_down_and_out_call
from mcsim.paths import simulate_gbm

S0, K, H, r, sigma, T = 100, 100, 90, 0.05, 0.2, 1
continuous = down_and_out_call(S0, K, H, r, sigma, T)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# ---- Left: a few paths, knocked out (grey) or alive (colour) ----
paths = simulate_gbm(S0, r, sigma, T, 252, 30, seed=7)
t = np.linspace(0, T, 253)
for p in paths:
    alive = p.min() > H
    ax1.plot(t, p, linewidth=1, color=None if alive else "lightgrey", alpha=0.9 if alive else 1,
             zorder=2 if alive else 1)
ax1.axhline(H, color="red", linewidth=2, label=f"Barrier H = {H}")
ax1.axhline(K, color="black", linestyle=":", label=f"Strike K = {K}")
ax1.set_title("Paths that touch the barrier are knocked out (grey)")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Price")
ax1.legend(loc="upper left")

# ---- Right: discretisation bias vs number of monitoring dates ----
steps = [4, 12, 26, 52, 126, 252, 504, 1000]
mc = [mc_down_and_out_call(S0, K, H, r, sigma, T, n, 200_000, seed=1) for n in steps]
prices = np.array([m[0] for m in mc])
errors = np.array([1.96 * m[1] for m in mc])
bgk = [down_and_out_call_discrete(S0, K, H, r, sigma, T, n) for n in steps]

ax2.errorbar(steps, prices, yerr=errors, fmt="o", capsize=3, label="Monte Carlo (95% CI)")
ax2.plot(steps, bgk, "s--", label="Continuous formula + BGK correction")
ax2.axhline(continuous, color="black", linestyle="--",
            label=f"Continuous barrier (exact) = {continuous:.3f}")
ax2.set_xscale("log")
ax2.set_xlabel("Number of monitoring dates (log scale)")
ax2.set_ylabel("Down-and-out call price")
ax2.set_title("Checking the barrier less often makes the option worth more")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/barrier.png", dpi=150)
for n, p, e, b in zip(steps, prices, errors, bgk):
    print(f"n={n:>5}  MC={p:.4f} ±{e:.4f}  BGK={b:.4f}")
print(f"continuous={continuous:.4f}")
