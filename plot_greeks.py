from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcsim.greeks import mc_delta
from mcsim.pricers import bs_delta

S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
N = 100_000
exact = bs_delta(S0, K, r, sigma, T)
bumps = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10])

results = {}
for crn in (True, False):
    est = [mc_delta(S0, K, r, sigma, T, N, h=h, seed=1, common_random_numbers=crn) for h in bumps]
    results[crn] = (np.array([e[0] for e in est]), np.array([e[1] for e in est]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# ---- Left: the estimate itself ----
for crn, label, c in [(False, "Independent random numbers", "C1"),
                      (True, "Common random numbers", "C0")]:
    est, se = results[crn]
    ax1.errorbar(bumps, est, yerr=1.96 * se, fmt="o-", capsize=3, color=c, label=label)
ax1.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes delta = {exact:.4f}")
ax1.set_xscale("log")
ax1.set_ylim(exact - 0.5, exact + 0.5)
ax1.set_xlabel("Bump size h (log scale)")
ax1.set_ylabel("Estimated delta (95% CI)")
ax1.set_title(f"Delta by finite difference, {N:,} paths")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.3)

# ---- Right: the standard error ----
for crn, label, c in [(False, "Independent random numbers", "C1"),
                      (True, "Common random numbers", "C0")]:
    ax2.loglog(bumps, results[crn][1], "o-", color=c, label=label)
ax2.set_xlabel("Bump size h (log scale)")
ax2.set_ylabel("Standard error of the delta estimate")
ax2.set_title("Independent draws: error grows like 1/h. Common draws: it stays flat")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/greeks.png", dpi=150)
for h, (e1, s1), (e2, s2) in zip(bumps, zip(*results[True]), zip(*results[False])):
    print(f"h={h:>5}  CRN={e1:.4f} ±{s1:.4f}   independent={e2:.4f} ±{s2:.4f}")