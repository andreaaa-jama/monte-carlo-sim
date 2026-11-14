from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mcsim.paths import simulate_gbm

S0, mu, sigma, T = 100, 0.05, 0.2, 1
n_steps, n_paths = 252, 100_000

paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=1)
t = np.linspace(0, T, n_steps + 1)
final = paths[:, -1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(t, paths[:20].T, linewidth=0.8)
ax1.set_title("20 simulated price paths (GBM)")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Price")

ax2.hist(final, bins=100, density=True, alpha=0.8)
ax2.axvline(final.mean(), color="black", linestyle="--",
            label=f"Mean = {final.mean():.2f}")
ax2.axvline(S0 * np.exp(mu * T), color="red", linestyle=":",
            label=f"Theory = {S0 * np.exp(mu * T):.2f}")
ax2.set_title(f"Final prices ({n_paths:,} paths)")
ax2.set_xlabel("Price at T")
ax2.set_ylabel("Density")
ax2.legend()

fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/gbm_paths.png", dpi=150)
print("Saved figures/gbm_paths.png")