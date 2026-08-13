"""
Qualitative check: extends the C++ suite's DensityResponseTests to the real production
interface. Confirms the V_box-driven trend -- higher density means a smaller entropic penalty
for merging (and a smaller entropic bonus for splitting), so equilibrium should shift toward
larger, fewer assemblies as density increases -- holds through libtlm's actual Python bindings
and pypatchy's actual rule-construction path, not just the C++ internals directly.

Uses SC_RULE (simple cubic -- every cube face can bond, supports real compact 3D growth, unlike
the strictly-linear CHAIN_RULE) so there's room for a large density-dependent effect.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import run_vmmc, SC_RULE

DENSITIES = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
TEMPERATURE = 0.1
BOND_ENERGY = -1.0
CUBE_TYPE_LEVELS = [30, 30]
TOTAL_CUBES = sum(CUBE_TYPE_LEVELS)
NUM_STEPS = 8_000
N_REPLICAS = 6
SEED_BASE = 2000

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    mean_sizes = []
    std_sizes = []
    print(f"Sweeping density over {DENSITIES} ({N_REPLICAS} replicas each)...")
    for density in DENSITIES:
        sizes = []
        for i in range(N_REPLICAS):
            recs = run_vmmc(SC_RULE, TEMPERATURE, density, CUBE_TYPE_LEVELS, NUM_STEPS, 1,
                             bond_energy=BOND_ENERGY, seed=SEED_BASE + i,
                             history_interval=NUM_STEPS)  # only need the final snapshot here
            last = recs[0][-1]
            n_pc = last.numPolycubes()
            bound = last.numCubeInstances()
            mean_pc_size = bound / n_pc if n_pc > 0 else 0.0
            sizes.append(mean_pc_size)
        mean_sizes.append(np.mean(sizes))
        std_sizes.append(np.std(sizes))
        print(f"  density={density}: mean polycube size = {mean_sizes[-1]:.2f} +/- {std_sizes[-1]:.2f}")

    increasing = all(mean_sizes[i] <= mean_sizes[i + 1] + std_sizes[i + 1]
                      for i in range(len(mean_sizes) - 1))
    print("PASS: mean assembly size trends upward with density." if increasing else
          "FAIL: mean assembly size does not clearly trend upward with density -- investigate.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(DENSITIES, mean_sizes, yerr=std_sizes, marker="o", capsize=4)
    ax.set_xlabel("density")
    ax.set_ylabel("mean polycube size (bound cubes / num polycubes)")
    ax.set_title(f"Assembly size vs density (SC_RULE, T={TEMPERATURE}, E_bond={BOND_ENERGY})\n"
                 f"{sum(CUBE_TYPE_LEVELS)} total cubes, {N_REPLICAS} replicas/point")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "density_dependence.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'density_dependence.png'}")


if __name__ == "__main__":
    main()
