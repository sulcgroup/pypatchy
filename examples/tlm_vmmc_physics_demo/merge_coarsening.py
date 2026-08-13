"""
Qualitative check, and the most direct visible signature of the actual bug fixed in commit f703d62c.

Tracks numPolycubes() vs step for SC_RULE (simple cubic -- supports real compact 3D growth).
Expect a rise-then-fall curve: nucleation/monomer addition initially raises polycube count as
many small clusters form from the free-cube pool, then genuine coalescence (merge) lowers it
again as those clusters combine into fewer, larger ones. This specifically requires merge to be
reachable at a non-negligible rate for real (not tiny) structures -- under the pre-fix
undirected/growth-then-classify design (see ../../../polycubes/vmmc_physics_derivation.md and
commit f703d62c), reaching that outcome needed pseudocluster growth to independently recruit
every internal bond of an entire polycube, probability decaying exponentially with structure
size, so this curve would very plausibly have plateaued near its peak instead of falling. That
comparison is an expectation from the (deliberately skipped, per plan) before/after
test, not something re-verified directly here -- what IS verified directly is that the fixed
code's own dynamics show real coalescence, not just monomer accretion.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import run_vmmc, SC_RULE

TEMPERATURE = 0.1
DENSITY = 0.3
BOND_ENERGY = -1.0
CUBE_TYPE_LEVELS = [40, 40]
NUM_STEPS = 4_000
HISTORY_INTERVAL = 40
N_REPLICAS = 15
SEED_BASE = 3000

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Running {N_REPLICAS} replicas of {NUM_STEPS} steps each...")
    all_pc = []
    for i in range(N_REPLICAS):
        recs = run_vmmc(SC_RULE, TEMPERATURE, DENSITY, CUBE_TYPE_LEVELS, NUM_STEPS, 1,
                         bond_energy=BOND_ENERGY, seed=SEED_BASE + i,
                         history_interval=HISTORY_INTERVAL)[0]
        all_pc.append([r.numPolycubes() for r in recs])
    steps = [r.stepCount() for r in recs]  # same for every replica
    arr = np.array(all_pc)
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0)

    peak_idx = int(np.argmax(mean_curve))
    peak_step, peak_value = steps[peak_idx], mean_curve[peak_idx]
    final_value = mean_curve[-1]
    print(f"Peak: {peak_value:.2f} polycubes at step {peak_step}")
    print(f"Final (step {steps[-1]}): {final_value:.2f} polycubes")
    fall_fraction = (peak_value - final_value) / peak_value if peak_value > 0 else 0.0
    print(f"Fractional decline from peak: {fall_fraction:.1%}")
    if peak_idx < len(mean_curve) - 1 and fall_fraction > 0.2:
        print("PASS: clear rise-then-fall (coarsening) signature -- merge is measurably "
              "contributing beyond simple monomer accretion.")
    else:
        print("FAIL: no clear coarsening signature (curve rose but didn't meaningfully fall, "
              "or never fell below its peak) -- investigate before trusting this result.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mean_curve, color="C0", label="mean over replicas")
    ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="C0")
    ax.axvline(peak_step, color="gray", linestyle=":", label=f"peak (step {peak_step})")
    ax.set_xlabel("step")
    ax.set_ylabel("number of polycubes")
    ax.set_title(f"Coarsening signature: rise (nucleation) then fall (merge)\n"
                 f"SC_RULE, T={TEMPERATURE}, density={DENSITY}, "
                 f"{sum(CUBE_TYPE_LEVELS)} total cubes, {N_REPLICAS} replicas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "merge_coarsening.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'merge_coarsening.png'}")


if __name__ == "__main__":
    main()
