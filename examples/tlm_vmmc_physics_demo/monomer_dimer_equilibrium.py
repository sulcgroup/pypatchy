"""
The rigorous, quantitative centerpiece of this demo folder.

Checks that the *equilibrium distribution* VMMC actually samples for a simple, analytically
tractable system matches the exact prediction from the derived target distribution
pi(sigma) ~ exp(-beta*E(sigma)) * prod_i[V_i^n_i / n_i!] * V_box^m (see "The model" in
../../../polycubes/vmmc_physics_derivation.md). This is a genuine detailed-balance-correctness
test, not an efficiency one: whatever specific proposal/acceptance mechanics the code uses
internally, the *equilibrium* it converges to should match this formula if and only if detailed
balance is actually satisfied -- so this directly tests the thing that derivation-and-fix effort
was about, not just "does something happen."

The system: DIMER_ONLY_RULE (see common.py) -- N cubes of type 0, N of type 1, each with exactly
one bond site, so the only possible objects are free monomers and dimers (no chains, no further
merging once dimerized). At dimer count x (0<=x<=N): n_free_A = n_free_B = N-x, m = x. With
V_A = V_B = N/density and V_box = 2N/density (only these two species exist in the system), the
ratio pi(x+1)/pi(x) telescopes to an exact closed form (derived fresh for this demo, not copied
from anywhere):

    pi(x+1)/pi(x) = exp(-E_bond/T) * (N-x)^2 * 2*density/N

which gives the full equilibrium distribution P(x) by cumulative product + normalization.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import run_vmmc, DIMER_ONLY_RULE

# Parameters chosen (see plan/session notes) to give a well-spread interior peak rather than
# saturating near x=0 or x=N, so the shape comparison is actually informative.
N = 20
TEMPERATURE = 1.0
DENSITY = 0.05
BOND_ENERGY = -0.6
NUM_STEPS = 20_000
HISTORY_INTERVAL = 200
N_REPLICAS = 25
BURN_IN_FRACTION = 0.25  # discard the first 25% of each replica's snapshots
SEED_BASE = 1000

OUTPUT_DIR = Path(__file__).parent / "output"


def theoretical_distribution(n: int, bond_energy: float, temperature: float, density: float):
    beta = 1.0 / temperature
    log_weights = [0.0]
    for x in range(n):
        ratio = math.exp(-beta * bond_energy) * (n - x) ** 2 * 2 * density / n
        log_weights.append(log_weights[-1] + math.log(ratio))
    m = max(log_weights)
    weights = [math.exp(w - m) for w in log_weights]
    total = sum(weights)
    p = [w / total for w in weights]
    mean = sum(x * p[x] for x in range(n + 1))
    var = sum((x - mean) ** 2 * p[x] for x in range(n + 1))
    return p, mean, math.sqrt(var)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    p_theory, mean_theory, std_theory = theoretical_distribution(N, BOND_ENERGY, TEMPERATURE, DENSITY)
    print(f"Theoretical equilibrium: mean={mean_theory:.3f}, std={std_theory:.3f}")

    print(f"Running {N_REPLICAS} replicas of {NUM_STEPS} steps each...")
    all_series = []  # one list of (step, x) per replica, for the diagnostic time-series plot
    samples = []      # pooled post-burn-in dimer-count samples across all replicas
    for i in range(N_REPLICAS):
        recs = run_vmmc(DIMER_ONLY_RULE, TEMPERATURE, DENSITY, [N, N], NUM_STEPS, 1,
                         bond_energy=BOND_ENERGY, seed=SEED_BASE + i,
                         history_interval=HISTORY_INTERVAL)[0]
        xs = [r.numPolycubes() for r in recs]
        steps = [r.stepCount() for r in recs]
        all_series.append((steps, xs))
        burn_in_n = int(len(xs) * BURN_IN_FRACTION)
        samples.extend(xs[burn_in_n:])

    samples = np.array(samples)
    mean_emp, std_emp = samples.mean(), samples.std()
    z = (mean_emp - mean_theory) / std_theory
    print(f"Empirical equilibrium ({len(samples)} samples): mean={mean_emp:.3f}, std={std_emp:.3f}")
    print(f"(empirical mean - theoretical mean) / theoretical std = {z:.2f}")
    if abs(z) < 2.0:
        print("PASS: empirical mean within 2 theoretical standard deviations of the prediction.")
    else:
        print("FAIL: empirical mean is more than 2 theoretical standard deviations off -- "
              "investigate before trusting this result (do not just retune parameters).")

    # --- Plot 1: theoretical P(x) vs empirical histogram ---
    fig, ax = plt.subplots(figsize=(7, 5))
    xs_range = np.arange(N + 1)
    ax.plot(xs_range, p_theory, "o-", color="C1", label="theoretical P(x)", zorder=3)
    ax.hist(samples, bins=np.arange(-0.5, N + 1.5, 1), density=True, alpha=0.5, color="C0",
            label=f"empirical (n={len(samples)} samples)")
    ax.set_xlabel("dimer count x")
    ax.set_ylabel("probability")
    ax.set_title(f"Monomer<->dimer equilibrium: theory vs VMMC simulation\n"
                 f"N={N}, T={TEMPERATURE}, density={DENSITY}, E_bond={BOND_ENERGY}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monomer_dimer_equilibrium_distribution.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'monomer_dimer_equilibrium_distribution.png'}")

    # --- Plot 2: time series for a few replicas, to visually confirm equilibration ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for steps, xs in all_series[:6]:
        ax2.plot(steps, xs, alpha=0.6, linewidth=1)
    ax2.axhline(mean_theory, color="k", linestyle="--", label="theoretical mean")
    burn_in_step = NUM_STEPS * BURN_IN_FRACTION
    ax2.axvline(burn_in_step, color="gray", linestyle=":", label="burn-in cutoff")
    ax2.set_xlabel("step")
    ax2.set_ylabel("dimer count x")
    ax2.set_title("Dimer count vs step (first 6 replicas) -- equilibration check")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "monomer_dimer_equilibrium_timeseries.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'monomer_dimer_equilibrium_timeseries.png'}")


if __name__ == "__main__":
    main()
