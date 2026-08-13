"""
Shared rule strings and a thin libtlm.runSimulations() wrapper for the scripts in this folder.

See README.md for what this folder is checking and why.
"""
import libtlm

# Exact Python-side match to the C++ test suite's canonical A/B chain rule
# (test_vmmc_invariants.cpp / test_checkpoint_resume.cpp's makeChainParams): type 0 binds
# LEFT=-1/RIGHT=1, type 1 binds LEFT=1/RIGHT=-1 -- an unbounded polarity-inverting X-axis chain.
# Field order is LEFT|RIGHT|BOTTOM|TOP|BACK|FRONT.
CHAIN_RULE = "-1|1||||_1|-1||||"

# Simple cubic crystal rule (2 types, no torsion) -- supports real 3D/multiply-connected growth,
# same rule used in tests/test_libtlm.py.
SC_RULE = "1|1|1|1|1|1_-1|-1|-1|-1|-1|-1"

# Type 0 has ONLY a LEFT patch (color 1); type 1 has ONLY a RIGHT patch (color -1). Each cube
# has exactly one bond site, so once bonded neither cube has any remaining free face -- this
# guarantees monomer<->dimer only, no chains, no larger merges (any merge attempt between two
# already-full dimers finds zero free faces on either side and is a structural no-op). That's
# what makes the equilibrium distribution analytically tractable in monomer_dimer_equilibrium.py.
DIMER_ONLY_RULE = "1|||||_|-1||||"


def run_vmmc(rule: str, temperature: float, density: float, cube_type_levels: list[int],
             num_steps: int, n_replicas: int, bond_energy: float = -1.0, seed: int = 0,
             history_interval: int = 1) -> list[list["libtlm.TLMHistoryRecord"]]:
    """
    Runs n_replicas independent VMMC simulations and returns one list of TLMHistoryRecord per
    replica, snapshotted every history_interval steps (confirmed empirically against the current
    libtlm build -- history_interval controls returned-snapshot spacing directly, and values > 1
    work fine here as long as history_interval <= num_steps).

    interaction_matrix is built explicitly (not via libtlm.generateInteractionMatrix's default)
    so the exact bond energy is known and controlled -- needed for the analytical comparison in
    monomer_dimer_equilibrium.py, and just good practice for the other scripts too. Every one of
    the rule's 4 angles gets the same strength (non-torsional), matching all three rules above.
    """
    interaction_matrix = {(-1, 1, angle): bond_energy for angle in range(4)}
    params = libtlm.TLMParameters(
        rule=rule,
        temperature=temperature,
        density=density,
        num_steps=num_steps,
        cube_type_levels=cube_type_levels,
        depletes_types=True,
        history_interval=history_interval,
        interaction_matrix=interaction_matrix,
        seed=seed,
        model_type="vmmc",
    )
    # record_interval=0 (3rd positional arg): suppresses libtlm's per-step console logging --
    # it controls console log frequency only, NOT which steps get returned (confirmed
    # empirically -- that's entirely history_interval's job).
    return libtlm.runSimulations(params, n_replicas, 0)
