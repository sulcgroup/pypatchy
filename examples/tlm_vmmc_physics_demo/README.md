# TLM VMMC physics demo

Exploratory verification scripts, not part of the analysis pipeline. They check that the physics
derived in [`vmmc_physics_derivation.md`](../../../polycubes/vmmc_physics_derivation.md) — a
from-scratch statistical-mechanics re-derivation of the VMMC target distribution and
move-proposal scheme, done independently of the existing code and then diffed against it — holds
up when run through the real production interface: `libtlm`'s Python bindings and pypatchy's own
rule construction (`PolycubesRule`), not just the C++ unit tests in
`polycubes/cpp/tests/test_vmmc_directed_moves.cpp`, which check the mechanism against hand-built
fixtures. This is the same interface `pypatchy.design.sat`'s SAT-validation pipeline uses in
production. Written to verify commit `f703d62c` on the `vmmc-proposal-bias` branch (in the
`polycubes` repo) before pushing it.

That document lives at the top level of the sibling `polycubes` repo, since it's fundamentally
about the VMMC engine implemented there, not about pypatchy — the relative link above assumes the
standard four-repo layout described in the container's top-level `/root/CLAUDE.md` (this repo and
`polycubes` checked out as siblings under the same parent directory); adjust the path if your
checkout differs.

Run any script directly from within the pypatchy micromamba env (auto-activated), e.g.:
```
python3 monomer_dimer_equilibrium.py
```
Each prints a PASS/FAIL summary and saves plot(s) to `output/`.

## `monomer_dimer_equilibrium.py` — the rigorous quantitative check

For a rule where every cube has exactly one bond site (so the only possible objects are free
monomers and dimers — no chains, no further merging), the derived target distribution
`pi(sigma)` (see "The model" in `vmmc_physics_derivation.md`) gives an exact closed-form
prediction for the equilibrium distribution of dimer counts. This is a genuine
detailed-balance-correctness test: the *equilibrium* the simulation
converges to should match this formula if and only if detailed balance is actually satisfied,
regardless of the specific proposal/acceptance mechanics used internally. Result: empirical mean
within 0.4 theoretical standard deviations of the prediction, histogram shape closely tracking the
theoretical curve.

## `density_dependence.py` — qualitative

Extends the C++ suite's `DensityResponseTests` to the production interface: higher density should
mean a smaller entropic penalty for merging (and smaller bonus for splitting), so mean assembly
size should increase with density. Result: clean, monotonic increase from ~3 to ~38 mean cubes per
polycube across a density sweep from 0.01 to 0.9.

## `merge_coarsening.py` — most direct visible signature of the bug fixed in commit `f703d62c`

Tracks polycube count over time for a rule that supports real compact 3D growth. Expect a
rise-then-fall (coarsening) curve: nucleation initially raises polycube count as small clusters
form, then genuine merge lowers it again as they combine. This specifically requires merge to be
reachable at a non-negligible rate for real structures — under the pre-fix design, reaching that
outcome needed pseudocluster growth to independently recruit every internal bond of an entire
polycube, a probability that decays exponentially with structure size. Result: a sharp peak
(~10.5 polycubes) around step 160, falling ~67% to a fluctuating plateau (~3.5) by step ~750 —
a textbook coarsening curve.

(Note: this folder deliberately does not include a live before/after comparison against the
pre-fix code — it would require rebuilding and temporarily swapping the shared production
`libtlm.so` install, judged not worth the risk/complexity for this check. The `merge_coarsening.py`
result above is compared against the pre-fix design's known theoretical failure mode, not against
a re-run of the actual old code.)

## `common.py`

Shared rule strings (`CHAIN_RULE`, `SC_RULE`, `DIMER_ONLY_RULE`) and a `run_vmmc()` helper wrapping
`libtlm.TLMParameters` + `libtlm.runSimulations`. See its docstrings for exact parameter meanings,
including one thing confirmed empirically against the current libtlm build that isn't obvious from
the existing test suite: `history_interval` (not the `runSimulations` third argument) controls
returned-snapshot spacing directly, and values greater than 1 work fine.

`pypatchy/polycubeutil/ext/tlm_analytics.py`'s `TLMDataGroup` is also available for turning raw
`list[list[TLMHistoryRecord]]` output into a pandas DataFrame if that's more convenient for future
scripts in this folder — not used here since these three needed specific per-replica/per-snapshot
access the raw records already provide directly.
