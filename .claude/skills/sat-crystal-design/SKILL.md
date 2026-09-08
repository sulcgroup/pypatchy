---
name: sat-crystal-design
description: Use when designing a PolycubesRule via the SAT-based search pipeline (pypatchy/design/sat/) - finding a rule that self-assembles a target crystal/finite structure while rejecting off-targets. Covers Polysat, the SolveStep pipeline (SATSolveStep/NegativeDesignStep/RankedValenceStep/VMMCDesignStep/VirtuallySeededGrowthDesignStep/FiniteSizeStep), SolveParams, MultiSolver, and PolysatSolution. This pipeline changes frequently and several pieces (multisolve(), crystalloyd_finder.py's role) have already shifted meaning since prior notes were written - verify against current source before trusting an old explanation, including this one. Builds directly on [[tlm-simulations]] (libtlm), [[polycubes-rules]] (the rule string format), and [[structure-topology]] (the Structure/TypedStructure target-matching machinery this whole pipeline is built on) - read those first if unfamiliar.
---

# SAT-based crystal/structure design

This is pypatchy's automated rule-design pipeline: given a target structure
(a finite polycube or an infinite crystal unit cell as a topology graph),
search for a [[polycubes-rules]] rule string that self-assembles it — and,
just as importantly, *doesn't* also assemble something else instead. It
encodes "does some coloring/rotation assignment exist that tiles this
topology" as a Boolean satisfiability (SAT) problem, solves it with Glucose4
(via PySAT), then validates each candidate solution by actually running it
through [[tlm-simulations]] (libtlm) — because a rule that's combinatorially
consistent on paper can still fail to nucleate, or preferentially assemble a
competing off-target crystal instead.

**This is the biggest and most actively-changing subsystem covered by any
pypatchy skill** (`pypatchy/design/` + `pypatchy/design/sat/` is ~17,000
lines). This doc covers the actual design/validation loop; it deliberately
does **not** cover `crystalloyd_finder.py`'s VMMC-simulation internals in
depth (2474 lines — read it directly if you're tuning `VMMCDesignConfig`
subclasses), nor the broader analysis/visualization utilities
(`design_yield.py`, `solve_vis.py`, `polycube_properties.py`,
`normalize_particle_set.py`, `design_amfa.py`, `pathway.py`,
`equilibration.py`) — those are downstream-of / adjacent-to this pipeline,
not part of running it. See [references/sat-api.md](references/sat-api.md)
for the SAT-variable scheme and class/field reference.

**A caution up front, given how fast this moves**: at least one claim in this
project's own prior memory notes was already stale by the time this doc was
written — `crystalloyd_finder.py` was noted as "old VMMC approach, unused,
reference only," but `VMMCDesignStep` currently *wraps* three of its runner
classes directly (`AnnealingCrystalloydSimulations`/
`ReplicaExchangeCrystalloydSimulations`/`BisectionCrystalloydSimulations`) —
it's centrally in active use. Don't assume a module is legacy just because an
older note said so; check who currently imports it.

## The core loop

```python
from pypatchy.design.solve_params import load_solve_params
from pypatchy.design.sat.solve_exec import solve

params = load_solve_params("my_topology_spec", nColors=4, nSpecies=2)
solution = solve(params)   # -> PolysatSolution | None

if solution:
    print(solution.decRuleNew())   # the PolycubesRule string
```

`solve()` is the thinnest possible entry point:
`Polysat(params); polysat.init(); polysat.find_solution()`. What actually
happens inside `find_solution()` is the interesting part:

```
loop (up to params.maxAltTries times):
    candidate = SATSolveStep.run()      # ask Glucose4 for a new SAT model
    for step in remaining pipeline steps:
        result = step.run(candidate, solver)
        PASS       -> continue to next step
        WEAK_PASS  -> continue, but flag this solution for human review if it survives
        REJECT     -> forbid this exact SAT model, feed back any constraints
                      (e.g. a competing crystal), try again
        TERMINAL   -> stop entirely (UNSAT -> False, solver timeout -> None)
    if every step PASSed (or PASS/WEAK_PASS with >=1 WEAK_PASS) -> done
```
Each step gets the *previous* step's `PolysatSolution` (or `None`, for
`SATSolveStep`, which is always first) and returns a `StepResult`. This is a
real pipeline of pluggable checks, not a monolithic solve function — see
"The step pipeline" below for what each one does and why the default
pipeline differs for a crystal vs. a finite target.

## Building `SolveParams`

`SolveParams` is the config object: target topology, `nSpecies`/`nColors`
search bounds, and the step pipeline. Two ways to get one:
- **`load_solve_params(spec_file_path, nColors, nSpecies)`** — loads a JSON
  spec from `~/.pypatchy/input/topologies/<name>.json` (or an explicit path).
  The spec JSON describes `bindings` (intra-unit-cell edges), optionally
  `extraConnections` (periodic/crystal edges — presence of these, or
  `crystal: true`, is what makes `SolveParams` pick the *crystal* default
  pipeline over the *finite-target* one), `nanoparticles`, and — for
  multifarious (multi-component) targets — a `topologies` list instead of a
  flat `bindings` key. This is exactly the `read_topology()` JSON shape
  documented in [[structure-topology]] — read that for what `bindings` vs.
  `extraConnections` actually means geometrically and why `Polysat` keeps
  both a full periodic and a bindings-only "local" copy of the target.
- **`SolveParams(name, nSpecies=, nColors=, **spec_dict)`** directly, e.g.
  from a dict you built in Python rather than a file on disk.

Constructing one builds a **default** step pipeline automatically (see
below) — customize it afterward via `params.set_step_kwargs(step_class_name,
**kwargs)` for tweaking an existing step's parameters, or by editing
`params.steps` (a `list[SolveStepSpec(step_class: str, step_kwargs: dict)]`)
directly to add/remove/reorder steps — `step_class` is looked up by name via
`eval()` in `Polysat.__init__`, so it must be an importable class name
visible in `design_rule.py`'s namespace (all the `SolveStep` subclasses are
star-imported there).

**Two other `SolveParams` knobs worth knowing up front**:
- `includeCanonicalConstraints` (default `True`) adds SAT-level
  symmetry-breaking clauses so the solver can't waste tries on
  species-permutation/rotation/color-ordering duplicates of a rule it's
  already tried. Above `canonical_constraints_max_clauses` (default 3M
  estimated clauses — this constraint's own construction is `O(nS²)` and can
  exhaust memory on a large search before the solve even starts), it's
  skipped automatically and `FilterSynonymsStep` catches duplicates at
  runtime instead — slower per-duplicate (a re-solve instead of exclusion)
  but bounded memory.
- `maxAltTries` (default 10) bounds total pipeline iterations before giving
  up — check `polysat.give_up_reason` (a `GiveUpReason` enum:
  `SOLVER_TIMEOUT`, `MAX_TRIES_EXHAUSTED`, `WEAK_PASS_PENDING_REVIEW`,
  `SOLVER_UNHANDLED_RESPONSE`, `WORKER_EXCEPTION`) after a `None` return to
  know *why* it gave up — they used to be indistinguishable from the return
  value alone.

## The step pipeline

Every step is a `SolveStep` subclass implementing
`run(soln: PolysatSolution | None, solver: Polysat) -> StepResult`, with
`StepResult.outcome` one of `PASS`/`WEAK_PASS`/`REJECT`/`TERMINAL`. See
[references/sat-api.md](references/sat-api.md) for the full `StepResult`/
`SolveStep` field reference. `SolveParams.__init__` builds one of two default
pipelines:

**Crystal target** (`extraConnections` present or `crystal=True`):
`SATSolveStep` → `FilterSynonymsStep` → `NegativeDesignStep` → `VMMCDesignStep`

**Finite target** (no periodicity):
`SATSolveStep` → `FilterSynonymsStep` → `FiniteSizeStep`

| Step | What it does |
|---|---|
| `SATSolveStep` | Always first, always present. Asks Glucose4 for a model of the SAT problem; also carries the SAT-problem-construction knobs (`segregate_bindings`, `segregate_colors`, `multifarious_behavior`, `forbid_self_interact`) since it's the one step `Polysat.init()` reads them from. `randomize_search=True` relabels/shuffles the CNF each try (deterministic given a fixed `random_seed`) so re-solving after a rejection doesn't just flip the cheapest bit of the previous model — use this when the search seems to be grinding through near-identical rejected candidates. |
| `FilterSynonymsStep` | Rejects a candidate whose canonical rule string (`rule.to_canonical()`) already appears in `forbidden_rules` — catches SAT symmetry duplicates that slipped past (or were never blocked by, if canonical constraints were skipped for size) the CNF-level symmetry breaking. |
| `NegativeDesignStep` | SAT-based off-target check: builds a *minimal* SAT sub-problem for each known off-target structure (pre-specified `forbidden_topologies`, or ones discovered lazily by `RankedValenceStep`/`VMMCDesignStep`), pins the candidate's own color/orientation variables, and asks whether the candidate could *also* satisfy that off-target. If yes → REJECT. Caches sub-problems per off-target-topology key. |
| `RankedValenceStep` | Runs the TLM `ranked_valence` deterministic model (`n_sims` replicas, `n_unit_cells`× material) and checks (via Python `TypedStructure.homomorphism()`, not libtlm's — deliberately, to avoid known C++ false negatives) whether any resulting polycube ≥ target size matches a target. No match but something assembled → REJECT with the assembled structures as new off-target `constraints` (these are what feed `NegativeDesignStep`'s lazily-discovered list). Nothing assembled at all → REJECT (a weaker signal: absence of assembly, not wrong assembly). |
| `VMMCDesignStep` | The finite-temperature complement to `RankedValenceStep` — `RankedValenceStep`'s greedy T=0 assembly can get stuck on one path and false-negative a rule that *would* assemble stochastically. Delegates to a `crystalloyd_finder.py` runner class chosen by the `VMMCDesignConfig` subclass you pass (`AnnealConfig`→annealing, `ReplicaExchangeConfig`→replica exchange, anything else→binary-search/bisection). Competing crystals it finds get scored for "solid enough to trust as a real off-target" via `offtarget_solidity_scorer`/`min_offtarget_solidity` (deliberately a *different*, usually stricter, gate than the one deciding "did my own target assemble" — a structure can be a good target match while being too locally glassy to trust as a genuine competing negative-design constraint), then BFS-cropped (`local_constraint_subgraphs`) into one or more off-target unit cells for `NegativeDesignStep`. A crop that turns out to still match the *design target itself* (e.g. a defect-free local motif inside an otherwise-defective competing crystal) is discarded rather than forwarded — forwarding it would make `NegativeDesignStep` reject the very target being designed for. |
| `VirtuallySeededGrowthDesignStep` | Tests growth from a fixed, already-known-correct periodic seed (built from the SAT solution's own `spatial_map` via `tlm_runner.build_seed_pattern`/`to_tlm_seed_config`) rather than spontaneous nucleation — i.e. "does this rule correctly *extend* the target lattice at a corner/edge/facet seam" rather than "does it nucleate the target from nothing." Runs a fixed sequence of seed geometries (`_SEEDED_GROWTH_STAGES`): corner and edge stages gate PASS/REJECT; bare-facet stages are diagnostic-only and never gate, because the model's cooperative-binding design means a facet legitimately can't self-nucleate a new layer above threshold — that's the intended emergent behavior, not a defect. |
| `FiniteSizeStep` | The finite-target analogue of the crystal-only assembly-check steps above — runs the deterministic `finite_size` TLM model in batches (`pcs_n_batches`/`pcs_batch_size`) and checks for a target match. |

`StepResult.constraints` (a list of `TypedStructure`) is how a step hands
newly-discovered off-target structures forward — `Polysat.find_solution()`
feeds every returned constraint into `_add_off_target()`, which does
subsumption bookkeeping (a smaller structure that homomorphically embeds
into an existing one is a strictly stronger constraint, so the weaker one is
pruned) before appending to `solver._off_target_unit_cells`, which
`NegativeDesignStep` reads on every subsequent try. This is how the pipeline
gets *smarter* over the course of one run: a competing crystal found by
`VMMCDesignStep` on try 5 becomes something `NegativeDesignStep` can reject
via cheap SAT check on try 6, without re-running the expensive simulation.

## Running many searches: `MultiSolver`

For sweeping a range of `(nSpecies, nColors)` pairs (smallest-first, since a
smaller rule is usually preferable when one exists) rather than one fixed
pair:
```python
from pypatchy.design.sat.multisolve import MultiSolver

ms = MultiSolver("my_topology_spec", num_cpus=8, minS=2, maxS=6, minC=1, maxC=8)
ms.run(callback=lambda nS, nC, result: print(nS, nC, result))
```
**The standalone `multisolve()` function is deprecated in favor of
`MultiSolver`** — it raises a `DeprecationWarning` on call; don't use it in
new code, and don't be surprised to find it in an older script.
`MultiSolver.run()` is memory- *and* CPU-aware: `num_cpus` is a total core
budget, not a process count — the actual worker-pool size
(`num_parallel_solves()`) divides it by `SolveParams.est_cpus_per_solve()`,
since a single solve can itself burst across many cores (`VMMCDesignStep`/
`RankedValenceStep`/`FiniteSizeStep` each run several libtlm replicas
concurrently via OpenMP, invisible to anything only counting solve
*processes*). A separate `mem_budget_gb` gate (auto-detected from system RAM
if not given) additionally throttles concurrency by each queued spec's
estimated memory cost — added after a real run exhausted server memory by
being CPU-budget-compliant but memory-blind. `ms.spec` is the loaded
`SolveParams` (with placeholder nS/nC) — mutate its `.steps` before calling
`.run()` the same way as a single-shot `SolveParams`.

## Reading a result: `PolysatSolution`

- `.decRuleNew() -> str` — the rule as a [[polycubes-rules]] string (the
  practical "what did I get" accessor).
- `.rule -> PolycubesRule` — the rule object itself.
- `.type_counts() -> list[int]` — per-species cube counts for one unit
  cell/target instance — feed this (scaled up) as `cube_type_levels` into a
  [[tlm-simulations]] `TLMParameters` for further/independent validation.
- `.nanoparticle_map`, `.hasNanoparticles()`, `.numNanoparticleTypes()`.
- `.to_polycube() -> PolycubeStructure`, `.has_coord_map()`,
  `.exportScene(modelname)` — spatial/visual export of the solved structure.
- `.describe() -> str` — human-readable summary (written into each
  solution's `soln_info.txt`).

## Where results land on disk

Each solution (or forbidden rejection) gets its own
`tlm_data_dir()/soln_<uuid>/` (per this repo's `CLAUDE.md`), containing
`problem_info.json` (the SAT problem shape + bindings, used by
`Polysat.matches_problem()` to check compatibility when re-forbidding),
`solution.pickle` (the `PolysatSolution`), `soln_info.txt` (human-readable),
and any step-specific artifacts (`VMMCDesignStep`/`RankedValenceStep` save
assembled/competing polycube JSON snapshots here too, e.g.
`rv_pass_0.json`/`rv_competing_0.json`). Structured per-try event logs go to
`get_log_dir()/SAT/<name>.jsonl` (`PolysatLogger`) — `pypatchy.design.reforbid`
and `pypatchy.design.solve_records_parser` exist specifically to parse these
back out for re-forbidding a later run or building the `solve_vis.py` pipeline
plots, rather than re-deriving everything from scratch. Passing a previous
run's `.jsonl` (via `SolveParams`'s `pre_forbid` list) re-applies its
rejections to a new run: structural rejections (`FilterSynonymsStep`,
`NegativeDesignStep`, `RankedValenceStep`, `FiniteSizeStep`) are reused
unconditionally, but `VMMCDesignStep` rejections are only reused if this
run's VMMC hyperparameters are still protocol-compatible with the logged
ones (`reforbid.vmmc_protocol_compatible`) *and* the candidate still
re-scores as rejected under this run's live scorer/thresholds
(`reforbid.rescore_vmmc_candidate`) — a scorer or threshold change since the
log was written can un-reject a candidate, and re-forbidding it anyway would
silently exclude part of the search space for no current reason.

## Further reading

[references/sat-api.md](references/sat-api.md) — the SAT variable scheme
(`B`/`F`/`C`/`P`/`O`/`T`/`D`/`A`), `TorsionType` (`NONE`/`FULL`/`PARTIAL`),
`PolycubeSATProblem`'s constraint-generation methods, `SolveParams`'s full
field list, and `PolysatSolution`'s full method list.

For `VMMCDesignStep`'s actual simulation-control knobs
(`AnnealConfig`/`BisectionConfig`/`ReplicaExchangeConfig`/`KineticsConfig`),
read `pypatchy/design/crystalloyd_finder.py` directly — it's too large
(2474 lines) to summarize reliably here, and it's exactly the kind of module
this doc's opening caution is about: don't assume you know its current role
from an old note.
