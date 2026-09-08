# `libtlm` API reference

Generated from `~/git/polycubes-clone/cpp/src/PolycubesPython.cpp` (~1150
lines, pybind11). This is a from-scratch re-derivation, not carried over from
an older note — that file changes often enough that re-reading it directly is
safer than trusting any cached summary, including this one, for anything you
depend on being exact. Line numbers are omitted since they drift; class/method
names are stable enough to `grep` for.

## `TLMParameters` (C++ `SimulationParameters`)

Four constructors are exposed:

1. **Full positional**: `(depletes_types: bool, temperature: float, density: float, rule: str, cube_type_levels: list[int], num_steps: int, interaction_matrix: dict[tuple[int,int,int], float], seed: int, history_interval: int)`
2. **Short positional**: `(depletes_types, temperature, density, rule, cube_type_levels, num_steps, history_interval)` — no interaction_matrix/seed.
3. **Keyword-argument (ergonomic, preferred for hand-written code)**:
   `TLMParameters(rule, temperature, density, num_steps, cube_type_levels, depletes_types=False, history_interval=1, interaction_matrix={}, seed=0, model_type="")`
   — `rule`/`temperature`/`density`/`num_steps`/`cube_type_levels` are the
   only required args. `interaction_matrix={}` triggers auto-generation via
   `generateInteractionMatrix(rule)` (see the "torsion" section of the main
   skill doc for why this can be wrong for a partial-torsion rule).
   `seed=0` means "use current time." `model_type` selects the simulation
   model directly (see main doc); empty defers to temperature-based fallback.
4. **Staged**: same as (3) but `stages: list[TLMStageInfo]` replaces
   `cube_type_levels`/`num_steps` (total steps = sum of stage `max_steps`).

**No `torsion` argument exists in any constructor** — it's expressed as the
angle dimension of `interaction_matrix` keys (`(color1, color2, angle)`,
angle 0-3). This is the single most load-bearing difference from older
documentation of this API.

Read-only properties: `depletes_types`, `temperature`, `density`,
`cube_types`, `num_steps`, `interaction_matrix`, `rand_seed`,
`history_interval`, `stages` (list of `TLMStageInfo`).
Read-write: `model_type` (str — can be reassigned after construction, as
`examples/run_libtlm.py` does).

Methods: `write_to_file(fp: str)`.

**Pickle**: supported, but the format changed incompatibly when torsion
moved into the interaction matrix — a tuple of fewer than 9 elements raises
`RuntimeError("Invalid state for TLMParameters!")` on unpickle. Old pickles
from before this change cannot be loaded, full stop.

## `TLMStageInfo` (C++ `StageInfo`)

`TLMStageInfo(cube_type_levels: list[int], max_steps: int, temperature: float = -1.0, density: float = -1.0, temperature_end: float = -1.0, linear_ramp: bool = False)`

`-1` for `temperature`/`density` defers to the global `TLMParameters` value.
`temperature_end > 0` ramps temperature across the stage — geometric
(constant ratio per step) by default, or linear (constant increment) if
`linear_ramp=True`. Read-write attributes matching the constructor args.

## `TLMVirtualSeedConfig` (C++ `VirtualSeedConfig`)

`TLMVirtualSeedConfig(unit_cell_dims: tuple[int,int,int], positions: list[tuple[int,int,int]], cube_type_idxs: list[int], rotations_xyzw: list[tuple[float,float,float,float]], fill_neg_x: bool, fill_neg_y: bool, fill_neg_z: bool, target_total_cubes: int)`

Built from four parallel lists rather than a dict (pybind11 can't key a map
on a Python tuple without a custom caster) — `positions[i]`/`cube_type_idxs[i]`/
`rotations_xyzw[i]` describe the same cell. `rotations_xyzw` entries are
scalar-last `(x, y, z, w)` quaternions, matching `TLMCubeData.getRotation()`'s
convention. The seed is the union of active half-spaces (`fill_neg_x` etc. —
one active axis = flat facet, two = edge, three = corner) filled with the
periodic pattern, tiled by `unit_cell_dims` on active axes (an inactive axis
is unconstrained/infinite, so its phase doesn't matter). Required (and only
used) when a `TLMSimulationGroup`'s `params.model_type ==
"virtually_seeded_growth"`. Building one from a SAT solution is
design-pipeline territory — see `pypatchy.design.tlm_runner.build_seed_pattern`/
`to_tlm_seed_config`.

## `TLMSimulationGroup`

Constructor: `TLMSimulationGroup(params: TLMParameters, log_every: int = 0, history_dir: str = "", seed_config: TLMVirtualSeedConfig | None = None)`.
Model is picked from `params.model_type` (or temperature fallback) — **not**
a constructor argument. `history_dir`, if set, streams each replica's
snapshots to `<history_dir>/sim<i>_history.bin` instead of accumulating in
RAM (read with `tlm_runner.read_history_bin`). `seed_config` is required iff
`model_type == "virtually_seeded_growth"`, ignored otherwise.

Methods:
- `init(n_replicas: int)` — initialize `n_replicas` fresh random-start replicas.
- `init_from(records: list[TLMHistoryRecord])` — checkpoint/resume; see the
  caveats in the main skill doc. VMMC-model groups only; raises otherwise.
- `run()` — run all replicas to completion (OpenMP-parallel in release builds).
- `step_all(n: int = 1)` — advance every non-done replica by `n` steps.
- `records(sim_idx: int) -> list[TLMHistoryRecord]` — full history for one replica.
- `dump_data(path: str)` — write `sim0.json, sim1.json, ...` to `path`. Raises
  for `virtually_seeded_growth` groups.
- `get_sim(sim_idx: int) -> TLMSimulation` — a live handle to one replica.

Properties: `num_sims` (int), `params` (`TLMParameters`).

## `TLMSimulation` (per-replica handle, via `group.get_sim(i)`)

- `step(n: int = 1)`, `run()`, `is_done() -> bool`, `current_step() -> int`
- `records() -> list[TLMHistoryRecord]`, `last_record() -> TLMHistoryRecord`
- `num_polycubes() -> int` (distinct connected clusters),
  `num_cubes() -> int` (total individual cube instances)
- `energy() -> float` (current total bond energy)
- `params() -> TLMParameters`
- `get_polycubes() -> list[TLMPolycube]` — the **current live state**
  (not a stored record — constructs a fresh snapshot on call)
- `set_temperature(T: float)` — reassign live temperature in place
  (replica-exchange/parallel tempering); raises for a staged (non-isothermal)
  simulation
- VMMC-only (else `None`): `acceptance_ratio() -> float | None`,
  `accepted_moves() -> int | None`, `rejected_moves() -> int | None`
- VirtuallySeededGrowth-only (else `None`): `is_jammed()`,
  `consecutive_rejects()`, `frontier_size()`

## `PolycubesProgram` (context-manager wrapper)

`PolycubesProgram(params: TLMParameters, n_replicas: int = 1)`. **Second
positional argument is an int, not a model-type string** — see the
"known-stale trap" callout in the main skill doc; set `params.model_type`
instead.

```python
with libtlm.PolycubesProgram(params, n_replicas) as prog:
    prog.set_halt_condition(some_callable)   # optional; called after every step in run_for
    prog.run_for(500)                        # advance up to 500 steps, or until halt fires
    for sim in prog.get_simulation_results(): # list[TLMSimulation]
        polycubes = sim.get_polycubes()
# __exit__ releases the underlying C++ group
```

`libtlm.does_not_match_topology(topology)` — returns a halt-condition
callable. **Currently a placeholder that always returns `False`** (per its
own docstring) — doesn't actually check anything against the topology yet.

## `TLMHistoryRecord` (C++ `HistoricalRecord`)

`stepCount() -> int`, `numPolycubes() -> int`, `getPolycubes() -> list[TLMPolycube]`,
`numCubeInstances() -> int`, `energy() -> float`.
VMMC-only, else `-1`/`None`: `accepted_moves() -> int`, `rejected_moves() -> int`,
`acceptance_ratio() -> float | None` (cumulative counts/ratio at this timepoint).
Pickle supported and backward-compatible (older 3-element pickles still load,
with accepted/rejected defaulting to -1).

## `TLMPolycube` (C++ `PolycubeData`)

`getID() -> int`, `numCubes() -> int`, `getCubes() -> list[TLMCubeData]`.
`to_runtime(params: TLMParameters) -> TLMRuntimePolycube` — converts to the
richer runtime representation (see below); prefer this over
`pypatchy.design.tlm_runner.toPolycube()` when scanning many polycubes and
you only need positions/bonds/topology, since it avoids building a
networkx-backed `PolycubeStructure`. Pickle supported.

## `TLMRuntimePolycube` (C++ `Polycube`, via `TLMPolycube.to_runtime()`)

`num_cubes() -> int`, `position_matrix() -> np.ndarray` (shape `(3, N)`, int32;
rows are x/y/z, columns are cubes), `cube_bond_data() -> list[tuple[int,int]]`
(one `(type_id, num_bonds)` pair per cube), `structure() -> TLMTopology`.

## `TLMCubeData` (C++ `CubeData`)

`getUID() -> int`, `getTypeID() -> int`, `getPosition() -> np.ndarray` (int3),
`getRotation() -> np.ndarray` (`[x,y,z,w]` scalar-last quaternion),
`getState() -> list[bool]`. Pickle supported.

## `TLMTopology` (C++ `Structure`)

`TLMTopology(bindings: set[tuple[tuple[int, DirIdx], tuple[int, DirIdx]]], monomer_attrs: dict[int, int])`.
`.homomorphism(other: TLMTopology) -> bool` — True if `self` maps injectively
into `other` under some cube-rotation symmetry (subgraph homomorphism).

## `OffTargetCandidateGenerator`

`OffTargetCandidateGenerator(targets: list[TLMTopology])`.
`has_next() -> bool`, `next() -> TLMTopology`.

## `DirIdx` enum

`INVALID_DIR_IDX=-1, LEFT=0, RIGHT=1, BOTTOM=2, TOP=3, BACK=4, FRONT=5, NUM_DIR_IDXS=6`
— values line up exactly with `polycubesRule.RULE_ORDER`'s index order (see
[[polycubes-rules]]), so no translation is needed between the two subsystems.

## `SimType` enum

C++ has `VMMC, LEGACY_VMMC, FINITE_SIZE, RANKED_VALENCE, VIRTUALLY_SEEDED_GROWTH`.
**The exposed Python enum (`libtlm.SimType`) only has the first four** —
`VIRTUALLY_SEEDED_GROWTH` is missing from the `py::enum_` binding even though
the model itself works via `model_type="virtually_seeded_growth"`. The enum
isn't accepted by any constructor anyway (model selection is always via the
`model_type` string) — treat it as informational only.

## Module-level free functions (legacy, kept for backwards compatibility)

- `setVerbose()` / `setNoVerbose()` — spdlog level.
- `read_tlm_params(file_path: str) -> TLMParameters`
- `isBoundedAndDeterministic(rule: str, torsion: bool, nTries: int) -> bool` —
  still takes an explicit legacy `torsion` bool (not migrated to the new scheme).
- `checkTopologyMatch(a: TLMTopology, b: TLMTopology) -> bool`
- `generateInteractionMatrix(rule: str, torsion: bool = False) -> dict[tuple[int,int,int], float]` —
  applies one global torsion switch to every color pair; see the main skill
  doc for why `tlm_runner.rule_to_interaction_matrix(rule)` is usually correct instead.
- `runSimulations(params: TLMParameters, numReplicas: int, logEnergyInterval: int) -> list[list[TLMHistoryRecord]]` —
  builds a `TLMSimulationGroup` internally (model via `params.model_type`/temperature
  fallback, same as everywhere else — not hardcoded to VMMC despite older docs claiming so).
- `runSimsAndSave(params, numReplicas, logEnergyInterval, write_to: str)` —
  same, then calls `dump_data(write_to)`. This is what
  `pypatchy.design.sat.tlm_util.run_tlm()` calls.
- `runPolycubes(rule: str, torsion: bool, numReps: int) -> list[TLMPolycube]` —
  legacy, still takes an explicit `torsion` bool (predates the interaction-matrix
  scheme, not reconciled with it). **Known bug**: hardcodes 1 cube per type
  internally — a rule needing more than one copy of any type to assemble will
  hit an assertion and abort. Don't use this for multi-copy rules; use
  `TLMSimulationGroup`/`PolycubesProgram` with real `cube_type_levels` instead.
