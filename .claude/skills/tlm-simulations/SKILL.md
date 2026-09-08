---
name: tlm-simulations
description: Use when setting up, running, resuming, or reading results from a TLM/VMMC polycube simulation (libtlm - the C++ lattice Monte Carlo engine, separate from oxDNA/PatchySimulationEnsemble entirely). Covers TLMParameters/TLMSimulationGroup/PolycubesProgram, choosing a simulation model (VMMC/FiniteSize/RankedValence/LegacyVMMC/VirtuallySeededGrowth), staged runs, checkpoint/resume, and the pypatchy-side TLMRecords/tlm_data readers. This API changes frequently - verify anything surprising against the current pybind11 source before trusting an older explanation (including this one). Not for the oxDNA/PL simulation pipeline (see setup-run-ensemble/patchy-model-formats) or the SAT-based rule-design pipeline that calls into this one - see sat-crystal-design for that.
---

# TLM/VMMC polycube simulations (`libtlm`)

TLM ("polycubes-clone") is a **separate simulation engine from oxDNA** — a
C++ lattice Monte Carlo model of polycube assembly (cubes snap onto a 3D
integer grid, no continuous dynamics), exposed to Python via `libtlm`
(pybind11). It shares the [[polycubes-rules]] rule-string format and
`RULE_ORDER`/`DirIdx` direction convention with the oxDNA side, but nothing
else — no `PatchySimulationEnsemble`, no `PLPSimulation`, no oxDNA input
files. It's the physics engine [[sat-crystal-design]] uses to
test whether a candidate rule actually assembles the target (and *only* the
target) — this skill covers running/reading a TLM simulation directly; see
[[sat-crystal-design]] for how that pipeline's `RankedValenceStep`/
`VMMCDesignStep`/`VirtuallySeededGrowthDesignStep` drive it.

**This binding is under active development** (the C++ source
`~/git/polycubes-clone/cpp/src/PolycubesPython.cpp` grew from ~710 to ~1150
lines recently, changing several signatures along the way — e.g. `torsion`
used to be a top-level bool and no longer is). Treat every signature/behavior
claim below as something to spot-check against that file if it doesn't match
what you're seeing — see [references/libtlm-api.md](references/libtlm-api.md)
for the full current class-by-class listing, generated directly from that
source. Example scripts in this repo (`examples/run_libtlm.py`) and even
comments in the C++ source itself can be stale — one example's own comment
says "would be good to have a kwargs option" for `TLMParameters`, which
already exists by the time this was written.

## Building `libtlm`

From `~/git/polycubes-clone/cpp/`:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON \
         -DPYTHON_EXECUTABLE=/path/to/your/python
make polycubes_py
```
`libtlm.so` is copied automatically to that interpreter's site-packages.
Prerequisite: Eigen v3.4 installed via CMake (`cmake .. && make install` in
the extracted Eigen source) — extracting the headers alone isn't enough.
`nlohmann/json`, `spdlog`, `pybind11` are fetched automatically. Rebuild
(`make polycubes_py` again) after pulling C++ changes — a stale `.so` is a
common source of "this method doesn't exist" confusion.

## The core workflow

```python
import libtlm

params = libtlm.TLMParameters(
    rule="1|1|1|1|1|1_-1|-1|-1|-1|-1|-1",
    temperature=0.1,
    density=0.1,
    num_steps=int(1e7),
    cube_type_levels=[20, 20],
    model_type="vmmc",          # or "finite_size", "ranked_valence", "legacy_vmmc",
                                 # "virtually_seeded_growth" (needs seed_config - see below)
)

group = libtlm.TLMSimulationGroup(params, log_every=0)
group.init(n_replicas=4)
group.run()                     # or step manually: group.step_all(1000) in a loop

for i in range(group.num_sims):
    sim = group.get_sim(i)
    last = sim.last_record()    # TLMHistoryRecord
    polycubes = last.getPolycubes()   # list[TLMPolycube]
```

- **`model_type`** on `TLMParameters` selects the physics: leave it `""` and
  the choice falls back to temperature (`T > 0` → VMMC, `T == 0` → FiniteSize
  deterministic-greedy). Set it explicitly when you want `"ranked_valence"`
  or one of the others regardless of T. There is no `torsion` argument
  anywhere in the current API — see "Torsion moved into the interaction
  matrix" below, it's the single biggest difference from older docs/examples.
- **`libtlm.SimType`** (the enum) exists for introspection but the exposed
  Python enum only has `VMMC`/`LEGACY_VMMC`/`FINITE_SIZE`/`RANKED_VALENCE` —
  **`VIRTUALLY_SEEDED_GROWTH` is missing from it**, even though the C++ enum
  and `model_type="virtually_seeded_growth"` string both support it. Always
  select the model via the `model_type` *string* on `TLMParameters`, never by
  trying to pass a `SimType` value into a constructor — no exposed
  constructor actually takes a `SimType` (see next point).
- **Known-stale trap**: `PolycubesPython.cpp`'s own docstring for
  `PolycubesProgram` shows `libtlm.PolycubesProgram(params, 'RankedValence')`
  — this is wrong for the *current* binding, whose second positional
  argument is `n_replicas: int`, not a model-type string. Passing a string
  there will fail to cast. Set `params.model_type = "ranked_valence"` instead
  (as `examples/run_libtlm.py` actually does), and pass replica count as the
  int. Likewise `TLMSimulationGroup`'s second positional argument is
  `log_every: int`, not a `SimType` — model selection is always via
  `params.model_type`.

## Choosing a model

| `model_type` | What it does | Use it for |
|---|---|---|
| `"vmmc"` (or T>0, unset) | Probabilistic Virtual-Move MC — proposes pseudocluster moves, accepts/rejects via Metropolis at temperature T. See the "further reading" pointer at the bottom for the algorithm's internals. | Realistic stochastic assembly — the default for "does this rule actually assemble" questions. |
| `"finite_size"` (or T=0, unset) | Deterministic greedy: iterates cubes in random order, executes the first valid move found, no rejection. | Fast positive-control checks; can get stuck on one assembly path and produce false negatives (this is exactly why [[sat-crystal-design]]'s `VMMCDesignStep` exists as a supplement to greedy-only `RankedValenceStep`). |
| `"ranked_valence"` | A specific deterministic assembly heuristic distinct from plain FiniteSize (see [[sat-crystal-design]] for how it's used as a negative-design check). | Negative design / off-target checking. |
| `"legacy_vmmc"` | An older VMMC implementation kept alongside the current one. | Only if you specifically need to compare against or reproduce old-VMMC results — prefer `"vmmc"` otherwise. |
| `"virtually_seeded_growth"` | Grows from a fixed periodic seed pattern (a `TLMVirtualSeedConfig`) instead of a random free-cube pool — tests whether a rule extends a *given* partial lattice correctly rather than whether it nucleates one from scratch. | Testing lattice/seam consistency of a target crystal structure. Needs `seed_config=` on `TLMSimulationGroup`; doesn't support `dump_data()` or `init_from()` (raises `RuntimeError`). Building the `TLMVirtualSeedConfig` from a SAT solution is design-pipeline-specific (`tlm_runner.build_seed_pattern`/`to_tlm_seed_config`) — out of scope here. |

## Torsion moved into the interaction matrix — the sharpest edge here

There is **no `torsion: bool`** anywhere in the current `TLMParameters`
constructors (the "no torsion property" comment is in the binding source
itself). Torsion is now expressed per color pair as the **angle dimension**
of `interaction_matrix`: keys are `(color1, color2, angle)` triples, `angle`
∈ {0,1,2,3} (90°-increment relative twist between the two patches). Giving a
color pair the *same* strength at all 4 angles reproduces old
"non-torsional" behavior; a nonzero strength at only *one* angle reproduces
old "torsional" behavior.

If you don't supply `interaction_matrix` yourself, the kwargs constructor
auto-generates one via `generateInteractionMatrix(rule)` — but that applies
**one global torsion switch to every color pair**, which is wrong for a rule
where different color pairs are independently torsional or not (a "partial
torsion" rule). `pypatchy.design.tlm_runner.rule_to_interaction_matrix(rule)`
builds the matrix correctly instead, deriving each color pair's torsion-ness
from that specific rule's own patches (`patch.has_torsion()`) — **prefer this
over leaving `interaction_matrix` empty** whenever the rule might have mixed
torsion:
```python
from pypatchy.design.tlm_runner import rule_to_interaction_matrix
imat = rule_to_interaction_matrix(rule)   # rule: a PolycubesRule
params = libtlm.TLMParameters(rule=str(rule), temperature=..., density=...,
                              num_steps=..., cube_type_levels=[...],
                              interaction_matrix=imat)
```
It assumes every patch of a given color agrees on `has_torsion()` — true by
construction for SAT-solver output, not enforced for a hand-authored rule.

## Staged simulations

`libtlm.TLMStageInfo(cube_type_levels, max_steps, temperature=-1, density=-1, temperature_end=-1, linear_ramp=False)`
describes one stage; `-1` for `temperature`/`density` means "use the global
value." `temperature_end > 0` interpolates T across the stage (geometric by
default, `linear_ramp=True` for constant-increment). Build `TLMParameters`
with a `stages=[...]` list instead of `cube_type_levels`/`num_steps`:
```python
params = libtlm.TLMParameters(
    rule=rule_str, temperature=0.1, density=0.1,
    stages=[
        libtlm.TLMStageInfo(cube_type_levels=[20, 20], max_steps=int(5e6), temperature=0.15),
        libtlm.TLMStageInfo(cube_type_levels=[20, 20], max_steps=int(5e6), temperature=0.08),
    ],
)
```
This is a **different staging mechanism** from pypatchy's own oxDNA-side
`StagedAssemblyParam`/`StageInfoParam` (see [[setup-run-ensemble]]) — same
idea (a sequence of phases with different conditions), unrelated
implementation, don't confuse the two when a user says "staged simulation."

## Checkpoint / resume (`init_from`) — read the caveats before using

`group.init_from(records: list[TLMHistoryRecord])` seeds each new replica's
*starting particle configuration* from a previous run's record (not its step
trajectory — just where the particles were), each replica getting a fresh
RNG seed so they diverge from there. Only supported for VMMC-model groups.
Two sharp edges, both silent-failure risks rather than errors:
- The resuming group's `params` should copy `rule`/`interaction_matrix`/
  `depletes_types`/`density`/`model_type` **verbatim** from whatever produced
  the records (only `stages`/`seed` should differ) — a mismatched rule can
  silently reconstruct a physically-wrong state in release builds rather
  than raising.
- `cube_type_levels` in the new group's stage(s) should be the **original**
  per-type totals used by the source run, not an already-reduced count —
  `init_from` deducts cubes already bonded in the record from that reservoir
  itself, the same way normal stepping depletes it. Passing an
  already-reduced count double-subtracts.

Also on `TLMSimulation` (per-replica handle, not the group): `set_temperature(T)`
reassigns a live replica's temperature in place (for replica-exchange/parallel
tempering) — but only for single-stage (isothermal) sims; raises for a staged one.

## Reading results

- **`TLMHistoryRecord`** (one snapshot, from `sim.records()`/`sim.last_record()`):
  `stepCount()`, `numPolycubes()`, `getPolycubes()`, `numCubeInstances()`,
  `energy()`, and (VMMC only, else -1/None) `accepted_moves()`,
  `rejected_moves()`, `acceptance_ratio()`.
- **`TLMPolycube`**: `getID()`, `numCubes()`, `getCubes() -> list[TLMCubeData]`.
  For large-scale filtering, `polycube.to_runtime(params) -> TLMRuntimePolycube`
  gives `position_matrix()`/`cube_bond_data()`/`structure()` without the cost
  of building a full `PolycubeStructure` — prefer it over the Python-side
  conversion below when you're scanning many polycubes and only need
  positions/bond-counts/topology, not the richer pypatchy object.
- **Converting to pypatchy's own `PolycubeStructure`** (needed for anything
  that expects pypatchy's structure/graph machinery, e.g. matching against a
  target via [[polycubes-rules]]/`structure.py`):
  ```python
  from pypatchy.design.tlm_runner import toPolycube
  pc = toPolycube(rule, tlm_polycube)   # rule: PolycubesRule, tlm_polycube: TLMPolycube
  ```

## File formats and the pypatchy-side readers

Three ways TLM data ends up on disk, and what reads each:

1. **`TLMSimulationGroup.dump_data(path)`** — one `sim<i>.json` file per
   replica in `path` (`{"settings": ..., "records": [...]}`).
   `pypatchy.polycubeutil.tlm_record_set.TLMRecords(json_path)` reads this
   into **pure-Python** record objects (`pypatchy.polycubeutil.tlm_data` —
   see below) — no `libtlm` import required, so this path works for
   post-hoc analysis on a machine without the compiled extension.
2. **`TLMRecords.export(pickle_path)`** — pickles the real `libtlm.TLMParameters`
   object plus a stream of real `libtlm.TLMHistoryRecord` objects.
   `TLMRecords(pickle_path)` reads it back — but this path **requires
   `libtlm` to be importable** (it's unpickling the actual C++-bound classes).
   **`TLMParameters` pickles from before the torsion/interaction-matrix
   rework cannot be loaded** — the binding's own pickle `__setstate__`
   explicitly rejects tuples shorter than 9 elements. Don't assume an old
   `.pickle` from a previous libtlm build will still load; re-run or
   re-export if it doesn't.
3. **`history_dir=` streaming** (`TLMSimulationGroup(params, history_dir=...)`) —
   for very long/many-replica runs, streams each replica's snapshots to
   `<history_dir>/sim<i>_history.bin` (length-prefixed JSON records) instead
   of accumulating in RAM. Read it with
   `pypatchy.design.tlm_runner.read_history_bin(path) -> list[dict]`.

**`pypatchy.polycubeutil.tlm_data`** mirrors `TLMHistoryRecord`/`TLMPolycube`/
`TLMCubeData`'s method names exactly, built from JSON dicts instead of C++
objects — the module's own comment warns to `import tlm_data` and reference
`tlm_data.TLMHistoryRecord`, **not** `from tlm_data import TLMHistoryRecord`,
since the name collides with the real `libtlm.TLMHistoryRecord`.

**The canonical "run N replicas and cache them" driver**:
`pypatchy.design.sat.tlm_util.run_tlm(i, tlm_params, n_sims, save_to_dir, record_interval=None)`
runs via `libtlm.runSimsAndSave` into a temp dir, exports each replica to
`save_to_dir/sim_<idx>.pickle` (this is where the `~/.pypatchy` file-layout
convention in this repo's `CLAUDE.md` — `tlm_data_dir/soln_<uuid>/sim_<i>.pickle`
— comes from), and also snapshots the run's params to
`get_log_dir()/SAT/last_settings_<i>.json`. Two things worth knowing before
trusting its output blindly: it returns each replica's **last** record via
`max(records.records.keys())`, which the function's own comment flags as
"off by one: TODO FIX" — treat "last record" as approximately-but-maybe-not-exactly
the final step; and its default `record_interval` — `max(1e6, num_steps // 1e6)`
— evaluates to `1e6` for essentially any `num_steps` below `1e12`, so it's
rarely doing what "adapt to num_steps" sounds like. Pass `record_interval`
explicitly rather than relying on the default if you care about it.

`tlm_data_dir()` (`pypatchy/util.py`) is the configured root for all of this
(`~/.pypatchy/settings.cfg`'s `[ANALYSIS] tlm_data_dir` — raises `ValueError`
if unset, unlike `simulation_data_dir` which is required at setup time).

## Further reading

[references/libtlm-api.md](references/libtlm-api.md) — the full class list
(`TLMParameters` all four constructors, `TLMStageInfo`, `TLMSimulationGroup`,
`TLMSimulation`, `TLMHistoryRecord`, `TLMPolycube`, `TLMCubeData`,
`TLMRuntimePolycube`, `TLMTopology`, `OffTargetCandidateGenerator`,
`TLMVirtualSeedConfig`, and the legacy module-level functions), generated
directly from `PolycubesPython.cpp` — read it when this doc doesn't give you
an exact signature you need.

For the VMMC algorithm's internals (pseudocluster construction, free-face
bookkeeping, bond energy) rather than just how to call it, read
`~/git/polycubes-clone/cpp/src/VMMCPolycubesSimulation.cpp`/`.h` directly —
worth doing rather than trusting a paraphrase, given how much the binding
layer alone has changed recently.
