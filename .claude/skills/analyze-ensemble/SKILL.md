---
name: analyze-ensemble
description: Use when asked to write or debug a script that analyzes a pypatchy PatchySimulationEnsemble (loading an ensemble by name/date, checking simulation completion, running the analysis pipeline, or summarizing observable data like cluster sizes/yields). Skip for polycube-rule-only or SAT-design work with no ensemble/analysis pipeline involved.
---

# Analyzing a PyPatchy ensemble

This is the recipe for the recurring "load ensemble X, run analysis Y, summarize Z"
task, so it doesn't need to be re-derived from scratch by grepping
`simulation_ensemble.py` / `analysis_lib.py` / `analysis_data.py` every time.
For building and running a new ensemble in the first place, see
[[setup-run-ensemble]] (`.claude/skills/setup-run-ensemble/`) instead.

**Reference example**: [examples/analyze_mf_3x3x3_soln_044b6c5b.py](../../../examples/analyze_mf_3x3x3_soln_044b6c5b.py)
is a complete, working script built with this recipe. Prefer copying its shape
over inventing a new one.

## The canonical shape

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble
from pypatchy.analysis.analysis_data import TIMEPOINT_KEY

with find_ensemble(name=ENSEMBLE_NAME, date=ENSEMBLE_DATE) as e:
    sims = e.ensemble()                     # list[PatchySimulation] - every param combo

    # optional: which sims haven't reached their target step count yet
    incomplete = {s for s in sims if not e.sim_stage_done(s, e.sim_most_recent_stage(s))}

    e.add_analysis_steps(
        SomeAnalysisPipelineHead("step_a_name", e.observables["some_observable.txt"]),
        SomeAnalysisPipelineStep("step_b_name", ...),
        ("step_a_name", "step_b_name"),      # (origin, destination) edge - required per link
    )

    for sim in sims:
        data = e.get_data("step_b_name", sim).get()   # -> pandas DataFrame (usually)
        ...
```

Notes on this shape:
- `find_ensemble` is a context manager (`with ... as e:`) — entering/exiting handles
  writing the ensemble's metadata file. Always use `with`, don't call it bare.
- `find_ensemble(*args, **kwargs)` — the maintainer's own docstring calls its
  argument handling "a nightmare." The reliable form is keyword-only:
  `find_ensemble(name=..., date=...)`, with `date` as either a `datetime.datetime`
  or a `"%Y-%m-%d"` string.
- `e.add_analysis_steps(*args)` takes a flat mix of `AnalysisPipelineStep`/`Head`
  instances and `(origin_name, dest_name)` tuples that link them — every step
  after the first head needs an explicit edge tuple or it won't be wired in.
  Steps are referenced elsewhere by the `name` string passed to their constructor,
  not by variable name.
- `e.get_data(step, sim, time_steps=None)` — per its own docstring, "horrific
  monster of a class method." The `sim` argument's meaning changes with its type:
  a single `PatchySimulation`/descriptor returns one `PipelineData`; a **list**
  analyzes each independently and returns a list of outputs; a **tuple** of
  `ParameterValue`s is treated as a descriptor for one specific simulation. Don't
  mix these up. Call `.get()` on the result to get the underlying `pandas.DataFrame`
  (for `PDPipelineData`, the common case).
- Per-simulation parameter values: `sim[param_name]` (e.g. `sim["T"]`) or
  iterate `sim.param_vals` (each a `ParameterValue` with `.param_name` and
  `.value_name()`).
- `TIMEPOINT_KEY` (`pypatchy.analysis.analysis_data`) is the standard column name
  for the timestep/frame index in returned DataFrames — sort/filter on it rather
  than guessing a column name.
- `e.observables["name.txt"]` looks up a configured `PatchySimObservable` by its
  output filename — needed as the `source=` arg for the `*FromClusters`/`*FromPatchyBonds`
  step constructors.

## Analysis step catalog (`pypatchy/patchy/analysis_lib.py`)

Pipeline **heads** (start a pipeline branch, read raw sim output):
- `LoadParticlesTraj(name, ...)` — loads the raw particle trajectory.
- `LoadEnergies(step_name, ...)` — loads the energy log into the analysis
  *pipeline* (composable with downstream `AnalysisPipelineStep`s); timepoints
  there are in **time units**, not steps — convert before comparing to
  trajectory timepoints. For a quick look at energy/order-parameter data with
  no downstream pipeline steps needed, skip this and use `energy_df` instead
  (below) — it's less setup.
- `BlobsFromClusters(name, source, ...)` — reads the `clusters.txt`-style
  observable output into blob/cluster data.
- `GraphsFromClusterTxt(name, source, ...)` — reads `clusters.txt` observable
  output and builds `networkx` cluster graphs (most common starting point for
  cluster-size/yield analysis).
- `GraphsFromPatchyBonds(name, source, ...)` / `GraphsFromRaspberryPatchyBonds(name, source, ...)` —
  same idea, built from a patchy-bonds (or raspberry-model) observable instead.

Downstream **steps** (consume a head's/step's output):
- `ClassifyClusters(name, target, ...)` / `ClassifyPolycubeClusters(name, target_name, ...)` —
  compares cluster graphs against a target structure/polycube to classify matches.
  `target` is a `Structure`/`TypedStructure` (see [[structure-topology]] for
  what that comparison actually checks, including a real gotcha around
  rotation-direction mapping in homomorphism results).
- `MatchGraphToCrystal(name, target, ...)` — compares graphs to a target crystal structure.
- `ComputeClusterYield(name, cutoff, ...)` — yield from `ClassifyClusters`/`ClassifyPolycubeClusters`
  output, given a size/match cutoff.
- `ComputeClusterSizeData(name, input_tstep=None, ...)` — min/max/mean/median
  cluster size per timepoint (used in the reference example).
- `ComputeSpecGroupClusterYield(name, aggregate_over, ...)` — yield aggregated
  across a group of simulations sharing an `EnsembleParameter`.

When in doubt about a class's exact constructor args, read its docstring/`__init__`
in `analysis_lib.py` directly rather than guessing — this catalog is for picking
the *right* step, not for its full signature.

## Energy/order-parameter data shortcut: `energy_df` (skip the pipeline entirely)

For just the raw energy log (no cluster graphs, no pipeline wiring), it's
usually less work to bypass `add_analysis_steps`/`LoadEnergies` altogether
and go through the oxpy_utils `Simulation` wrapper instead:

```python
df = e.ipy(sim).analysis.energy_df.copy()   # e.ipy(sim) -> oxpy_utils Simulation
for pv in sim:                              # ParameterValue iteration - sim itself is iterable
    df[pv.param_name] = pv.value_name()
```

- `e.ipy(sim, stage=None) -> Simulation` wraps the sim's most recent stage (or
  a given `Stage`) as an `oxpy_utils.oxdna_simulation.Simulation`; `e.ipy_all(sim)`
  returns one per stage.
- `.analysis.energy_df` is a lazy, cached property — it reads/parses the energy
  file on first access and reuses it after. Columns depend on `sim_type`:
  `time, U, K, Etot` for MD, `time, U, acceptance_translational,
  acceptance_rotational, acceptance_volume` for MC. It raises for VMMC sims —
  those need `oxpy_utils.vmmc_umbrella.vmmc.VmmcAnalysis` instead.
- Good for quick per-sim time series (energy plots, order parameters) across
  a sweep; concatenate per-sim DataFrames (tagging each with its param values,
  as above) with `pd.concat(..., ignore_index=True)` to compare across the
  ensemble, e.g. with `groupby(["T", param_name])` for a multi-line plot.

## Loading a snapshot as a scene, and splitting/saving clusters

`PLPSimulation` ([pypatchy/patchy/pl/plscene.py](../../../pypatchy/patchy/pl/plscene.py))
is pypatchy's in-memory representation of one conf/snapshot — particles with
positions/orientations plus interaction potentials. It's what you want when the
analysis pipeline's DataFrame summaries aren't enough and you need to actually
inspect or export the geometry (e.g. pull out individual clusters as their own
sim files).

**Preferred, when working from an ensemble**: `e.get_scene(sim, stage=None, step=None)`.
This resolves the stage's top/traj files, reads particle types, and — unlike
loading a scene by hand — sets temperature and attaches the sim's oxDNA energy
model for you (via `e.get_energy_model(sim, stage)`: the run's own
`interaction_type`/plugin/params, evaluated through oxpy). `step=None` gets the
last conf; pass an explicit step (must be a multiple of `print_conf_interval`)
for an earlier snapshot.

```python
scene = e.get_scene(sim)          # PLPSimulation, energy model + T already set
clusters = sorted(
    scene.split_scene_by_clusters(bond_energy=-0.5, min_cluster_size=2),
    key=lambda s: s.num_particles(), reverse=True,
)
```

**Standalone** (a raw sim folder not managed by an ensemble — e.g. someone else's
run, or output you're inspecting outside pypatchy's directory layout): build the
scene by hand with a `patchyio` writer. This is more manual — nothing sets
the energy model or temperature for you, so do that explicitly. Picking the right
writer key (`"lorenzo"` below, or another) depends on which format the files
were actually written in — see [[patchy-model-formats]] if that's not obvious
from context:

```python
from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.polycubeutil.polycubesRule import PolycubesRule

get_writer("lorenzo").set_directory("path/to/raw/sim/folder")
plpset = polycube_rule_to_PL(PolycubesRule(rule_str="...")).to_multidentate(
    MultidentateConvertSettings(n_teeth=4, dental_radius=0.5))
scene = get_writer("lorenzo").read_scene(top_file="init.top", traj_file="last_conf.dat",
                                          particle_types=plpset)
scene.set_temperature(0.095)
from pypatchy.patchy.pl.oxpy_energy import OxpyEnergyModel
scene.set_energy_model(OxpyEnergyModel("lorenzo", {
    "interaction_type": "DetailedPatchySwapInteraction",       # same as the run's input file
    "plugin_search_path": "/root/Github/oxDNA/contrib/rovigatti",
    "DPS_sigma_ss": 0.082, "T": 0.095,                         # + any other model params the run used
}))
```

Use the same `interaction_type`, `plugin_search_path` and model params as the
run's oxDNA input file; energies then come from that plugin itself.

**Splitting into clusters**: `scene.split_scene_by_clusters(bond_energy=0.0, min_cluster_size=2)`
returns a generator of one `PLPSimulation` per connected component (raises if
the scene has no energy model — that's why it must be set before calling this). `bond_energy` is the cutoff *below* which two particles
count as bound (more negative = stricter/fewer bonds). Each returned sub-scene
is independently inboxed, re-numbered (uids reset to index within the cluster),
and box-sized to fit just that cluster.

**Saving clusters to disk**: use a `patchyio` writer per cluster — it can be a
different format than what you read with:

```python
from pathlib import Path
clusters_dir = Path("path/to/output/clusters")
for i, subscene in enumerate(clusters):
    w = get_writer("flavio")
    w.set_directory(clusters_dir / f"cluster{i}_flavio")
    w.directory().mkdir()
    w.write(subscene, topology="cluster.top", conf_file="cluster_init.conf",
            particle_file="particles.txt", patchy_file="patches.txt")
```

The `write(...)` kwarg names (`topology=`/`conf_file=`/... above) are
writer-specific — check the relevant writer class's `write()` signature in
[patchyio.py](../../../pypatchy/patchy/pl/patchyio.py) if targeting a different
format than `flavio`.

## Other useful `PatchySimulationEnsemble` methods

- `e.ensemble() -> list[PatchySimulation]` — every simulation (parameter
  combination) in the ensemble.
- `e.sim_most_recent_stage(sim) -> Stage`, `e.sim_stage_done(sim, stage) -> bool` —
  completion status; use before summarizing so partial runs are called out
  rather than silently treated as final.
- `e.missing_analysis_data(step)` — which sims still need this step computed.
- `e.show_analysis_pipeline()` — draws the pipeline graph (useful for debugging
  a step-linking mistake, not for scripts meant to run headless).

## When writing a new analysis script

1. Check `examples/` for an existing script analyzing a similar ensemble/observable
   before writing one from scratch — copy and adapt rather than starting fresh.
2. Only reach into `analysis_lib.py` internals when the catalog above doesn't
   cover the need; otherwise this recipe plus the catalog should be enough
   context to write the script without a broader exploration pass.
