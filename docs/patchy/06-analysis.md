# Analysis

pypatchy has two complementary ways to get results out of an ensemble:

- the **analysis pipeline**, for things computed at every timepoint of every simulation:
  cluster sizes, yields, energies. Results come back as pandas DataFrames, and are cached to
  disk.
- **scenes**, for looking at one configuration in detail: loading a conf as particles, computing
  energies with the real oxDNA interaction, splitting it into clusters, and writing those out.

Everything below assumes a loaded ensemble:

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble
with find_ensemble(name="3dX_T_sweep", date="2026-11-18") as e:
    ...
```

## The analysis pipeline

A pipeline is a directed graph. Each node is a **step** with a unique name, and each edge (a
**pipe**) sends one step's output into another. **Head** steps have no inputs: they read raw
files from the simulation folder (trajectory, energy file, observable output). Other steps
transform their parents' output.

### Example

Cluster sizes and yield of a "3D cross" target, from cluster data written by the
`PLClusterTopology` observable (adapted from
[`examples/setup_sims_programmatic_minimal.py`](../../examples/setup_sims_programmatic_minimal.py),
which also plots the results):

```python
from pypatchy.patchy.analysis_lib import (LoadEnergies, GraphsFromClusterTxt,
    ComputeClusterSizeData, ClassifyClusters, ComputeClusterYield, YIELD_KEY)
from pypatchy.analysis.analysis_data import TIMEPOINT_KEY
from pypatchy.polycubeutil.polycubesRule import rdir
from pypatchy.structure import Structure

obs = e.observables["clusters.txt"]          # observables are keyed by output file name
every = int(obs.print_every)

# target: one CT0 bonded to six CT1s
cross = Structure(bindings=[(0, i, i + 1, rdir(i)) for i in range(6)])

with e:
    e.add_analysis_steps(
        LoadEnergies("energies", output_tstep=int(e.get_param("print_energy_every"))),
        GraphsFromClusterTxt("graphs", source=obs),
        ComputeClusterSizeData("sizes", output_tstep=every),
        ClassifyClusters("classify", target=cross, output_tstep=every),
        ComputeClusterYield("yield", cutoff=0.75, overreach=False, output_tstep=every),
        ("graphs", "sizes"),          # pipes: (from, to)
        ("graphs", "classify"),
        ("classify", "yield"),
    )

    for sim in e.ensemble():
        df = e.get_data("yield", sim).get()                 # a DataFrame
        print(sim["T"], df[YIELD_KEY].iloc[-1] / e.get_param("num_assemblies"))
```

Steps and pipes are passed together in one call, in any order. Every non-head step needs a pipe
into it. The pipeline is saved in the ensemble's metadata, so next session it's already there
when you load the ensemble; you only need `get_data`.

`e.show_analysis_pipeline()` draws the graph, which is handy for spotting a missing pipe.

### Step catalog

All in `pypatchy.patchy.analysis_lib`.

**Heads** (read simulation files):

| Step | Constructor | Output |
|---|---|---|
| `GraphsFromClusterTxt` | `(name, source, output_tstep=None)` | For each timepoint, a list of `networkx.Graph`s, one per cluster. Nodes have a `particle_type`. Reads `PLClusterTopology` output (Romano model). `source` is the `Observable` object. |
| `GraphsFromPatchyBonds` | `(name, source, output_tstep=None)` | For each timepoint, one `MultiDiGraph` of all bonds. Reads `PatchyBonds` output (Rovigatti model). |
| `GraphsFromRaspberryPatchyBonds` | `(name, source, output_tstep=None)` | For each timepoint, a list of cluster graphs (clusters of 2 or more particles). Reads `RaspberryPatchyBonds` output. |
| `LoadEnergies` | `(name, *, output_tstep)` | DataFrame: `timepoint`, `pe`, `ke`, `te` (potential, kinetic, total energy per particle). Always recomputed. |
| `LoadParticlesTraj` | `(name, normalize_coords=True, input_tstep=1, output_tstep=None)` | The trajectory, as configurations. Set `input_tstep` to the `print_conf_interval` (as a keyword). Memory-hungry. |

**Steps** (transform data):

| Step | Constructor | Input → Output |
|---|---|---|
| `ComputeClusterSizeData` | `(name, input_tstep=None, output_tstep=None, minsize=0)` | cluster graphs → DataFrame: `timepoint`, `size_min`, `size_max`, `size_mean`, `size_median`, `size_stdev`. Clusters smaller than `minsize` are ignored. |
| `ClassifyClusters` | `(name, target, input_tstep=None, output_tstep=None)` | cluster graphs → DataFrame: `timepoint`, `clustercategory`, `sizeratio`, one row per cluster. See below. |
| `ClassifyPolycubeClusters` | `(name, target_name, ...)` | cluster graphs **and** a `LoadParticlesTraj` → same columns as `ClassifyClusters`, but matching particle types too. `target_name` is a `PolycubeStructure`. |
| `ComputeClusterYield` | `(name, cutoff, overreach=False, input_tstep=None, output_tstep=None)` | classified clusters → DataFrame: `timepoint`, `yield`. |
| `ComputeSpecGroupClusterYield` | `(name, aggregate_over, ...)` | Aggregates yields across an ensemble parameter. Experimental; probably gives NaNs. |

`MatchGraphToCrystal` and `BlobsFromClusters` exist but don't currently work.

For exact arguments, read the class's `__init__` in `analysis_lib.py`.

### Targets and yield

`ClassifyClusters` compares each cluster to a **target structure** and assigns a
`ClusterCategory`:

| Category | Meaning |
|---|---|
| `MATCH` | the cluster is the target |
| `SUBSET` | the cluster is a piece of the target |
| `SMALLER_NOT_SUB` | smaller than the target but not a piece of it (a misassembly) |
| `OVER` | bigger than the target |

`sizeratio` is the cluster's size divided by the target's size.

The comparison is graph (sub)isomorphism on the bond network only. **It ignores particle types
and binding directions**, so two structures with the same connectivity are indistinguishable.
Use `ClassifyPolycubeClusters` if types matter.

`ComputeClusterYield` throws out misassemblies and clusters with `sizeratio < cutoff` (and
`OVER` clusters, unless `overreach=True`), then **sums `sizeratio`** over the remaining
clusters at each timepoint. A perfect run therefore has yield `num_assemblies`, not 1. Divide
by `num_assemblies` for a fraction. With `cutoff=1.0` you count only complete structures; lower
cutoffs give partial credit to large fragments.

A target is a `pypatchy.structure.Structure` (or a subclass). Ways to make one:

```python
from pypatchy.structure import Structure, read_topology

# from bindings: (particle, face/patch index, particle, face/patch index)
cross = Structure(bindings=[(0, i, i + 1, rdir(i)) for i in range(6)])

# from a JSON file with a "bindings" list (relative paths resolve
# against ~/.pypatchy/input/topologies/)
target = read_topology("my_target.json")

# a polycube structure (also a valid target, and what ClassifyPolycubeClusters needs)
from pypatchy.polycubeutil.polycube_structure import load_polycube
target = load_polycube("my_polycube.json")       # relative to ~/.pypatchy/input/
```

> Earlier versions took a target *name* and loaded `~/.pypatchy/input/targets/<name>.json`.
> Passing a string now raises an error. Load the file with `read_topology` and pass the result.

### Getting data out

```python
data = e.get_data("yield", sim)        # PipelineData for one simulation
df = data.get()                         # DataFrame (or {timepoint: ...} dict for graph steps)
data.trange()                           # the timepoints
```

`sim` can be:

- one simulation (`e.get_simulation(T=0.01, duplicate=0)`) → one `PipelineData`;
- a **list** of simulations → a list of `PipelineData`, one each;
- **omitted**, or a partial tuple of parameter values → for DataFrame steps, one combined
  DataFrame, with a column for each ensemble parameter.

Timepoints are in simulation steps, under the column `TIMEPOINT_KEY` (`"timepoint"`). The
optional `time_steps` argument restricts the range (it must line up with the step's
`output_tstep`). With no `time_steps`, the last timepoint is not included.

### Time steps

Each step has an `input_tstep` (how often its input has data) and an `output_tstep` (how often
it produces output). Heads that read an observable take `input_tstep` from the observable's
`print_every`. Downstream steps inherit it from their parent. `output_tstep` must be a multiple
of `input_tstep`. Use a coarse `output_tstep` on graph heads for long runs; see
[Performance](#performance).

### Caching and recomputing

DataFrame results are cached in each simulation's folder as `<step name>.h5`, and loaded
instead of recomputed next time. Some consequences:

- **The cache is keyed only by step name.** If you change a step's parameters (a different
  `cutoff`, a different target) but keep its name, you'll get the old cached results. Give it a
  new name, or force a recompute.
- **Re-adding a step with an existing name does nothing**, and the old definition stays. Call
  `e.clear_pipeline()` and add the steps again.
- **A shorter result never overwrites a longer cached one.** So if you recompute after
  shortening a run, delete the `.h5` file.
- Graph and trajectory steps aren't currently cached to disk, so they're re-parsed in every new
  Python session. Only DataFrame steps are cached.

To force a step (and everything downstream of it) to recompute in this session:

```python
e.get_analysis_step("classify").force_recompute = True
```

and to be thorough, delete `<sim folder>/classify.h5` and the downstream `.h5` files.

`e.analysis_status()` shows which simulations have cached data for which steps, and
`e.missing_analysis_data("yield")` lists those without.

### Writing your own step

Subclass `AnalysisPipelineStep` (or `AnalysisPipelineHead` for one that reads files). Implement
`__call__`, which takes one argument per incoming pipe and returns a `PipelineData`, and
`get_output_data_type()`. Also set `load_cached_files`. For a step that outputs a DataFrame:

```python
import pandas as pd
from pypatchy.analysis.analysis_pipeline_step import AnalysisPipelineStep
from pypatchy.analysis.analysis_data import (PipelineDataType, PDPipelineData,
                                             ObjectPipelineData, TIMEPOINT_KEY,
                                             load_cached_pd_data)

class CountClusters(AnalysisPipelineStep):
    """Number of clusters of at least `minsize` particles at each timepoint."""

    load_cached_files = load_cached_pd_data

    def __init__(self, name, input_tstep=None, output_tstep=None, minsize=2):
        super().__init__(name, input_tstep, output_tstep)
        self.minsize = minsize

    def __call__(self, graphs: ObjectPipelineData) -> PDPipelineData:
        rows = {TIMEPOINT_KEY: [], "n_clusters": []}
        for t, gs in graphs.get().items():
            if t % self.output_tstep == 0:
                rows[TIMEPOINT_KEY].append(t)
                rows["n_clusters"].append(sum(len(g) >= self.minsize for g in gs))
        df = pd.DataFrame(rows)
        return PDPipelineData(df, df[TIMEPOINT_KEY].values)

    def get_output_data_type(self):
        return PipelineDataType.PIPELINE_DATATYPE_DATAFRAME
```

Then `e.add_analysis_steps(CountClusters("nclusters"), ("graphs", "nclusters"))`.

Output data types are `PIPELINE_DATATYPE_DATAFRAME` (use `PDPipelineData`),
`PIPELINE_DATATYPE_GRAPH` and `PIPELINE_DATATYPE_OBJECTS` (`ObjectPipelineData`, a
`{timepoint: value}` dict), and `PIPELINE_DATATYPE_RAWDATA`. If you plan to call
`show_analysis_pipeline()`, override `draw()` too, since the default one fails for some data
types.

> **Custom steps aren't restored when the ensemble is reloaded.** The metadata loader only
> knows the built-in step classes, so an ensemble whose saved pipeline includes a custom step
> raises `unknown step type` on load. Until that's fixed, add custom steps in each session, and
> remove them (`e.clear_pipeline()`, then re-add the built-in steps) before the metadata is
> saved.

## Energies without the pipeline

For a quick look at energies, skip the pipeline and use oxpy-utils' simulation object:

```python
df = e.ipy(sim).analysis.energy_df      # columns: time, U, K, Etot  (MD)
```

`e.ipy(sim)` returns the `oxpy_utils` `Simulation` for the sim's most recent stage (or pass a
stage); `e.ipy_all(sim)` returns one per stage. `energy_df` is read once and cached. Note that
`time` is in time units (steps × `dt`), not steps.

## Scenes: working with individual configurations

A **scene** (`PLPSimulation`, `pypatchy/patchy/pl/plscene.py`) is one configuration as Python
objects: the box, particle types, and each particle's position and orientation. Use one when
summary statistics aren't enough.

```python
scene = e.get_scene(sim)                     # last conf of the most recent stage
scene = e.get_scene(sim, stage, step=2e8)    # a specific step (a multiple of print_conf_interval)
```

`get_scene` also sets the temperature and attaches an **energy model**: a description of the
oxDNA interaction the simulation actually ran with (`interaction_type`, `plugin_search_path`,
model parameters). All energies are computed by that real plugin, through oxpy, so analysis
always agrees with the simulation.

### Energies and bonds

```python
scene.get_potential_energy()                 # per-particle potential energy
scene.interaction_energy(p1, p2)             # energy between two particles

g = scene.compute_scene_graph(bond_energy=0)  # networkx graph: edge where pair energy < bond_energy
clusters = list(scene.split_scene_by_clusters(bond_energy=-0.5, min_cluster_size=2))
```

- `bond_energy` is the threshold below which two particles count as bonded. More negative is
  stricter. This criterion uses the **total** pair energy (repulsion included), whereas
  `PLClusterTopology` uses only the patch term, so the two can disagree slightly near contact.
- `split_scene_by_clusters` yields a separate `PLPSimulation` for each cluster, re-indexed and
  in a box fitted to the cluster.
- Patch-level bonds (`compute_scene_multigraph`) are inferred from geometry within bonded pairs.
  If which patches are bonded is ambiguous (common with wide multidentate teeth), you'll see a
  `PatchBondAmbiguityWarning`.

For many energy queries on one scene, open a session once instead of letting each call open its
own:

```python
with scene.energy_session() as s:
    s.pair_energy(uid1, uid2)
    s.energy_with(uid, others)
    s.pair_energies()                   # all nonzero pairs
    s.total_energy()
```

**Only one energy session can be open at a time** (an oxpy limitation). Inside a
`with scene.energy_session()` block, don't call the convenience methods above
(`compute_scene_graph`, `get_potential_energy`, `interaction_energy`, …), because they each
open their own session and will raise `RuntimeError`.

### A scene from files pypatchy didn't set up

To load a configuration that isn't part of an ensemble, use a writer and attach the energy
model yourself (it must match the input file the run used):

```python
from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.oxpy_energy import OxpyEnergyModel

w = get_writer("lorenzo")
w.set_directory("/path/to/run")
ptypes = w.read_particle_types(topology="init.top", DPS_interaction_matrix_file="interactions.txt")
scene = w.read_scene("init.top", "last_conf.dat", ptypes)
scene.set_temperature(0.095)
scene.set_energy_model(OxpyEnergyModel("lorenzo", {
    "interaction_type": "DetailedPatchySwapInteraction",
    "plugin_search_path": "/path/to/oxDNA/contrib/rovigatti",
    "DPS_sigma_ss": 0.082, "DPS_normalise_patches": 0, "T": 0.095,
}))
```

### Saving clusters as their own simulations

```python
from pathlib import Path
for i, sub in enumerate(clusters):
    w = get_writer("flavio")
    w.set_directory(Path("/path/to/out") / f"cluster{i}")
    w.directory().mkdir(parents=True)
    w.write(sub, topology="init.top", conf_file="init.dat",
            particle_file="particles.txt", patchy_file="patches.txt")
```

## Plotting

`pypatchy.patchy.vis_lib` has seaborn/matplotlib helpers for common plots:

| Function | What it does |
|---|---|
| `plot_analysis_data(e, step_name, data_source_key="yield", other_spec=None, cols=None, rows=None, color=None, stroke=None, trange=None, norm=None)` | Facet grid of a DataFrame step's column over time. `rows`/`cols`/`color`/`stroke` each take an ensemble parameter name. Parameters not assigned to any of these are aggregated over, with the spread shown as a band; that's how `duplicate` gets averaged. `other_spec` fixes parameter values. `norm` divides by a parameter (e.g. `"num_assemblies"`) or a number. Returns a `FacetGrid`. |
| `plot_compare_ensembles(es, step_name, data_source_key, ...)` | The same, across several ensembles (over the simulations they share). |
| `plot_compare_analysis_outputs(e, sources, data_source_key, ...)` | Several steps' outputs in one grid. |
| `show_clusters(e, sim, step_name, timepoint=-1, figsize=4, min_cluster_size=2)` | Draws the cluster graphs from a graph step at one timepoint (default: the last). |
| `plot_total_graph(e, classify_step, ...)` | Total `sizeratio` over time. Takes the `ClassifyClusters` step **object**, not its name. |
| `plot_energy(e, ..., load_energy_step_name="load_energies", e_type="te", ax=None)` | Energy curves from a `LoadEnergies` step. |
| `plot_energies(e, sim, load_energy_step_name="load_energies")` | Potential, kinetic and total energy for one simulation. |

For example, yield fraction against time, one panel per temperature, averaged over duplicates:

```python
from pypatchy.patchy.vis_lib import plot_analysis_data
grid = plot_analysis_data(e, "yield", data_source_key="yield", cols="T", norm="num_assemblies")
grid.savefig("yield.png")
```

The `YieldCurveFigure`/`PolycubesFigure` classes and `plot_last_conf_cluster_size_histograms`
are currently broken. Writing your own matplotlib code on top of `e.get_data(...)` is always an
option, and is what the example script does.

## Performance

- **Memory.** Graph heads keep a graph for every retained timepoint, for every simulation, in
  memory. On large or long runs, `GraphsFromClusterTxt` over a whole ensemble can exhaust RAM.
  Use a coarse `output_tstep` on the head, analyze a subset of simulations at a time, and
  consider `e.set_nocache(True)` (which stops results being kept in memory; the disk cache still
  works). `LoadParticlesTraj` reads the **whole** trajectory before subsampling, so avoid it on
  long trajectories unless you need it.
- **Big analyses belong on a compute node,** not a laptop.
- **Parallelism.** Setting `e.metadata["parallel"] = N` makes `get_data` on a *list* of
  simulations use a pool of N processes. Results computed this way aren't kept in the parent's
  memory cache, but DataFrame results still reach the disk cache.
- **Inside Slurm jobs**, `get_data` queries `scontrol` for job info and raises if that fails.
