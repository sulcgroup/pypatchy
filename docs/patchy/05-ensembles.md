# Setting up and running ensembles

A group of simulations that differ only in a few parameters is called an **ensemble**, which
annoys thermodynamicists. In pypatchy, the ensemble (`PatchySimulationEnsemble`) is the unit
you set up, run, and analyze; individual simulations are just points in its parameter space.

An ensemble is defined by:

- an **export name** (e.g. `"3dX_T_sweep"`) and a **setup date**. Together they identify the
  ensemble: setting the same spec up on two different days gives you two separate ensembles;
- the **particle types** (see [Particle sets](04-particles.md));
- **constant parameters**, shared by every simulation;
- **ensemble parameters**, each with a list of values. The simulations are the Cartesian
  product of these lists;
- optionally, **observables** for oxDNA to write during the run, a **server config** naming
  which oxDNA build to run on, and a multi-stage **staging** protocol.

## Two ways to define an ensemble

### As a JSON spec file

Put a JSON file in `~/.pypatchy/input/`, for example `~/.pypatchy/input/3dX_T_sweep.json`:

```json
{
    "export_name": "3dX_T_sweep",
    "rule": "1|1|1|1|1|1_-1|||||",
    "server_settings": "my_machine",
    "const_params": {
        "num_assemblies": 8,
        "particle_type_levels": {"CT0": 1, "CT1": 6},
        "density": 0.1,
        "steps": 5e7,
        "print_conf_interval": 1e5,
        "print_energy_every": 1e5,
        "narrow_type": 0
    },
    "ensemble_params": [
        ["T", [0.01, 0.025, 0.05]],
        ["duplicate", [0, 1, 2]]
    ],
    "observables": [
        {"name": "clusters.txt", "print_every": 1e5,
         "cols": [{"type": "PLClusterTopology", "show_types": 1}]}
    ]
}
```

Then create and run it:

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble

with find_ensemble(cfg="3dX_T_sweep") as e:
    e.do_setup(clean_build="force")
    e.start_simulations()
```

### In Python

`build_ensemble(cfg, mdt)` takes the same structure as a dict, so nothing has to be written to
disk first. Each spec key has a module-level constant in `simulation_ensemble.py`; using the
constants protects your script against key renames:

```python
import datetime
from pypatchy.patchy.simulation_ensemble import (build_ensemble, EXPORT_NAME_KEY,
    ENSEMBLE_SETUP_DATE, CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY,
    PARTICLE_TYPE_LVLS_KEY, OBSERABLES_KEY)
from oxpy_utils.utils.observable import Observable, ObservableColumn

e = build_ensemble(
    cfg={
        EXPORT_NAME_KEY: "3dX_T_sweep",
        "rule": "1|1|1|1|1|1_-1|||||",
        SERVER_SETTINGS_KEY: "my_machine",
        CONST_PARAMS_KEY: {
            "num_assemblies": 8,
            PARTICLE_TYPE_LVLS_KEY: [1, 6],     # list form: counts in rule order
            "density": 0.1,
            "steps": 5e7,
            "print_conf_interval": 1e5,
            "print_energy_every": 1e5,
            "narrow_type": 0,
        },
        ENSEMBLE_PARAMS_KEY: [
            ("T", [0.01, 0.025, 0.05]),
            ("duplicate", [0, 1, 2]),
        ],
        OBSERABLES_KEY: [
            Observable("clusters.txt", 1e5, ObservableColumn(name="PLClusterTopology", show_types=1)),
        ],
    },
    mdt={ENSEMBLE_SETUP_DATE: datetime.datetime.now()},
)
with e:
    e.do_setup(clean_build="force")
    e.start_simulations()
```

[`examples/setup_sims_programmatic_minimal.py`](../../examples/setup_sims_programmatic_minimal.py)
is a complete, runnable version of this that also analyzes and plots the results.

### Spec keys

| Key (constant) | JSON key | Required | Meaning |
|---|---|---|---|
| `EXPORT_NAME_KEY` | `export_name` | yes | Ensemble name. Used for the data directory and metadata file names. |
| `CONST_PARAMS_KEY` | `const_params` | yes (may be `{}`) | Parameters shared by all simulations. |
| `ENSEMBLE_PARAMS_KEY` | `ensemble_params` | yes | The sweep. List of `[name, [values]]`. |
| — | `rule` | one particle spec is needed | A polycubes rule string (or `PolycubesRule`). |
| — | `cube_types` | | Polycube cube types as rule JSON. |
| `PARTICLE_TYPES_KEY` | `particle_types` | | A `PLParticleSet`, or a dict describing particle files to load. |
| `DEFAULT_PARAM_SET_KEY` | `default_param_set` | no | Name of a preset in `spec_files/input_files/`. Default `"default"`; `null` for none. |
| `SERVER_SETTINGS_KEY` | `server_settings` (also `server_spec`, `server_config`) | no | Server config name, dict, or `PatchyServerConfig`. Default: the one named in `settings.cfg`. |
| `OBSERABLES_KEY` (sic) | `observables` | no | Observables oxDNA should write. |
| — | `randseed` | no | Seeds Python's and numpy's RNGs (used for particle placement). |

`ENSEMBLE_SETUP_DATE` goes in `mdt`, not `cfg`, and is required by `build_ensemble`.

> **Old specs.** Specs written for older pypatchy versions may use `num_steps`,
> `print_conf_every`, or `cube_type_levels`. The first two are no longer recognized and are
> **silently ignored**, so the simulation uses the default param set's values instead. Rename
> them to `steps` and `print_conf_interval`. (`cube_type_levels` happens to still work, for
> the reason described under [particle counts](#particle-counts).) The top-level `particles`
> key is no longer read.

## Parameters

### Where a parameter's value comes from

Every oxDNA input-file setting and every pypatchy setting is a parameter, resolved per
simulation. From highest to lowest priority:

1. the current **stage**'s parameters (for staged runs);
2. the simulation's **ensemble parameter** values;
3. **`const_params`**;
4. the **default param set** (`~/.pypatchy/spec_files/input_files/default.json` unless
   `default_param_set` says otherwise);
5. the **server config's `input_file_params`**.

The last two are in that order on purpose but it's easy to trip over: **a value in the default
param set overrides the same key in your server config.** If a server-specific setting doesn't
seem to take effect, check whether `default.json` also sets it.

### The default param set

`default.json` holds the parameters that rarely change: integrator and thermostat settings,
`dt`, output file names, `PATCHY_alpha`, `PATCHY_radius`, `use_torsion`, a default `density`
and `steps`, and so on. Read it before writing a spec, so you know which values you're
inheriting. The top-level keys and the keys under `"input"` are merged; the nesting is only for
readability. You can make your own presets (`spec_files/input_files/my_preset.json`) and select
one with `"default_param_set": "my_preset"`.

### Parameters you must provide

These aren't in the shipped default and must be in `const_params` or `ensemble_params`:

- **`T`** — temperature, in simulation units.
- **`num_assemblies`** — how many copies of the target's particle complement to put in the box.
- **particle counts** — see below.
- **the writer's required file names** — depends on `patchy_format`:
  `patchy_file` and `particle_file` for `flavio` and `josh_flavio`;
  `DPS_interaction_matrix_file` for `lorenzo`. These can live in the server config's
  `input_file_params` instead, which is usually tidier. See
  [Models and file formats](03-models-and-formats.md).
- **model parameters** your interaction needs (for example `narrow_type` for the Romano model).

`e.is_ready()` checks for the most important of these and warns about anything missing.

### Particle counts

pypatchy needs to know how many of each particle type make up **one** assembly. It multiplies
those counts by `num_assemblies` to get the number of particles in the box.

Counts are looked up as parameters named after the particle types (`CT0`, `CT1`, … for types
generated from a rule). The usual way to set them is a `particle_type_levels` group in
`const_params`:

```json
"particle_type_levels": {"CT0": 1, "CT1": 6}
```

or, equivalently, the list form, in the order of the particle types:

```json
"particle_type_levels": [1, 6]
```

How this works under the hood: any parameter whose value is a dict (with no `"type"` key) is a
**parameter group**, and its sub-keys are flattened into individual parameters. So what actually
matters is that parameters named `CT0` and `CT1` exist; the group name is only a label. That's
why the old `cube_type_levels` name still works. The list form is special-cased and only works
inside `const_params`.

To **sweep** the particle mix, put groups in `ensemble_params`, each with a `name` that becomes
the folder label:

```json
["particle_type_levels", [
    {"name": "lean", "CT0": 1, "CT1": 6},
    {"name": "rich", "CT0": 1, "CT1": 8}
]]
```

> The older nested form `{"name": "...", "value": {"particle_type_levels": {...}}}` shown in
> earlier lab documentation **no longer works**. It doesn't error, but the counts aren't found
> and fall back to `const_params`. The rule itself also can't currently be an ensemble
> parameter; use one ensemble per rule.

### Box size

The box is a cube with side length `(N / density)^(1/3)`, where `N` is the total particle
count. For staged runs, `N` is the count after **all** stages have added their particles, so
the box is sized for the final concentration. A stage can override this by setting `box_size`
or `density`.

### `duplicate`

`duplicate` is just an ordinary ensemble parameter with values like `[0, 1, 2]`. It gives you
independent replicas at each combination of the other parameters. pypatchy doesn't treat it
specially (it is even written into the oxDNA input file, where oxDNA ignores it), but the
analysis and plotting tools work well with it, since any parameter you don't plot is
aggregated over.

## Observables

Observables tell oxDNA to write extra output while it runs. For patchy simulations the
important ones record which particles are bonded, because that's what cluster and yield
analysis is built on. Which observable you can use depends on the plugin (see
[the concordance table](03-models-and-formats.md#which-pieces-go-together)).

The `observables` list accepts any one of these forms (don't mix forms within one list):

- **dicts**: `{"name": <output file>, "print_every": <steps>, "cols": [{"type": <observable type>, ...options}]}`;
- **`Observable` objects** from `oxpy_utils.utils.observable`:
  `Observable(<output file>, print_every, ObservableColumn(name=<observable type>, **options))`;
- **names of premade specs** in `~/.pypatchy/spec_files/observables/`, e.g.
  `"plclustertopology"`.

The naming is confusing. An `Observable`'s `name` is its **output file name**
(`"clusters.txt"`), while an `ObservableColumn`'s `name` is the **observable type**
(`"PLClusterTopology"`). In the dict form the observable type goes under `"type"`.

Analysis steps take the observable object as their data source, looked up with
`e.observables[...]`. For dicts and `Observable` objects the key is the output file name
(`e.observables["clusters.txt"]`). For premade specs it's the spec name
(`e.observables["plclustertopology"]`), but only in the session that created the ensemble.
After the ensemble is reloaded from metadata, it's the file name like the others. Also check the premade spec's `print_every` (1e7 in
the shipped files) against your run length; for short runs you'll want the dict form with a
smaller interval.

Premade specs shipped with pypatchy:

| Spec | Column type | Plugin it needs |
|---|---|---|
| `plclustertopology` | `PLClusterTopology` | `romano` |
| `patchbonds` | `PatchyBonds` | `rovigatti` |

There's no premade spec for the raspberry model; use
`Observable("clusters.txt", <interval>, ObservableColumn(name="RaspberryPatchyBonds", show_types=1))`.

> Observables cannot be combined with a server config that sets `absolute_paths: true`.

If you forgot an observable, `e.dnaAnalysis(observable, sim=None, stage=None)` can generate
its output afterwards from the saved trajectory (by running oxDNA's `DNAnalysis`), as long as
the trajectory was saved often enough.

## Loading an existing ensemble

Each ensemble has a metadata file in `~/.pypatchy/input/`, named
`<export_name>_<YYYY-MM-DD>_metadata.json`. It records the full spec (with defaults
resolved), the analysis pipeline, and run bookkeeping. To work with an ensemble again later:

```python
with find_ensemble(name="3dX_T_sweep", date="2026-11-18") as e:
    ...
```

`find_ensemble` accepts many argument combinations; these are the ones to use:

| Call | Behavior |
|---|---|
| `find_ensemble(cfg="spec_name")` | **Creates** an ensemble from `~/.pypatchy/input/spec_name.json`, dated today. If metadata for that ensemble already exists, it is **overwritten**. |
| `find_ensemble(name="export_name", date="YYYY-MM-DD")` | **Loads** the ensemble set up on that date. If there's no metadata, it falls back to creating one from `export_name.json`. |
| `find_ensemble(..., set_server="other_config")` | Overrides the server config. |

Two traps:

- **Always pass `date`** when loading. It defaults to *today*, so
  `find_ensemble(name="3dX_T_sweep")` on a later day doesn't find your ensemble: it creates a
  **new, empty** one from the spec.
- **Don't use `cfg=` on an ensemble that already has results**, because it rewrites the
  metadata file, including the saved analysis pipeline.

`list_simulation_ensembles()` (in `simulation_ensemble.py`) prints the spec files in the
input directory along with the matching data directories, which is useful when you've
forgotten a date.

### Saving changes

Metadata is written when the ensemble is built, and again whenever a `with e:` block exits.
If you change the ensemble outside a `with` block (`e.add_params(...)`,
`e.const_params["steps"] = ...`, adding analysis steps), call `e.dump_metadata()` yourself or
the change is lost. Use the `with` form unless you have a reason not to.

`e.add_params(name=value)` adds a new constant parameter, and **asserts if that name already
exists**. To change an existing value, assign to `e.const_params[name]`.

## Running simulations

```python
with find_ensemble(name="3dX_T_sweep", date="2026-11-18") as e:
    e.do_setup(clean_build="force")
    e.start_simulations()
```

`do_setup(sims=None, stage=None, clean_build=False)` creates each simulation's directory and
writes its oxDNA input file, topology, particle/patch files, initial configuration, and
observables. `clean_build` options:

- `"force"` wipes and rebuilds the stage directory, and also removes any later stages. Use
  this for a fresh setup.
- `False` (the default) leaves existing directories alone. Note this includes **half-built**
  ones, which then fail later with a missing-file error. If setup was interrupted, rebuild
  with `"force"`.
- `True` asks for confirmation interactively, which hangs a non-interactive job. Avoid it.

The initial configuration is generated by oxDNA's `confGenerator`, so
`<oxdna_path>/build/bin/confGenerator` must exist.

`start_simulations()` runs `do_setup()` first if needed, then runs every simulation. How it
runs them depends on the server config:

| Server config | What happens |
|---|---|
| `is_slurm: false`, `cuda_mps: false` | Each simulation runs **in the current process**, through oxpy, one after another. `start_simulations()` blocks until they're all done and raises `RuntimeError` if one doesn't complete. Fine for small runs, and for one big Slurm job that should run sims sequentially. |
| `cuda_mps: true` | All simulations run **concurrently** as worker processes in the current job, spread across the visible GPUs, which they share through an NVIDIA MPS daemon that pypatchy starts and stops. Already-finished simulations are skipped, so an interrupted ensemble can just be re-run. If some fail, the rest finish and a `RuntimeError` lists the failures. With `backend: CPU` it runs sims concurrently on CPU cores with no MPS. Limits: `max_sims_per_gpu`, `max_concurrent_sims`. |
| `is_slurm: true` | Each simulation gets a `slurm_script.sh` and is submitted with `sbatch`. `start_simulations()` returns as soon as the jobs are queued. |

**On an HPC cluster the recommended pattern is one Slurm job for the whole ensemble, with
`cuda_mps: true`**, rather than one job per simulation. Write a small Python script that loads
the ensemble and calls `start_simulations()`, and submit that script as a single GPU job. Older
GPUs (Pascal, e.g. GTX 1080) are limited to 16 MPS clients per GPU, so set
`max_sims_per_gpu: 16` there.

There's no automatic resubmission ("babysitter") for jobs that hit a wall-clock limit. For
`cuda_mps` runs, re-running `start_simulations()` resumes by skipping completed sims. A
simulation that was cut off partway through a stage isn't continued from where it stopped.

### Checking progress

- `e.sim_stage_done(sim, stage)` — whether a stage reached its final step.
- `e.sim_most_recent_stage(sim)` — the latest stage that has output.
- `e.time_length(sim)` — the step count of the last saved configuration (or, given a list of
  sims or nothing, the minimum across them).

> **Interrupted single-stage runs look finished.** Ensembles without an explicit staging
> protocol get one implicit stage, `"default"`, which is allowed to fall short of its target.
> As a result `sim_done()` and `start_simulations()` treat a run that was killed early as
> complete. To check that a run actually reached `steps`, use `sim_stage_done()` or compare
> `time_length(sim)` against `e.sim_get_param(sim, "steps")`.

## Staged assembly

A **staged** protocol changes conditions partway through a run: adding more particles,
changing temperature (annealing or ramping), and so on. Each stage runs as its own oxDNA
simulation, starting from the previous stage's last configuration.

Stages are given by a `staging` parameter. It can be in `const_params` (one protocol for every
simulation) or in `ensemble_params` (to compare protocols). From
`tests/test_files/ensemble_specs/example_staged.json`, which compares adding the caps early
vs. late:

```json
"ensemble_params": [
    ["staging", [
        {
            "name": "slow",
            "type": "staging",
            "stages": {
                "first_stage": {
                    "add_method": {"type": "random", "particles": {"CT0": 1}}
                },
                "second_stage": {
                    "t": 1e6,
                    "add_method": {"type": "random", "particles": {"CT1": 6}}
                }
            }
        },
        {
            "name": "cool",
            "type": "staging",
            "stages": {
                "first_stage": {
                    "add_method": {"type": "random", "particles": {"CT0": 1}}
                },
                "second_stage": {
                    "t": 5e6,
                    "T": 0.1,
                    "add_method": {"type": "random", "particles": {"CT1": 6}}
                }
            }
        }
    ]]
]
```

Each stage has:

- **`t`** — the step at which the stage starts (default 0). Stages are ordered by `t`. Only
  the first stage may have `t = 0`. A stage runs until the next stage's `t`; the last stage
  runs until the simulation's `steps`.
- **`add_method`** (optional) — particles to add at the start of the stage:
  - `{"type": "random", "particles": {"CT0": 1, ...}}` places particles at random positions.
    Counts are per assembly and are multiplied by `num_assemblies`.
  - `{"type": "polycube", ...}` places pre-assembled polycube structures (`FromPolycubeAdder`).
- **any other key** — a parameter override for that stage (`T`, `density`, `box_size`,
  any oxDNA input setting, …).

When there's a staging protocol, particles are only added by the stages' `add_method`s; the
`particle_type_levels` counts aren't used for placement. Don't stage `density` or
`num_assemblies` unless you mean to, since they change how the box and particle counts are
computed.

Stage 0 runs in the simulation folder itself; later stages run in subfolders named after the
stage.

### Building stages in Python

The same thing in Python, as a temperature ramp (adapted from
[`examples/run_ramped_octohedra.py`](../../examples/run_ramped_octohedra.py)):

```python
import numpy as np
from pypatchy.patchy.param_val_types import StagedAssemblyParam, StageInfoParam
from pypatchy.patchy.particle_adders import RandParticleAdder

temps = np.arange(0.0725, 0.0525, -0.0005)
times = np.linspace(0, e.get_param("steps"), num=len(temps), endpoint=False)
stages = {}
for i, (T, t) in enumerate(zip(temps, times)):
    name = f"stage{i + 1}"
    if i == 0:
        stages[name] = StageInfoParam(name, t=t, T=T,
                                      add_method=RandParticleAdder({"CT0": 1, "CT1": 6}))
    else:
        stages[name] = StageInfoParam(name, t=t, T=T)

with e:
    e.add_params(staging=StagedAssemblyParam(staging_info=stages, staging_val_name="ramp"))
```

### Running a staged ensemble

`start_simulations()` runs the next stage of each simulation. To run every stage to the end,
loop per simulation:

```python
from concurrent.futures import ThreadPoolExecutor

def run_all_stages(sim):
    while e.sim_get_next_stage(sim) is not None:
        e.start_simulation(sim)

with ThreadPoolExecutor() as pool:
    for f in [pool.submit(run_all_stages, sim) for sim in e.ensemble()]:
        f.result()
```

Since a local `start_simulation()` blocks, the thread pool is what lets several simulations
progress at once.

If a stage was interrupted partway through, `start_simulation()` logs `"<stage> incomplete!"`
and returns without running anything, and a bare `start_simulations()` raises
`IncompleteStageError`. Rebuild that stage with
`e.do_setup([sim], stage, clean_build="force")` and run it again. Note that `"force"` also
removes any later stages.

## Where the data goes

Each ensemble gets a directory under `simulation_data_dir`:

```
<simulation_data_dir>/<export_name>_<YYYY-MM-DD>/        e.tld()
```

Inside, there's one nested folder level per ensemble parameter, in the order they appear in
`ensemble_params`, named `<param>_<value>`. For named values (particle-count groups, staging
protocols), the `name` is used. For `ensemble_params` of `T` then `duplicate`:

```
3dX_T_sweep_2026-11-18/
├── T_0.01/
│   ├── duplicate_0/        ← one simulation: input, init.top, init.dat, trajectory.dat,
│   ├── duplicate_1/          last_conf.dat, energy.dat, observable output, analysis caches
│   └── duplicate_2/
├── T_0.025/
│   └── ...
```

`e.folder_path(sim)` gives a simulation's folder (pass a stage for a later stage's subfolder).

## Useful methods

```python
e.info()                        # print a summary of the ensemble
e.tld()                         # top-level data directory
e.ensemble()                    # list of all simulations (PatchySimulation objects)
e.get_simulation(T=0.01, duplicate=0)   # one simulation...
e.get_simulation(T=0.01)        # ...or a list, if any parameter is left unspecified
e.folder_path(sim)              # a simulation's data directory
e.sim_get_param(sim, "T")       # resolved value of any parameter for one simulation
e.get_param("steps")            # resolved constant parameter (const_params, then server)
sim["T"]                        # a simulation's own ensemble-parameter value
e.time_length(sim)              # steps reached by the last saved conf
e.sim_most_recent_stage(sim)
e.sim_stage_done(sim, stage)
e.is_ready()                    # pre-flight check; warns about missing parameters
e.get_scene(sim)                # load a conf as a PLPSimulation (see Analysis)
e.ipy(sim)                      # the oxpy_utils Simulation object for this sim
e.all_folders_exist()
e.dump_metadata()
```

## Gotchas, collected

- Loading without `date=` creates a new ensemble instead of loading yours.
- `cfg=` overwrites existing metadata.
- Changes made outside `with e:` need `e.dump_metadata()`.
- Old key names (`num_steps`, `print_conf_every`) are silently ignored.
- Default param set values beat server config `input_file_params`.
- `clean_build=True` prompts interactively. Use `"force"`.
- Interrupted unstaged runs are reported as done.
- `print_conf_interval` must be smaller than `steps`.
- `e.const_params["run_serial"]` (seen in older scripts) has no effect.
