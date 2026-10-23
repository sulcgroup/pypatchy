---
name: setup-run-ensemble
description: Use when asked to set up, configure, or run a new pypatchy patchy-particle simulation (or sweep/ensemble of simulations) via oxDNA - defining particle types/rule, sweep parameters, observables, server config, and calling do_setup()/start_simulations(). Not for analyzing results of an already-run ensemble - see the analyze-ensemble skill for that.
---

# Setting up and running a patchy particle ensemble

This is the recipe for building a `PatchySimulationEnsemble` from scratch and
running it, so it doesn't need to be re-derived from scratch by grepping
`simulation_ensemble.py` every time.

**Reference examples** (prefer copying one of these over inventing a new shape):
- [examples/setup_sims_programmatic_minimal.py](../../../examples/setup_sims_programmatic_minimal.py) —
  smallest complete build → run → analyze → plot example. Start here.
- [examples/setup_sims_programmatic.py](../../../examples/setup_sims_programmatic.py) —
  building particle types from a `PolycubesRule` with manually-overridden
  interaction strengths, no analysis/run.
- [examples/run_ramped_octohedra.py](../../../examples/run_ramped_octohedra.py) —
  loading an existing ensemble by JSON config and running a multi-stage
  (temperature-ramped) protocol in parallel across sims.

Once it's running, see [[analyze-ensemble]] (`.claude/skills/analyze-ensemble/`)
for pulling results back out.

## Check `~/.pypatchy/spec_files/` before building anything from scratch

This directory is the same layout on any pypatchy installation and usually
holds most of what a new ensemble needs, so read it before writing a `cfg`
dict by hand.

**`spec_files/input_files/default.json`** — a baseline `ParamSet` covering most
common oxDNA/patchy input params: `sim_type`/`ensemble` (MD/NVT), `dt`,
`thermostat`, `steps`, `print_conf_interval`, `print_energy_every`,
`PATCHY_radius`, `PATCHY_alpha`, `use_torsion`, `density`, standard file names
(`topology`, `conf_file`, `lastconf_file`, `trajectory_file`, `energy_file`),
and more. **This loads automatically** — `build_ensemble`/`find_ensemble` calls
`cfg.get(DEFAULT_PARAM_SET_KEY, "default")`, so if `cfg` doesn't set
`DEFAULT_PARAM_SET_KEY` at all, `"default"` (this file) is applied as a matter
of course. Passing `DEFAULT_PARAM_SET_KEY: None` explicitly (as
[setup_sims_programmatic.py](../../../examples/setup_sims_programmatic.py) does)
opts *out* of it entirely — every param then has to come from `CONST_PARAMS_KEY`
or elsewhere. Prefer leaving `DEFAULT_PARAM_SET_KEY` unset (inheriting the
baseline) unless there's a specific reason to start bare — it means
`CONST_PARAMS_KEY` for a new ensemble usually only needs true overrides plus
genuinely ensemble-specific keys (`num_assemblies`, `interaction_potentials`,
`narrow_type`, ...), not a full restatement of every input param. Read the file
(or `get_param_set("default")` from `simulation_specification.py`) first so you
know what's already covered and what its defaults actually are (e.g. its
`steps`/`dt` may not fit your case and need overriding in `CONST_PARAMS_KEY`).

**`spec_files/server_configs/*.json`** — named `PatchyServerConfig`s, one per
machine/setup (different `oxdna_path`, `backend` CPU vs CUDA, `interaction_type`
plugin, `plugin_search_path`, `patchy_format`, `is_slurm`, ...). These are
**machine- and setup-specific** — picking the wrong one doesn't error, it just
quietly runs against the wrong oxDNA build/backend/plugin. Before building a
`PatchyServerConfig` inline: list this directory, read the candidates, and ask
the user which one to use (or whether to build a new one) — don't guess. A
plain string is enough once picked — `SERVER_SETTINGS_KEY` accepts the config's
name directly and resolves it via `load_server_settings()`:
```python
SERVER_SETTINGS_KEY: "home"   # loads spec_files/server_configs/home.json
```
`pypatchy.server_config.get_server_config()` returns whichever config is
currently marked active in `~/.pypatchy/settings.cfg`'s `[SETUP] server_config`
— a reasonable default to suggest, but still confirm with the user rather than
assuming it's right for this particular ensemble.

Each config's `oxdna_path` points at a *specific checkout* of oxDNA, and its
`plugin_search_path` points at one `contrib/<model>/` directory inside that
checkout (e.g. `contrib/romano`, `contrib/rovigatti`, `contrib/evans`) that has
actually been compiled — different checkouts/machines may have different
subsets of `contrib/` built. A config's `interaction_type`/`plugin_search_path`
only works if that plugin is actually built at that `oxdna_path`; see
[[patchy-model-formats]] for how this ties to `patchy_format` and observables.

**`spec_files/observables/*.json`** — pre-made `Observable` specs, loadable by
name with `observable_from_file("patchybonds")` (from `patchy_sim_observable.py`;
`.json` suffix optional) instead of constructing an `Observable`/`ObservableColumn`
by hand. Currently on disk: `patchybonds.json` and `plclustertopology.json`.
Not every observable used in practice has a file here — see
[[patchy-model-formats]] for which observable goes with which model/plugin,
and what to do when there's no premade json (e.g. the raspberry model's
`RaspberryPatchyBonds`).

## Patchy models and file formats

Which `patchy_format`/`interaction_type`/observable combination to use, and
what their files actually contain, is documented separately in
[[patchy-model-formats]] (`.claude/skills/patchy-model-formats/`) — read that
before building a `PatchyServerConfig` inline or hand-writing `CONST_PARAMS_KEY`
keys like `patchy_file`/`DPS_interaction_matrix_file` for a format you haven't
used before.

## The canonical shape

```python
import datetime
from pypatchy.patchy.simulation_ensemble import build_ensemble, EXPORT_NAME_KEY, \
    ENSEMBLE_SETUP_DATE, CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY, \
    PARTICLE_TYPE_LVLS_KEY, OBSERABLES_KEY
from pypatchy.server_config import PatchyServerConfig
from oxpy_utils.utils.observable import Observable, ObservableColumn

e = build_ensemble(
    cfg={
        EXPORT_NAME_KEY: "my_ensemble_name",
        "rule": "1|1|1|1|1|1_-1|||||",          # a PolycubesRule string ...
        PARTICLE_TYPE_LVLS_KEY: [1, 6],          # ... or see "Particle types" below
        CONST_PARAMS_KEY: {
            "num_assemblies": 8,
            "density": 0.3,
            "steps": 5e6,
            "print_conf_interval": 1e4,
            "print_energy_every": 1e4,
            "interaction_potentials": ["fr_excl_vol", "fr_patchy"],
            "narrow_type": 1,
            "patchy_file": "patches.txt",
            "particle_file": "particles.txt",
        },
        ENSEMBLE_PARAMS_KEY: [
            ("T", [0.01, 0.025]),
            ("duplicate", list(range(3))),       # repeats: 2 T's x 3 duplicates = 6 sims
        ],
        OBSERABLES_KEY: [Observable("clusters.txt", 1e4,
                                    ObservableColumn(name="PLClusterTopology", show_types=1))],
        SERVER_SETTINGS_KEY: PatchyServerConfig(
            name="my_ensemble_name_local",
            oxdna_path="/path/to/oxDNA",
            patchy_format="flavio",
            is_slurm=False,
        ),
    },
    mdt={ENSEMBLE_SETUP_DATE: datetime.datetime.now()},
)

with e:   # context manager auto-saves ensemble metadata on exit
    e.do_setup(clean_build="force")
    e.start_simulations()
```

Notes on this shape:
- `build_ensemble(cfg, mdt)` constructs an ensemble entirely in-process — no
  external JSON needed. `mdt` **must** contain `ENSEMBLE_SETUP_DATE` (a
  `datetime.datetime` or `"%Y-%m-%d"` string) or it raises
  `MissingEnembleSetupInfoError`; `cfg` **must** contain `EXPORT_NAME_KEY` for
  the same reason.
- Alternative to `build_ensemble`: `find_ensemble(cfg="my_ensemble_name")` loads
  the same `cfg` dict shape from a JSON file at `get_input_dir() / "my_ensemble_name.json"`
  (uses today's date unless `date=`/`sim_date=` is also passed). Once an ensemble
  has been set up once, re-load it for a later run/analysis with
  `find_ensemble(name=..., date=...)` as in the analyze-ensemble recipe — don't
  call `build_ensemble` again with the same export name/date, or its metadata
  file gets overwritten warnings.
- Every `*_KEY` above is a module-level string constant in `simulation_ensemble.py`
  (or `param_val_types.py` for `PARTICLE_TYPES_KEY`/`STAGES_KEY`) — pass the
  constant, not a hardcoded string, so a future key rename doesn't silently
  break the script. If you find an old ensemble-spec JSON with different key
  names than these constants (e.g. `num_steps`/`cube_type_levels` instead of
  `steps`/`particle_type_levels`), it's stale — don't copy it; use the constants.
- `with e:` isn't optional decoration — exiting the block is what writes the
  ensemble's metadata file (`dump_metadata()`). An ensemble built without ever
  entering/exiting a `with` still gets one metadata write from `build_ensemble`
  itself, but any params added after that (e.g. `e.add_params(...)`,
  `e.const_params[...] = ...`) need a `with e:` or an explicit `e.dump_metadata()`
  to persist.
- `e.do_setup(sims=None, stage=None, clean_build=False)` writes oxDNA input
  files. `clean_build="force"` wipes and rebuilds an existing stage directory;
  the default `False` leaves one alone (use this for resuming); avoid passing
  bare `True` — it prompts interactively for confirmation, which hangs a
  non-interactive script.
- `e.start_simulations(e=None, stage=None)` calls `do_setup` first if not
  already done, then runs each sim. **Local (non-slurm) runs execute in-process
  via oxpy and block until each sim finishes**, raising `RuntimeError` itself on
  failure — no separate join()/polling needed. Slurm runs submit via `sbatch`
  and return as soon as each job is queued (non-blocking).
- `e.is_ready(sims=None) -> bool` checks setup preconditions (e.g.
  `server_settings` is set) and `warnings.warn()`s per problem — worth calling
  before `start_simulations()` on a big/expensive sweep.

## Particle types: two ways to specify

1. **Rule string** (`"rule"` key) — a `PolycubesRule` string (see
   [[polycubes-rules]] for the string syntax itself), converted internally via
   `polycube_rule_to_PL()`. Pair it with `PARTICLE_TYPE_LVLS_KEY` (a list of
   particle counts per cube type, in the same order as the rule's cube types)
   if the counts aren't 1-each.
2. **Pre-built `PLParticleSet`** (`PARTICLE_TYPES_KEY` from `param_val_types.py`) —
   build one yourself, e.g. `polycube_rule_to_PL(PolycubesRule(rule_str=...))`,
   optionally `.to_multidentate(MultidentateConvertSettings(n_teeth=..., dental_radius=...))`,
   and mutate `particle_types.interaction_matrix()[(color_a, color_b)] = strength`
   to override specific bond strengths before passing it in. Use this over a
   rule string when you need custom per-patch interaction strengths or a
   multidentate conversion — [setup_sims_programmatic.py](../../../examples/setup_sims_programmatic.py)
   is the reference for this path; see [[patchy-model-formats]] for what
   multidentate conversion actually does and what `n_teeth`/`dental_radius`/etc mean.

## `CONST_PARAMS_KEY` — common keys

Shared (non-swept) simulation params, as a flat dict. Common, format-independent
ones seen in the examples: `num_assemblies`, `density`, `steps`,
`print_conf_interval`, `print_energy_every`, `narrow_type`,
`interaction_potentials` (list of names — see [[patchy-model-formats]]).
`patchy_file`/`particle_file` are specific to the `flavio`/`josh_flavio`
writers — other formats need different keys here instead (e.g. `lorenzo` needs
`DPS_interaction_matrix_file`), so check `get_writer(patchy_format).reqd_args()`
for the format actually in use rather than assuming these two; see
[[patchy-model-formats]] for the full per-format breakdown and what
`interaction_potentials` names are recognized.

`backend`, `backend_precision`, `interaction_type`, `plugin_search_path` would
normally live in the server config's `input_file_params` — but if you're
building a `PatchyServerConfig` inline with no `input_file_params` (as in the
minimal example), they can go in `CONST_PARAMS_KEY` instead; both end up in the
same oxDNA input file.

## `ENSEMBLE_PARAMS_KEY` — the sweep

A list of `(param_name, [values])` tuples; the ensemble is the full cross
product. `"duplicate"` is the conventional param name for running N identical
repeats at each other param combination (not a special-cased keyword, just a
naming convention the examples use) — pair it with `e.sim_get_param(sim, "duplicate")`
when plotting/labeling per-repeat lines.

## `SERVER_SETTINGS_KEY` — `PatchyServerConfig`

See "Check `~/.pypatchy/spec_files/`" above first — usually this should be a
string naming an existing config rather than a `PatchyServerConfig` built
inline. If building one inline (e.g. no existing config fits, and the user has
confirmed that), required fields are `name`, `oxdna_path`, `patchy_format`
(must be a key `get_writer()` recognizes, e.g. `"flavio"`/`"lorenzo"`). Notable
optional fields: `is_slurm` (despite the name, `False` does **not** mean
"doesn't run on a slurm cluster" — plenty of ensembles with `is_slurm=False`
still run inside a slurm allocation, just as one job running all sims directly
via oxpy, rather than each sim submitting its own nested `sbatch` job; see
`submits_per_sim_slurm_job()` if that distinction matters), `cuda_mps`,
`input_file_params`, `absolute_paths`.

`cuda_mps: true` makes `start_simulations()` run *all* of the ensemble's sims at once as concurrent
worker processes in the current job (via oxpy_utils's `SimulationManager`), spread evenly over the
GPUs and sharing each one through an NVIDIA MPS daemon it starts and stops itself - use it when
there are many more sims than GPUs and each is small. It also works with `backend: CPU`
(concurrent, no GPUs/MPS). Two optional limits go in the same config: `max_sims_per_gpu` (default:
the MPS client limit, 48) and `max_concurrent_sims` (default: CPUs available). `start_simulations()`
sets up and runs exactly what the non-batched path would (finished sims are skipped, so an
interrupted ensemble can simply be re-run), lets the remaining sims finish if one fails, then raises
`RuntimeError` listing those that didn't complete.

## Observables (`OBSERABLES_KEY`)

A list of `oxpy_utils.utils.observable.Observable(filename, print_every, *ObservableColumn(...))`.
Common columns seen in the examples: `ObservableColumn(name="PLClusterTopology", show_types=1)`
(writes cluster membership/adjacency — pairs with `GraphsFromClusterTxt` in
[[analyze-ensemble]]) and `ObservableColumn(name="PatchyBonds", show_types=1)`
(pairs with `GraphsFromPatchyBonds`).

## Multi-stage / staged-assembly protocols (e.g. temperature ramps)

For a protocol that changes params partway through a run (e.g. ramping T down
over time, or adding particles at a specific stage), use `StagedAssemblyParam`
(`param_val_types.py`) instead of a single scalar param:

```python
from pypatchy.patchy.param_val_types import StagedAssemblyParam, StageInfoParam
from pypatchy.patchy.particle_adders import RandParticleAdder

stages = {
    "stage1": StageInfoParam("stage1", t=0, T=0.0725,
                             add_method=RandParticleAdder({ptype.name(): count
                                                            for ptype, count in ...})),
    "stage2": StageInfoParam("stage2", t=1e9, T=0.07),
    # ...
}
e.add_params(STAGES_KEY=StagedAssemblyParam(staging_val_name="default", staging_info=stages))
e.dump_metadata()   # persist immediately if not inside a `with e:` block
```

- `StageInfoParam(stage_name, t=..., T=..., add_method=None, ...)` — `t` is the
  stage's start time (in steps); only the first stage typically needs
  `add_method` (an initial particle-placement strategy — `RandParticleAdder`
  places particles at random with no overlap-checked structure); later stages
  usually just change params like `T`.
- Run stage-by-stage rather than a single `start_simulations()` call:
  ```python
  def run_single_sim(sim):
      while e.sim_get_next_stage(sim) is not None:
          e.start_simulation(sim)
  ```
  Since a local `start_simulation()` call blocks in-process, run multiple sims'
  stage-loops concurrently with `concurrent.futures.ThreadPoolExecutor` (as in
  [run_ramped_octohedra.py](../../../examples/run_ramped_octohedra.py)) rather
  than looping sims serially, if you actually want them to run in parallel.
