# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**PyPatchy** is a Python library for designing, configuring, running, and analyzing coarse-grained patchy particle simulations using the oxDNA simulation engine. It bridges:

- **Polycubes** – combinatorial 3D assembly rules defining how cube-like particles bind
- **PL format** – pypatchy's own flexible particle representation, convertible to any oxDNA patchy particle model
- **Simulation ensembles** – collections of parameterized simulations to run via oxDNA
- **Analysis pipelines** – tools to process simulation trajectories

## Setup and environment

Install dependencies via conda:
```bash
conda env create -f pypatchy.yml
pip install -e .
```

Required external packages (not in `pypatchy.yml`):
- `oxpy` and `oxDNA_analysis_tools` (oat): see https://lorenzo-rovigatti.github.io/oxDNA/install.html#python-bindings
- `oxpy_utils`: `pip install git+https://github.com/sulcgroup/oxpy-utils.git`

For the `design` package (SAT-based crystal design):
- `python -m pip install 'python-sat[pblib,aiger]'`
- Compiled `libpolycubes.so` (from https://github.com/Akodiat/polycubes/) must be placed in `bin/`
- The TLM library from https://github.com/sulcgroup/polycubes-clone/ (`cpp/`)

After installation, run the configure script to set up the working directory structure at `~/.pypatchy/`:
```bash
python -m pypatchy.configure
```

This sets up `~/.pypatchy/` — see [File & data layout](#file--data-layout) below for exactly what lives where.

## Running tests

```bash
pytest tests/
```

Run a single test file:
```bash
pytest tests/test_polycube_lib.py
```

Run a single test:
```bash
pytest tests/test_polycube_lib.py::test_read_string_rule
```

Test files marked as "good" (stable):
- `test_prerequisites.py` – checks oxpy, oat, oxpy_utils, and directory structure
- `test_polycube_lib.py` – polycube rule and structure basics

WIP test files: `test_patchy_io.py`, `test_analysis.py`, `test_solver_lib.py`, `test_ensembles_basic.py`, `test_ensembles_advanced.py`

Tests require external data downloaded from a server (may require ASU VPN). Test data lives in `tests/test_files/`.

## Architecture

### Core abstractions (`pypatchy/`)

- **`patchy_base_particle.py`** – Abstract base classes: `PatchyBaseParticleType`, `PatchyBaseParticle`, `BasePatchType`, `BaseParticleSet`. All particle types inherit from these.
- **`structure.py`** – `Structure`, `TypedStructure`, `Binding`, `StructuralHomomorphism` – graph-based structural representations of particle assemblies.
- **`util.py`** – Stateless helper functions (numpy math, filesystem paths, color utilities). Key path helpers: `get_local_dir()`, `get_input_dir()`, `get_output_dir()`, `simulation_run_dir()`.
- **`server_config.py`** – `PatchyServerConfig` dataclass; loads JSON from `~/.pypatchy/spec_files/server_configs/`. Controls oxDNA path, slurm settings, patchy interaction format.
- **`configure.py`** – Sets up `~/.pypatchy/` working directory structure.

### Polycubes subsystem (`pypatchy/polycubeutil/`)

Represents the abstract combinatorial design space:
- **`polycubesRule.py`** – `PolycubesRule` (rule string → set of cube types with patches) and `PolycubeRuleCubeType`. Rules are encoded as a compact string format like `"1:2||1:2||..."`. Patches are directional (6 faces: left/right/bottom/top/back/front) with integer color codes—positive and negative colors bind to each other.
- **`polycube_structure.py`** – `PolycubeStructure`: a connected assembly of cubes on a 3D integer grid, backed by a `networkx` graph.
- **`cube.py`** – `PolycubesStructureCube`: a single cube instance within a structure.

### PL subsystem (`pypatchy/patchy/pl/`)

"PL" is a legacy name that no longer stands for anything in particular. It's pypatchy's own Python abstraction for patchy particles, closest in shape to the format used by `oxDNA/contrib/romano`, but it has since evolved into a more flexible, independent representation. The goal is for `pypatchy.patchy.pl` objects to be convertible to *any* oxDNA patchy particle model via `pypatchy.patchy.pl.patchyio`, not just the romano format:
- **`plparticle.py`** – `PLPatchyParticle`: concrete particle class combining type and instance; holds `cm_pos`, orientation vectors `a1`/`a3`, velocity, angular momentum.
- **`plpatch.py`** – `PLPatch`: patch on a PL particle, with position, color, and optional torsion direction.
- **`plparticleset.py`** – `PLParticleSet`: collection of particle types; handles multidentate conversion settings.
- **`plscene.py`** – `PLPSimulation`: a simulation scene with particles placed in a box.
- **`patchyio.py`** – Readers/writers for PL format files (`.top`, `.dat`, `.mgl`, etc.) and conversion to/from oxDNA format. `get_writer()` returns the appropriate writer for a server config's format.
- **`plpatchylib.py`** – Bridge: `polycube_rule_to_PL()` converts a `PolycubesRule` to a `PLParticleSet`.
- **`plpotential.py`** – Potential energy function definitions for different PL interaction types.

### Simulation ensemble (`pypatchy/patchy/`)

- **`simulation_ensemble.py`** – `PatchySimulationEnsemble`: the central class for managing sets of related simulations. Handles parameter sweeps, ensemble construction, running via oxDNA, and loading results.
- **`simulation_specification.py`** – `PatchySimulation`, `ParamSet`: defines the parameter space for a simulation or ensemble.
- **`ensemble_parameter.py`** – Typed parameter definitions for ensemble parameters.
- **`stage.py`** – `Stage`: a single simulation stage (multi-stage simulations are supported).
- **`patchy_sim_observable.py`** – `PatchySimObservable`: observable definitions for extracting data during simulation.

### Analysis (`pypatchy/analysis/`)

- **`analysis_pipeline.py`** – `AnalysisPipeline`: composable pipeline of analysis steps applied to simulation data.
- **`analysis_pipeline_step.py`** – Individual analysis steps.
- **`analysis_data.py`** – `PDPipelineData`: pandas-backed data container for pipeline results.
- **`analyzable.py`** – `Analyzable` mixin for objects that can be analyzed.

### Design (`pypatchy/design/`)

Higher-level crystal design tools:
- **`sat/`** – SAT-based approach to finding polycube rules that assemble target structures. Uses PySAT. Requires the TLM library.
- **`solve.py`**, **`solve_exec.py`** – SAT solver execution.
- **`normalize_particle_set.py`**, **`reduce_rule.py`** – Utilities for simplifying/normalizing particle sets.
- **`structure_math.py`** – Mathematical utilities for crystal structure reasoning.

### Key data flow

```
PolycubesRule (string)
  → polycube_rule_to_PL()
  → PLParticleSet
  → PatchySimulationEnsemble (+ server config + parameters)
  → oxDNA simulation files
  → AnalysisPipeline
  → results
```

## File & data layout

Path helpers all live in `pypatchy/util.py`. Don't hardcode `~/.pypatchy` paths — use `get_local_dir()`, `get_input_dir()`, `get_output_dir()`, `get_log_dir()`, `simulation_run_dir()`, `tlm_data_dir()`.

### `~/.pypatchy/` — config, inputs, and metadata (never raw simulation data)

- `settings.cfg` — `[ANALYSIS]` section has `simulation_data_dir` (raw patchy-particle sim data root, required) and `tlm_data_dir` (raw polycube/TLM sim data root, optional — `tlm_data_dir()` raises `ValueError` if unset). `[SETUP]` has `server_config` — the *name* (no `.json`) of the active file under `spec_files/server_configs/`.
- `input/` (`get_input_dir()`) — `topologies/`, `targets/`, `tlm/`: input spec files and design targets.
- `spec_files/` — `server_configs/<name>.json` (looked up by name via `get_spec_json()`, must match `settings.cfg`'s `server_config` exactly), `input_files/`, `observables/`.
- `output/` (`get_output_dir()`) — **metadata and logs only**:
  - `output/logs/` (`get_log_dir()`) — ensemble run logs, named `log_<export_name>_<setup-date>_at<timestamp>.log`.
  - `output/SAT/` — SAT solver artifacts: exported scene JSONs, forbidden-solutions log, multisolve status files.
  - `output/logs/SAT/` — per-solve-run `.log`/`.jsonl` event logs, plus `last_settings_*.json` snapshots of TLM run parameters.
  - `output/tlm/` — analytics output from `polycubeutil/ext/tlm_analytics.py`.

### Raw simulation data — lives outside `~/.pypatchy/` entirely

- **Patchy-particle (oxDNA) simulations**, root = `simulation_data_dir` (`simulation_run_dir()`). Each `PatchySimulationEnsemble` gets a top-level directory `simulation_run_dir() / "<export_name>_<setup-date>"` (`ensemble.tld()`); individual runs live under `tld() / <param-value subfolder>[/ <stage name>]` (`ensemble.folder_path()`).
- **Polycube/TLM simulations**, root = `tlm_data_dir`. Each SAT solution attempt gets its own `tlm_data_dir/soln_<uuid>/`, containing `problem_info.json`, `solution.pickle`, and raw `sim_<i>.pickle` TLM simulation records (written by `design/sat/tlm_util.py::run_tlm()`).

### Docker container specifics

Inside the `pypatchy` Docker image (built from this repo's `Dockerfile`; see the container-wide `/root/CLAUDE.md`, sourced from `docker/ROOT_CLAUDE.md`), the above resolve concretely to:
- `~/.pypatchy` → `/root/.pypatchy`
- `simulation_data_dir` → `/root/patchy_run`
- `tlm_data_dir` → `/root/data/tlm_data`
- `oxdna_path` (in the active server config) → `/root/Github/oxDNA/`
- Sibling repos: `/root/Github/{pypatchy,oxpy-utils,oxDNA,polycubes}` — `polycubes` here is `sulcgroup/polycubes-clone` (cloned under a shortened name), not `Akodiat/polycubes`.

`settings.cfg` sets `server_config = docker`; the Dockerfile writes the same content to both `spec_files/server_configs/fr.json` and `spec_files/server_configs/docker.json` so that name resolves. The two files are identical duplicates for now — if the "docker" server config ever needs to diverge from "fr", edit `docker.json` directly rather than assuming they stay in sync.