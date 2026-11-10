# CLAUDE.md

This file provides container-level guidance to Claude Code when working inside the `pypatchy` Docker image. It covers how the bundled repos relate to each other; each repo also has its own `CLAUDE.md` with repo-specific detail that loads on-demand when you work inside it.

## What this image is

A ready-to-run environment for designing, compiling, and simulating DNA-nanotechnology "patchy particle" systems with oxDNA, built around **PyPatchy** as the top-level orchestration package. Built from `Dockerfile` in the `pypatchy` repo (`docker build -f Dockerfile .`, see `dock.sh`).

## The repos (`/root/Github/`)

| Directory | GitHub repo | Branch | Role |
|---|---|---|---|
| `pypatchy` | `sulcgroup/pypatchy` | `main` (or `dock.sh --pypatchy-branch`) | Top-level package. Defines polycube rules, converts them to oxDNA-compatible particles, builds/runs simulation ensembles, analyzes results. |
| `oxpy-utils` | `sulcgroup/oxpy-utils` | default | Simulation-running library (VMMC, FFS, umbrella sampling, sequence design) wrapping the `oxpy` Python bindings. Used by pypatchy's simulation-execution layer. |
| `oxDNA` | `JeDeveloper/oxDNA` | default | The C++ simulation engine itself, plus its Python bindings (`oxpy`) and analysis tools (`oat`). Compiled here with the `romano` and `rovigatti` patchy-particle interaction contrib models. |
| `tacoxDNA` | `lorenzo-rovigatti/tacoxDNA` | default | Cloned as a sibling because oxpy-utils' tacoxDNA bridge and its tests default to looking for it at `/root/Github/tacoxDNA`. |
| `hoomd-blue` | `glotzerlab/hoomd-blue` | tag `v7.2.0` | **Only in images built with `dock.sh --with-hoomd`** (tagged `:hoomd` by default). HOOMD-blue MD, built from source with GPU support (no MPI) and installed into the `pypatchy` env with `gsd`; used by pypatchy's HOOMD simulation engine (server config `"engine": "hoomd"`). The build directory is deleted after install. |

This image does **not** clone or build `sulcgroup/polycubes-clone` (the TLM/`libtlm` C++ lattice-MC engine) — that repo is private and not needed for the standard build. A couple of standalone example scripts (`examples/run_libtlm.py`, `examples/tlm_vmmc_physics_demo/`) reference `libtlm` directly and will fail with an `ImportError` unless you separately compile that repo yourself and have access to it; nothing else in pypatchy imports it eagerly. `pypatchy/structure.py`'s `to_tlm_topology()` does a lazy `import libtlm` inside the method body for the same reason — importing `pypatchy.structure` itself never requires `libtlm`.

## Data flow across repos

```
polycubes rule string (pypatchy)
  → PLParticleSet (pypatchy, via polycube_rule_to_PL)
  → oxDNA input/topology files
  → oxpy / oxpy-utils drives the oxDNA engine (compiled from oxDNA repo)
  → trajectory + observable output
  → pypatchy.analysis pipeline
```

## Environment

Everything runs in a micromamba env named `pypatchy`, created from `pypatchy/pypatchy.yml`. It auto-activates on shell login (`.bashrc` runs `micromamba activate pypatchy`), so a plain `bash -l` gives you the right environment already active — no manual `conda activate` needed.

- `oxpy-utils` and `pypatchy` are both `pip install -e`'d into this env.

## Build order (see `Dockerfile` for exact commands)

1. Clone the repos.
2. Create the `pypatchy` micromamba env from `pypatchy.yml`.
3. Compile oxDNA (`cmake -DPYTHON=ON`, `make install`, `make romano`, `make rovigatti`) — CPU-only in this image; CUDA is a known TODO (Dockerfile has commented-out `-DCUDA=ON`).
4. `pip install -e` oxpy-utils, then pypatchy.
5. Build the oxDNA analysis tools (`oat`) via `analysis/build.sh`.

## Runtime config & data layout

- `/root/.pypatchy/` — pypatchy's working directory (created by the Dockerfile, normally via `python -m pypatchy.configure`):
  - `settings.cfg` `[ANALYSIS]` section — `simulation_data_dir = /root/patchy_run`; `[SETUP]` selects `server_config = docker`.
  - `spec_files/server_configs/fr.json` — the `docker` server config: CPU backend, `oxdna_path = /root/Github/oxDNA/`, `patchy_format = "flavio"`.
  - `spec_files/input_files/`, `spec_files/observables/` — default input/observable JSON, copied from `pypatchy/spec_files/`.
  - `input/` — simulation inputs.
  - `output/` — **metadata and logs only, never raw simulation data.** `output/logs/` holds general ensemble run logs.

Raw simulation data lives outside `.pypatchy/output` entirely:
- `/root/patchy_run/` — raw oxDNA/patchy-particle trajectory and config data (`simulation_data_dir`), read/written by `PatchySimulationEnsemble`.

- `/root/scripts/` and `/root/pypatchy.code-workspace` — a generated VS Code multi-root workspace covering `scripts/` and every repo under `Github/`, for Dev Containers / Remote-SSH attach.

## Gotchas

- No GPU/CUDA support baked in yet — CPU backend only.
- The pypatchy repo's own `COPY . /root/Github/pypatchy` is commented out in the Dockerfile; the image always builds from a fresh `git clone` of the `main` branch, not local working-directory changes. Rebuild after pushing, not after just editing locally.
