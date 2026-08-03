# CLAUDE.md

This file provides container-level guidance to Claude Code when working inside the `pypatchy` Docker image. It covers how the four bundled repos relate to each other; each repo also has its own `CLAUDE.md` with repo-specific detail that loads on-demand when you work inside it.

## What this image is

A ready-to-run environment for designing, compiling, and simulating DNA-nanotechnology "patchy particle" systems with oxDNA, built around **PyPatchy** as the top-level orchestration package. Built from `Dockerfile` in the `pypatchy` repo (`docker build -f Dockerfile .`, see `dock.sh`).

## The four repos (`/root/Github/`)

| Directory | GitHub repo | Branch | Role |
|---|---|---|---|
| `pypatchy` | `sulcgroup/pypatchy` | `unstable` | Top-level package. Defines polycube rules, converts them to oxDNA-compatible particles, builds/runs simulation ensembles, analyzes results. |
| `oxpy-utils` | `sulcgroup/oxpy-utils` | default | Simulation-running library (VMMC, FFS, umbrella sampling, sequence design) wrapping the `oxpy` Python bindings. Used by pypatchy's simulation-execution layer. |
| `oxDNA` | `lorenzo-rovigatti/oxDNA` | default | The C++ simulation engine itself, plus its Python bindings (`oxpy`) and analysis tools (`oat`). Compiled here with the `romano` and `rovigatti` patchy-particle interaction contrib models. |
| `polycubes` | `sulcgroup/polycubes-clone` | `nightmare` | **Naming mismatch, note this**: the GitHub repo and Josh's local dev checkout are both named `polycubes-clone`, but this Dockerfile clones it to `/root/Github/polycubes` (no `-clone` suffix). Provides the TLM (`libtlm.so`) library + Python bindings used by pypatchy's SAT-based crystal design subsystem (`pypatchy/design/`). |

**Do not confuse `polycubes` (above, the TLM library from `sulcgroup/polycubes-clone`) with `Akodiat/polycubes`** — a separate, unrelated polycubes implementation that pypatchy's own `CLAUDE.md` also mentions (as the source of an optional `libpolycubes.so` for its `design` package). Only `sulcgroup/polycubes-clone` is cloned into this image.

## Data flow across repos

```
polycubes rule string (pypatchy)
  → PLParticleSet (pypatchy, via polycube_rule_to_PL)
  → oxDNA input/topology files
  → oxpy / oxpy-utils drives the oxDNA engine (compiled from oxDNA repo)
  → trajectory + observable output
  → pypatchy.analysis pipeline
```

The polycubes/TLM library is consulted separately, during design (SAT solving in `pypatchy/design/sat/`), to search for rules that assemble a target crystal structure — it doesn't sit in the runtime simulation data flow above.

## `pypatchy/design/` — SAT-based crystal design subsystem

Given a target crystal topology, this subsystem searches for a polycube rule (particle-coloring scheme) whose local bindings reproduce that topology, then validates candidates by actually running polycube stochastic-assembly simulations via the `libtlm` library (compiled from the `polycubes` repo).

Key files (`pypatchy/design/sat/`):
- `design_rule.py` — `Polysat`: builds the `PolycubeSATProblem`, drives the solve loop (`find_solution`), tracks forbidden solutions, and owns `data_dir` (= `tlm_data_dir()`, see below) where each solution attempt gets its own `soln_<uuid>/` subfolder.
- `polycube_sat_problem.py` — low-level CNF/clause builder (colors, species, positions, rotations).
- `sat_problem.py` / `solution.py` — generic SAT wrapper and `PolysatSolution`; `exportScene()` writes to `output/SAT/`.
- `solve_steps.py` — post-processing pipeline run on a raw SAT model: `FilterSynonymsStep`, `NegativeDesignStep`, `RankedValenceStep`, `FiniteSizeStep`, `VMMCDesignStep`. The latter two are what actually invoke `libtlm` to run assembly simulations.
- `tlm_util.py` — `run_tlm()`: runs `libtlm.runSimsAndSave`, then exports each simulation's records as `sim_<i>.pickle` into the solution's `tlm_data_dir` subfolder.
- `multisolve.py` / `solve.py` / `solve_exec.py` — CLI/driver entry points.
- `solve_records_parser.py` — parses JSONL solve-event logs into pandas records for analysis.

Top-level `pypatchy/design/` files: `normalize_particle_set.py`, `reduce_rule.py`, `structure_math.py` (rule/topology math), `tlm_runner.py` (polycube↔TLM conversion helpers), `crystalloyd_finder.py`, `polycube_properties.py` (post-hoc rating of assembled structures), plus alternative design strategies: `design_amfa.py`, `design_yield.py`, `design_staged.py`, `pathway.py`, `substitution_solve.py`.

## Environment

Everything runs in a micromamba env named `pypatchy`, created from `pypatchy/pypatchy.yml`. It auto-activates on shell login (`.bashrc` runs `micromamba activate pypatchy`), so a plain `bash -l` gives you the right environment already active — no manual `conda activate` needed.

- `oxpy-utils` and `pypatchy` are both `pip install -e`'d into this env.
- `python-sat` is installed separately (needed for pypatchy's SAT design subsystem).

## Build order (see `Dockerfile` for exact commands)

1. Clone all four repos.
2. Create the `pypatchy` micromamba env from `pypatchy.yml`.
3. Compile oxDNA (`cmake -DPYTHON=ON`, `make install`, `make romano`, `make rovigatti`) — CPU-only in this image; CUDA is a known TODO (Dockerfile has commented-out `-DCUDA=ON`).
4. Compile the polycubes/TLM C++ library with Python bindings; `libtlm.so` is copied into the env's site-packages.
5. `pip install -e` oxpy-utils, then pypatchy.
6. Build the oxDNA analysis tools (`oat`) via `analysis/build.sh`.
7. `pip install python-sat`.

## Runtime config & data layout

- `/root/.pypatchy/` — pypatchy's working directory (created by the Dockerfile, normally via `python -m pypatchy.configure`):
  - `settings.cfg` `[ANALYSIS]` section — `simulation_data_dir = /root/patchy_run`, `tlm_data_dir = /root/data/tlm_data` (see below); `[SETUP]` selects `server_config = docker`.
  - `spec_files/server_configs/fr.json` — the `docker` server config: CPU backend, `oxdna_path = /root/Github/oxDNA/`, `patchy_format = "flavio"`.
  - `spec_files/input_files/`, `spec_files/observables/` — default input/observable JSON, copied from `pypatchy/spec_files/`.
  - `input/` — simulation inputs/targets.
  - `output/` — **metadata and logs only, never raw simulation data.** Specifically:
    - `output/SAT/` — solver-run artifacts: exported scene JSONs, forbidden-solutions log, multisolve status files.
    - `output/logs/` — general ensemble run logs; `output/logs/SAT/` holds per-solve-run `.log`/`.jsonl` event logs plus `last_settings_*.json` snapshots of TLM parameters.
    - `output/tlm/` — analytics output from `polycubeutil/ext/tlm_analytics.py`.

Raw simulation data lives outside `.pypatchy/output` entirely, in two separate trees depending on simulation type:
- `/root/patchy_run/` — raw oxDNA/patchy-particle trajectory and config data (`simulation_data_dir`), read/written by `PatchySimulationEnsemble`.
- `/root/data/tlm_data/` — raw polycube/TLM stochastic-assembly simulation data (`tlm_data_dir`). `Polysat` allocates one subfolder per SAT solution attempt, `soln_<uuid>/`, containing `problem_info.json`, `solution.pickle`, and the raw `sim_<i>.pickle` records written by `run_tlm()`.

- `/root/scripts/` and `/root/pypatchy.code-workspace` — a generated VS Code multi-root workspace covering `scripts/` and every repo under `Github/`, for Dev Containers / Remote-SSH attach.

## Gotchas

- No GPU/CUDA support baked in yet — CPU backend only.
- `polycubes-clone` upstream name vs. `polycubes` on-disk name in this image (see table above) — check which one a path reference means before assuming.
- The pypatchy repo's own `COPY . /root/Github/pypatchy` is commented out in the Dockerfile; the image always builds from a fresh `git clone` of the `unstable` branch, not local working-directory changes. Rebuild after pushing, not after just editing locally.
- As of writing, the Dockerfile actually hardcodes `tlm_data_dir = /home/josh/lab_data/tlm_data/raws` (a host-only path, left over from the "todo: make this settable... for systems with separate home and scratch drives" comment above it in the Dockerfile) — this path doesn't exist inside the container. It should read `/root/data/tlm_data` per the layout above; treat the current value as a known bug until the Dockerfile is fixed.
