# Installation and configuration

Getting a working patchy-particle setup means installing three things that have to agree with
each other:

1. **pypatchy** — this repository. It builds, runs, and analyzes simulations, but it does not
   contain a simulator.
2. **oxDNA**, compiled with Python bindings (`oxpy`) and with the patchy-particle plugin(s) you
   need. This is what actually runs the simulations, and pypatchy also uses it (through
   `oxpy`) to compute energies during analysis.
3. **oxpy-utils** — the lab's Python layer over `oxpy` (simulation objects, observables,
   batched runs). pypatchy imports it everywhere.

All three should be installed into the **same** Python environment.

If you just want a working environment and have Docker, the repository's `Dockerfile` builds
all of this (plus the polycubes/TLM tooling) in one go; it's also the most reliable reference
for the exact build steps. The manual route follows.

## 1. Clone the repositories

```bash
mkdir -p ~/git && cd ~/git
git clone git@github.com:sulcgroup/pypatchy.git
git clone git@github.com:sulcgroup/oxpy-utils.git
git clone git@github.com:lorenzo-rovigatti/oxDNA.git      # see "Which oxDNA?" below
```

You'll need SSH keys set up with GitHub
([GitHub docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).
If you can't see a repository, ask Petr to add you to the `sulcgroup` organization.

### Which oxDNA?

Which oxDNA checkout you need depends on the patchy model you'll run (see
[Models and file formats](03-models-and-formats.md)):

| You want to use | Plugin (`contrib/` dir) | Where it lives |
|---|---|---|
| Flavio Romano's torsional patchy model (`flavio` format) | `romano` | mainline [lorenzo-rovigatti/oxDNA](https://github.com/lorenzo-rovigatti/oxDNA) |
| Lorenzo Rovigatti's `DetailedPatchySwapInteraction` (`lorenzo` format; has a CUDA version) | `rovigatti` | mainline oxDNA |
| The raspberry model (`raspberry` format; patches + repulsion sites; has a CUDA version) | `evans` | Josh's fork, [JeDeveloper/oxDNA](https://github.com/JeDeveloper/oxDNA) — not (yet) in mainline |
| Romano model with allosteric patches (`josh_flavio` format) | `romano` (modified) | [sulcgroup/alloassembly](https://github.com/sulcgroup/alloassembly) |

The JeDeveloper fork tracks mainline and adds `contrib/evans`, so if you're unsure, cloning
the fork gets you every plugin except the allosteric one.

> Older lab documentation pointed to `Akodiat/oxDNA_torsion` for the Romano model. That's no
> longer necessary: `contrib/romano` in mainline oxDNA has the torsional patchy interaction.

## 2. Create the Python environment

pypatchy's conda dependencies are in `pypatchy.yml` (Python 3.12, conda-forge). The file has
no `name:` field, so give the environment a name explicitly. `mamba`/`micromamba` solve
environments much faster than `conda`; the commands are otherwise identical.

```bash
cd ~/git/pypatchy
mamba env create -n pypatchy -f pypatchy.yml
conda activate pypatchy
```

`requirements.txt` exists as a pip-only fallback, but the conda environment is what's
maintained.

## 3. Build oxDNA with Python bindings and the patchy plugins

Build oxDNA *with the `pypatchy` environment active*, so `oxpy` and `oxDNA_analysis_tools`
(OAT) get installed into it:

```bash
conda activate pypatchy
cd ~/git/oxDNA
mkdir build && cd build
cmake .. -DPython=ON -DOxpySystemInstall=On      # add -DCUDA=ON for GPU support
make install -j$(nproc)                          # oxDNA binary + installs oxpy and OAT
make romano -j$(nproc)                           # Romano (flavio) model + PLClusterTopology
make rovigatti -j$(nproc)                        # DetailedPatchySwap (lorenzo) model + PatchyBonds
make evans -j$(nproc)                            # raspberry model (JeDeveloper fork only)
```

Things that commonly go wrong here:

- **The plugins are not built by `make` or `make install`.** Every `contrib/` target is
  excluded from the default build; you must run `make romano` / `make rovigatti` /
  `make evans` explicitly. The resulting `.so` files are written into
  `contrib/<name>/` itself, which is the directory you'll later point `plugin_search_path`
  at.
- **`-DOxpySystemInstall=On` matters inside a conda environment.** Without it, `make install`
  runs `pip install --user`, which puts `oxpy` in `~/.local` rather than the environment. On a
  shared HPC Python you *don't* want this flag; inside your own conda env you do.
- **CMake picks the wrong Python.** Pass it explicitly:
  `cmake .. -DPython=ON -DPYTHON_EXECUTABLE=$(which python) -DPYTHON_INCLUDE_DIR=...`
- **OAT fails to build** (numpy header size mismatch, or similar): try
  `python -m pip install ./analysis --no-build-isolation` from the oxDNA root.
- **CUDA builds:** `-DCUDA=ON` compiles the CUDA versions of the plugins too
  (`CUDADetailedPatchySwapInteraction`, `CUDARaspberryPatchyInteraction`). The Romano
  `PatchyShapeInteraction` is CPU-only.

See the [oxDNA installation docs](https://lorenzo-rovigatti.github.io/oxDNA/install.html) for
the full set of CMake options.

## 4. Install oxpy-utils and pypatchy

```bash
conda activate pypatchy
pip install -e ~/git/oxpy-utils
pip install -e ~/git/pypatchy
```

Editable (`-e`) installs mean a `git pull` is all you need to update; you don't need to
reinstall.

## 5. Create the `~/.pypatchy` directory

```bash
python -m pypatchy.configure
```

This creates `~/.pypatchy/settings.cfg` (only if it doesn't already exist), creates
`input/targets/` and `output/logs/`, and copies the repository's `spec_files/` into
`~/.pypatchy/spec_files/`.

> **Run `configure` once.** It copies `spec_files/` unconditionally, so re-running it later
> overwrites any edits you've made to the default param set, observables, or any server
> config that shares a name with one in the repository. If you need to re-run it, back up
> `~/.pypatchy/spec_files/` first.

### What lives where

`~/.pypatchy/` holds configuration, inputs, and metadata. **Raw simulation data never goes
here** — it goes in `simulation_data_dir` (below), which should be on a large/scratch disk.

```
~/.pypatchy/
├── settings.cfg                 global settings (see below)
├── input/                       ensemble spec JSONs, ensemble metadata files, targets
│   ├── targets/                 target structures for cluster classification
│   └── topologies/
├── output/
│   └── logs/                    ensemble run logs: log_<export_name>_<date>_at<timestamp>.log
└── spec_files/
    ├── input_files/             default parameter sets ("presets"), e.g. default.json
    ├── observables/             premade observable specs, e.g. plclustertopology.json
    └── server_configs/          one JSON per machine/oxDNA build
```

When writing your own scripts, use the path helpers in `pypatchy.util` rather than hard-coding
these: `get_local_dir()`, `get_input_dir()`, `get_output_dir()`, `get_log_dir()`,
`simulation_run_dir()`.

## 6. Edit `settings.cfg`

`configure` writes a template with someone else's paths in it. Fix it:

```ini
[ANALYSIS]
simulation_data_dir = /scratch/<you>/patchysimulations

[SETUP]
server_config = my_machine
```

- **`simulation_data_dir`** — where pypatchy creates simulation folders and writes raw data and
  analysis caches. This gets large; on a cluster, use scratch. Create the directory yourself.
- **`server_config`** — the *name* (no `.json`) of a file in
  `~/.pypatchy/spec_files/server_configs/`. This is the default server config, used when an
  ensemble doesn't specify one.
- `tlm_data_dir` (optional, same section) is only used by the polycubes/TLM lattice
  simulations, not by oxDNA patchy simulations.

Anything else you find in an old `settings.cfg` is legacy and can be ignored.

## 7. Write a server config

A server config describes *one oxDNA build on one machine*: where oxDNA is, which plugin and
interaction to use, which file format that plugin reads, and how to launch jobs. Create
`~/.pypatchy/spec_files/server_configs/my_machine.json`. A minimal local CPU config for the
Rovigatti model:

```json
{
    "oxdna_path": "/home/you/git/oxDNA",
    "patchy_format": "lorenzo",
    "is_slurm": false,
    "input_file_params": {
        "backend": "CPU",
        "backend_precision": "mixed",
        "interaction_type": "DetailedPatchySwapInteraction",
        "plugin_search_path": "/home/you/git/oxDNA/contrib/rovigatti",
        "DPS_interaction_matrix_file": "interactions.txt",
        "DPS_normalise_patches": 0,
        "DPS_sigma_ss": 0.082
    }
}
```

The `patchy_format`, `interaction_type`, and `plugin_search_path` must all describe the same
model — picking pieces from different models doesn't raise an error at setup time, it produces
files the plugin can't read, or silently runs the wrong interaction. The
[models page](03-models-and-formats.md#which-pieces-go-together) has the table of matching
combinations.

All fields (`pypatchy/server_config.py`, `PatchyServerConfig`):

| Field | Default | Meaning |
|---|---|---|
| `oxdna_path` | *required* | Root of the oxDNA checkout to use. |
| `patchy_format` | *required* | Which pypatchy reader/writer to use: `flavio`, `josh_flavio`, `lorenzo`, `subhajit`, `raspberry`. |
| `input_file_params` | `{}` | Key/values written into every oxDNA input file. Put `backend`, `backend_precision`, `interaction_type`, `plugin_search_path`, and model-specific parameters here. |
| `is_slurm` | `false` | `true`: every simulation is submitted as its own `sbatch` job. `false`: simulations run directly in the current process — *including* when that process is itself a Slurm job. |
| `slurm_bash_flags` | `{}` | `#SBATCH` options for per-simulation jobs, e.g. `{"p": "general", "n": 1, "t": "7-00:00"}`. One-letter keys become `-p general`; longer keys become `--key="value"`. Only used when `is_slurm` is true. |
| `slurm_includes` | `[]` | Shell lines placed at the top of each sbatch script (`module load ...` and the like). |
| `cuda_mps` | `false` | Run all of an ensemble's simulations concurrently as worker processes in the current job, sharing GPUs through NVIDIA MPS. Also works with the CPU backend (concurrent processes, no MPS). See [Running](05-ensembles.md#running-simulations). |
| `max_sims_per_gpu` | MPS client limit | Cap on concurrent sims per GPU in `cuda_mps` mode. |
| `max_concurrent_sims` | number of CPUs | Cap on total concurrent sims in `cuda_mps` mode. |
| `absolute_paths` | `false` | Write absolute file paths into input files. |
| `no_oxpy_check` | `true` | Skip a pre-run oxpy sanity check of the generated configuration. |

The repository's `spec_files/server_configs/` has more examples, but treat them as templates:
several (`agave_*`) are for a cluster that no longer exists, and the `test_*` ones contain
placeholder paths.

> `interaction_potentials` appears in some older server configs. It's deprecated and ignored
> (with a `FutureWarning`): pypatchy now computes energies with the real oxDNA plugin via oxpy.
> Delete it when you see it.

## 8. Check the installation

```bash
conda activate pypatchy
pytest ~/git/pypatchy/tests/test_prerequisites.py
```

This checks that:

- `oxpy`, `oxDNA_analysis_tools`, and `oxpy_utils` import;
- the `~/.pypatchy` directories and your `simulation_data_dir` exist;
- for **every** server config in `~/.pypatchy/spec_files/server_configs/`, the compiled plugin
  `<plugin_search_path>/<interaction_type>.so` exists.

The last check fails for the template configs `configure` copied in (they point at other
people's paths). Delete or fix the configs you won't use, and the failure will then point
only at real problems with your own.

Then check that the plugin actually runs, by running a small ensemble (the
[quick start](README.md#quick-start) is a good one) or by running
[`examples/setup_sims_programmatic_minimal.py`](../../examples/setup_sims_programmatic_minimal.py)
after editing its server config.
