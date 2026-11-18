# Patchy particle simulations with pypatchy

This is the user guide for designing, running, and analyzing patchy-particle simulations with
pypatchy and oxDNA. It replaces the "Patchy Particle Simulations and Data Formats" and "The
PyPatchy Python Package" pages on the old lab wiki.

pypatchy doesn't simulate anything itself. It builds simulation inputs for oxDNA's
patchy-particle plugins, runs them (locally, concurrently on GPUs, or through Slurm), keeps
track of parameter sweeps, and analyzes the results.

```
polycubes rule / particle set
        │
        ▼
PatchySimulationEnsemble  ◄── spec (JSON or Python) + server config + default params
        │  do_setup(), start_simulations()
        ▼
oxDNA + patchy plugin  ──►  trajectories, energies, bond observables
        │
        ▼
analysis pipeline / scenes  ──►  cluster sizes, yields, energies, plots, exported structures
```

## Contents

1. [Background: patchy particle models](01-background.md) — what the models compute: potentials,
   torsion and narrow types, multidentate patches, and which model to use.
2. [Installation and configuration](02-installation.md) — pypatchy, oxDNA with plugins,
   `~/.pypatchy`, `settings.cfg`, server configs.
3. [Models and file formats](03-models-and-formats.md) — which format goes with which plugin,
   what the files look like, reading/writing/converting them.
4. [Particle sets](04-particles.md) — polycubes rules, colors, interaction strengths, custom
   particle types, multidentate conversion.
5. [Setting up and running ensembles](05-ensembles.md) — specs, parameters, observables,
   running locally / on GPUs / on Slurm, staged assembly, where data goes.
6. [Analysis](06-analysis.md) — the analysis pipeline, targets and yield, scenes and energies,
   plotting.
7. [Visualizing configurations](07-visualization.md) — ppview, oxView, exports.
8. [Converting patchy particles to DNA origami](08-patchy-to-origami.md) — from a patchy
   structure to an oxDNA origami assembly and a staple order sheet.
9. [Known issues](known-issues.md) — things that are currently broken or surprising.

## Quick start

This assumes pypatchy, oxDNA (with `make romano`), and oxpy-utils are installed, and that
`python -m pypatchy.configure` has been run ([Installation](02-installation.md)).

**1. A server config** for the Romano model on your machine,
`~/.pypatchy/spec_files/server_configs/local_romano.json`:

```json
{
    "oxdna_path": "/home/you/git/oxDNA",
    "patchy_format": "flavio",
    "is_slurm": false,
    "input_file_params": {
        "backend": "CPU",
        "backend_precision": "double",
        "interaction_type": "PatchyShapeInteraction",
        "plugin_search_path": "/home/you/git/oxDNA/contrib/romano",
        "patchy_file": "patches.txt",
        "particle_file": "particles.txt"
    }
}
```

**2. An ensemble spec**, `~/.pypatchy/input/cross_quickstart.json`. It describes a "3D cross":
one particle with six `+1` patches and six caps with a single `−1` patch, at two temperatures,
three replicas each:

```json
{
    "export_name": "cross_quickstart",
    "rule": "1|1|1|1|1|1_-1|||||",
    "server_settings": "local_romano",
    "const_params": {
        "num_assemblies": 8,
        "particle_type_levels": {"CT0": 1, "CT1": 6},
        "density": 0.3,
        "steps": 5e6,
        "print_conf_interval": 1e4,
        "print_energy_every": 1e4,
        "narrow_type": 1
    },
    "ensemble_params": [
        ["T", [0.01, 0.025]],
        ["duplicate", [0, 1, 2]]
    ],
    "observables": [
        {"name": "clusters.txt", "print_every": 1e4,
         "cols": [{"type": "PLClusterTopology", "show_types": 1}]}
    ]
}
```

**3. Run it:**

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble

with find_ensemble(cfg="cross_quickstart") as e:
    e.do_setup(clean_build="force")
    e.start_simulations()        # runs all six simulations here, one after another
```

**4. Analyze it**, later or in the same session:

```python
from pypatchy.patchy.simulation_ensemble import find_ensemble
from pypatchy.patchy.analysis_lib import GraphsFromClusterTxt, ClassifyClusters, ComputeClusterYield
from pypatchy.polycubeutil.polycubesRule import rdir
from pypatchy.structure import Structure

with find_ensemble(name="cross_quickstart", date="2026-11-18") as e:   # the date you ran it
    obs = e.observables["clusters.txt"]
    cross = Structure(bindings=[(0, i, i + 1, rdir(i)) for i in range(6)])
    e.add_analysis_steps(
        GraphsFromClusterTxt("graphs", source=obs),
        ClassifyClusters("classify", target=cross),
        ComputeClusterYield("yield", cutoff=1.0),
        ("graphs", "classify"), ("classify", "yield"),
    )
    for sim in e.ensemble():
        y = e.get_data("yield", sim).get()
        print(sim, y["yield"].iloc[-1] / 8)       # fraction of complete crosses at the end
```

[`examples/setup_sims_programmatic_minimal.py`](../../examples/setup_sims_programmatic_minimal.py)
does the same thing entirely in Python, with plots.

## Vocabulary

| Term | Meaning |
|---|---|
| **ensemble** | A set of simulations sharing a spec, varying over the ensemble parameters. Identified by export name + setup date. |
| **simulation** | One point in the ensemble's parameter space (`PatchySimulation`). |
| **stage** | One segment of a simulation's protocol. Every simulation has at least one. |
| **rule** | A polycubes rule string: a set of cube types and their patch colors. |
| **particle set** | pypatchy's representation of particle types (`PLParticleSet`). |
| **scene** | One configuration, as particles in a box (`PLPSimulation`). |
| **server config** | Which oxDNA build/plugin/format to use and how to launch jobs on a machine. |
| **format / writer** | A patchy file format (`flavio`, `lorenzo`, `raspberry`, …) and the class that reads/writes it. |
| **multidentate** | Patches split into rings of smaller "teeth". |
| **observable** | Extra output written by oxDNA during a run (e.g. cluster/bond lists). |
| **pipeline step** | One node of the analysis graph. |
| **"PL"** | Legacy prefix for pypatchy's particle classes; no longer stands for anything. |

## Related tools

- **Polycubes** ([web app](https://akodiat.github.io/polycubes/),
  [code](https://github.com/Akodiat/polycubes)) — lattice-level assembly of polycube rules;
  quick checks of what a rule builds.
- **TLM** (`sulcgroup/polycubes-clone`) — fast lattice Monte Carlo for polycubes. Separate from
  the oxDNA simulations described here.
- **oxpy-utils** ([sulcgroup/oxpy-utils](https://github.com/sulcgroup/oxpy-utils)) —
  simulation objects, observables, DNA structure editing.
- **oxDNA** ([docs](https://lorenzo-rovigatti.github.io/oxDNA/)) and
  **oxDNA_analysis_tools**.
