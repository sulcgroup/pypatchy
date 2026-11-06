"""
Minimal example of constructing a patchy particle ensemble programmatically, without using
find_ensemble or any external JSON config, then actually running it.

Uses the "3D cross" rule: one cube type with all six patches colored to bind to each other
(color 1), and a second cube type with a single patch (color -1) that caps it. This is the
same rule used by tests/test_files/ensemble_specs/example_basic.json.

Requires the OXDNA_PATH environment variable to point at a checkout of
https://github.com/lorenzo-rovigatti/oxDNA with the "romano" (PatchyShapeInteraction) plugin
compiled under contrib/romano. There's no way to derive this from the installed oxpy package -
oxpy is just a compiled core.so sitting in site-packages, with no metadata linking it back to
whatever source tree it was built from.
"""

import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
from oxpy_utils.utils.observable import Observable, ObservableColumn

from pypatchy.patchy.simulation_ensemble import build_ensemble, EXPORT_NAME_KEY, \
    ENSEMBLE_SETUP_DATE, CONST_PARAMS_KEY, ENSEMBLE_PARAMS_KEY, SERVER_SETTINGS_KEY, PARTICLE_TYPE_LVLS_KEY, \
    OBSERABLES_KEY
from pypatchy.patchy.analysis_lib import LoadEnergies, GraphsFromClusterTxt, ComputeClusterSizeData, \
    ClassifyClusters, ComputeClusterYield, YIELD_KEY
from pypatchy.analysis.analysis_data import TIMEPOINT_KEY
from pypatchy.polycubeutil.polycubesRule import rdir
from pypatchy.server_config import PatchyServerConfig
from pypatchy.structure import Structure
from pypatchy.util import get_output_dir

rule = "1|1|1|1|1|1_-1|||||"

# PLClusterTopology (contrib/romano) writes cluster membership + intra-cluster adjacency every
# print_every steps. GraphsFromClusterTxt below turns this into networkx graphs.
cluster_observable = Observable("clusters.txt", 1e4, ObservableColumn(name="PLClusterTopology", show_types=1))

oxdna_path = os.environ.get("OXDNA_PATH")
assert oxdna_path, "Set the OXDNA_PATH environment variable to a checkout of oxDNA with contrib/romano built."
plugin_search_path = str(Path(oxdna_path) / "contrib" / "romano")

e = build_ensemble(
    cfg={
        EXPORT_NAME_KEY: "example_3dX_minimal",
        "rule": rule,
        CONST_PARAMS_KEY: {
            "num_assemblies": 8,
            # "dt": 0.01, # this is unphysically high but good for a quick running example
            "density": 0.3,
            "steps": 5e6,
            "print_conf_interval": 1e4,
            "print_energy_every": 1e4,
            "narrow_type": 1,
            "patchy_file": "patches.txt",
            "particle_file": "particles.txt",
            PARTICLE_TYPE_LVLS_KEY: [1, 6],
            # these would normally live in a server config's input_file_params, but since
            # we're building PatchyServerConfig on the fly here (no input_file_params set),
            # they need to go somewhere - const_params works just as well.
            "backend": "CPU",
            "backend_precision": "double",
            "interaction_type": "PatchyShapeInteraction",
            "plugin_search_path": plugin_search_path,
        },
        ENSEMBLE_PARAMS_KEY: [
            ("T", [0.01, 0.025]),
            ("duplicate", list(range(3)))
        ],
        OBSERABLES_KEY: [cluster_observable],
        SERVER_SETTINGS_KEY: PatchyServerConfig(
            name="example_3dX_minimal_local",
            oxdna_path=oxdna_path,
            patchy_format="flavio",
            is_slurm=False
        ),
    },
    mdt={
        ENSEMBLE_SETUP_DATE: datetime.datetime.now()
    }
)

# if we use ensemble as a context manager it will automatically save the changes when we exit context
with e:
    # start_simulations() (via start_simulation()) now runs each simulation in-process via oxpy
    # and blocks until it finishes, raising RuntimeError itself if a run doesn't complete
    # successfully - no need to separately join() or check completion here.
    e.do_setup(clean_build="force")
    e.start_simulations()

# ---------------------------- ANALYSIS ---------------------------- #

# the target for yield analysis: a single "3D cross" assembly, hardcoded as a star graph -
# one CT0 center (node 0) bonded to all 6 CT1 caps (nodes 1-6). ClassifyClusters.compare() only
# checks undirected subgraph isomorphism, so the direction indices below aren't actually load-
# bearing, but we include the real ones anyway since this is meant to be a complete example.
# CT0 has a patch on every face of RULE_ORDER = (left, right, bottom, top, back, front), i.e.
# indices 0-5; CT1 (the caps) has a single patch, at its own index 0, that binds to each of them.
cross_target = Structure(bindings=[(0, i, i+1, rdir(i)) for i in range(6)])

energies = LoadEnergies(step_name="energies", output_tstep=int(e.get_param("print_energy_every")))
graphs = GraphsFromClusterTxt(name="graphs", source=cluster_observable)
graphs.force_recompute=True
cluster_sizes = ComputeClusterSizeData(name="cluster_sizes", output_tstep=int(cluster_observable.print_every))
classify = ClassifyClusters(name="classify", target=cross_target, output_tstep=int(cluster_observable.print_every))
yields = ComputeClusterYield(name="yield", cutoff=0.15, overreach=False,
                             output_tstep=int(cluster_observable.print_every))

e.add_analysis_steps(
    energies, graphs, cluster_sizes, classify, yields,
    ("graphs", "cluster_sizes"),
    ("graphs", "classify"),
    ("classify", "yield"),
)

plots_dir = get_output_dir() / "example_3dX_minimal_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# yield (as computed by ComputeClusterYield) is a sum of size ratios of matching clusters, so its
# max possible value is num_assemblies (i.e. every particle assembled into a complete target). divide
# by num_assemblies to get a 0-1 fraction for plotting.
num_assemblies = e.get_param("num_assemblies")

# gather per-sim data first, grouped by T so each T's duplicates can be plotted together
size_data_by_sim = {}
yield_data_by_sim = {}
sims_by_T = {}
for sim in e.ensemble():
    T = e.sim_get_param(sim, "T")
    sims_by_T.setdefault(T, []).append(sim)
    size_data_by_sim[sim] = e.get_data(cluster_sizes, sim).get()
    yield_data_by_sim[sim] = e.get_data(yields, sim).get()

# one figure per T, with all of that T's duplicates overlaid on each panel
for T, sims in sorted(sims_by_T.items()):
    fig, ((ax_energy, ax_size), (ax_dist, ax_yield)) = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"T={T} ({len(sims)} duplicates)")

    all_last_conf_sizes = []
    last_conf_t = None
    for sim in sims:
        duplicate = e.sim_get_param(sim, "duplicate")
        energy_data = e.get_data(energies, sim).get()
        size_data = size_data_by_sim[sim]
        yield_data = yield_data_by_sim[sim]
        graphs_by_time = e.get_data(graphs, sim).get()
        last_conf_t = max(graphs_by_time)
        all_last_conf_sizes.extend(len(g) for g in graphs_by_time[last_conf_t])

        color = f"C{duplicate}"
        ax_energy.plot(energy_data[TIMEPOINT_KEY], energy_data[LoadEnergies.POTENTIAL_ENERGY_KEY],
                       color=color, linestyle="-", label=f"dup{duplicate} potential")
        ax_energy.plot(energy_data[TIMEPOINT_KEY], energy_data[LoadEnergies.KINETIC_ENERGY_KEY],
                       color=color, linestyle="--", label=f"dup{duplicate} kinetic")

        ax_size.plot(size_data[TIMEPOINT_KEY], size_data[ComputeClusterSizeData.MEAN_KEY],
                     color=color, label=f"dup{duplicate}")

        ax_yield.plot(yield_data[TIMEPOINT_KEY], yield_data[YIELD_KEY] / num_assemblies,
                      color=color, label=f"dup{duplicate}")

    ax_energy.set_xlabel("timestep")
    ax_energy.set_ylabel("energy")
    ax_energy.legend(fontsize="small")

    ax_size.set_xlabel("timestep")
    ax_size.set_ylabel("mean cluster size")
    ax_size.legend(fontsize="small")

    # last-conf cluster size distribution, pooled across all of this T's duplicates
    ax_dist.hist(all_last_conf_sizes, bins=range(1, max(all_last_conf_sizes, default=1) + 2))
    ax_dist.set_xlabel("cluster size")
    ax_dist.set_ylabel(f"count (t={last_conf_t}, pooled over duplicates)")

    ax_yield.set_xlabel("timestep")
    ax_yield.set_ylabel("yield (fraction of num_assemblies)")
    ax_yield.set_ylim(0, 1.)
    ax_yield.legend(fontsize="small")

    fig.tight_layout()
    plot_path = plots_dir / f"T_{T}.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Wrote plot for T={T} ({len(sims)} duplicates) to {plot_path}")

# comparison plot: cluster size and yield vs time, one line per T (averaged over duplicates)
cmp_fig, (ax_size_cmp, ax_yield_cmp) = plt.subplots(1, 2, figsize=(12, 5))
cmp_fig.suptitle("Comparison across temperatures (mean over duplicates)")

for T, sims in sorted(sims_by_T.items()):
    timepoints = size_data_by_sim[sims[0]][TIMEPOINT_KEY]
    mean_size = sum(size_data_by_sim[sim][ComputeClusterSizeData.MEAN_KEY].values for sim in sims) / len(sims)
    ax_size_cmp.plot(timepoints, mean_size, label=f"T={T}")

    mean_yield = sum(yield_data_by_sim[sim][YIELD_KEY].values for sim in sims) / (len(sims) * num_assemblies)
    ax_yield_cmp.plot(timepoints, mean_yield, label=f"T={T}")

ax_size_cmp.set_xlabel("timestep")
ax_size_cmp.set_ylabel("mean cluster size")
ax_size_cmp.legend()

ax_yield_cmp.set_xlabel("timestep")
ax_yield_cmp.set_ylabel("yield (fraction of num_assemblies)")
ax_yield_cmp.set_ylim(0, 1.)
ax_yield_cmp.legend()

cmp_fig.tight_layout()
cmp_plot_path = plots_dir / "comparison_across_T.png"
cmp_fig.savefig(cmp_plot_path)
plt.close(cmp_fig)
print(f"Wrote temperature comparison plot to {cmp_plot_path}")