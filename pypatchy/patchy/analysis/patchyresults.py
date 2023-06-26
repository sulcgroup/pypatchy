import itertools
from os import path

from IPython.core.display_functions import display

import matplotlib.pyplot as plt
from matplotlib import cm

from pypatchy.polycubeutil.polycubesRule import *

from pypatchy.patchy.analysis.patchyrunresult import *
from pypatchy.util import get_export_setting_file_name


class PatchyRunSet:
    def __init__(self, froot, analysisparams):
        fpath = path.join(froot, get_export_setting_file_name())
        self.root_dir = froot
        with open(fpath, 'r') as f:
            run_setup = json.load(f)
            self.export_name = run_setup['export_name']
            self.narrow_types = run_setup['narrow_types']
            self.temperatures = run_setup['temperatures']
            if 'rules' in run_setup:
                self.rule = [PolycubeRuleCubeType(ct) for ct in run_setup['rules']]
            else:
                self.rule = [PolycubeRuleCubeType(ct) for ct in run_setup['cube_types']]
            self.export_group_names = [r['name'] if 'name' in r else r['exportGroupName'] for r in
                                       run_setup['export_groups']]
            self.runs = [PatchyRunType(r, self.root_dir) for r in run_setup['export_groups']]
        self.targets = {}
        for target_name in analysisparams['targets']:
            self.targets[target_name] = {
                'name': target_name,
                'graph': graphShape(self.root_dir + os.sep + analysisparams['targets'][target_name]['file']),
            }
            # number of replications of the analysis target per "assembly"
            # for an analysis target that's just the entire asesmbly, this will equal 1. for
            # an analysis target that is a repeated subgraph of the entire assembly, the
            # target will be some integer greater than 1
            if 'rel_count' in analysisparams['targets'][target_name]:
                self.targets[target_name]['rel_count'] = analysisparams['targets'][target_name]['rel_count']
            else:
                self.targets[target_name]['rel_count'] = 1

                # --------- BASIC ACCESSORS ------------

    def name(self):
        return self.export_name

    def num_narrow_types(self):
        return len(self.narrow_types)

    def num_temperatures(self):
        return len(self.temperatures)

    def get_narrow_types(self):
        return self.narrow_types

    def get_temperatures(self):
        return self.temperatures

    def get_rule(self):
        return self.rule

    # -----------DISPLAY FUNCTIONS ---------
    def print_short(self, textwidth=64):
        print("Name:", self.export_name.rjust(textwidth - len("Name:")))
        print("Rule:")
        display(self.rule.to_dataframe())
        print("Num Export Groups:" + f"{len(self.export_group_names)}".rjust(textwidth - len("Num Export Groups:")))

    def print_status(self):
        """
        Helper method which prints the overall status of this result set and the state
        of analysis
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        # print basic data
        self.print_short()
        # TODO: export groups

        print("Analysis Status Codes: 0=No Data, 1=No Graph Pickles, 2=Graph Pickles")
        display(self.status_dataframe())
        pd.reset_option('all')

    def status_dataframe(self):
        df = pd.DataFrame([
            {
                "type": sim.name,
                "dupl.": sim.duplicate_number,
                "nt": sim.narrow_type_number,
                "T": sim.temperature,
                "density": sim.particle_density,
                "# assemblies": sim.num_assemblies,
                "nP": sim.num_particles,
                "nS": sim.num_particle_types,
                "status": sim.status(),
                "# timepts": sim.num_timepoints(),
                "max time(Mstps)": sim.max_time() / 1e6 if sim.max_time() is not None else np.NaN,
                **{
                    f"{targname} cat.d. #": sim.get_num_cluster_cat_timepoints(targname)
                    for targname in self.targets.keys()}

            }
            for sim in self.flat_runs()
        ])
        df = df.set_index(["type", "nt", "T", "dupl."])
        return df

    # ----------- ACCESSORS ----------- #
    #### --------- Simulation Groups ----------- #
    def get_run(self, type_idx, duplicate, nt, temperature):
        return self.get_run_type(type_idx).get_run(duplicate, nt, temperature)

    def get_run_type(self, id):
        if isinstance(id, int):
            return self.runs[id]
        elif isinstance(id, str):
            return [r for r in self.runs if r.name == id][0]

    def flat_runs(self):
        return [*itertools.chain.from_iterable([r.run_results for r in self.runs])]

    def run_count(self):
        return len(self.flat_runs())

    def temperatures(self):
        return np.unique([r.temperature for r in self.flat_runs()])

    def narrow_types(self):
        return np.unique([r.narrow_type_number for r in self.flat_runs()])

    def export_groups(self):
        return list(set([r.name for r in self.flat_runs()]))

    #### ------------ Yield Computation Targets ------------- #
    def num_targets(self):
        return len(self.targets)

    def target_names(self):
        return [*self.targets.keys()]

    def has_target(self, targetname):
        return targetname in self.target_names()

    def get_target_ref_graph(self, target):
        return self.targets[target]['graph']

    # ------------ USER INPUT FUNCTIONS ----------- #
    def choose_narrow_type(self):
        return int(
            input(f"Input the narrow type of the data ({', '.join([f'{nt}' for nt in self.get_narrow_types()])}): "))

    def choose_export_group(self):
        return input(f"Enter the export group ({','.join(self.export_group_names)}):")

    def choose_temperature(self):
        return float(input(f"Enter the temperature ({','.join([f'{temp}' for temp in self.get_temperatures()])}) :"))

    def choose_duplicate(self, export_group):
        return int(input(
            f"Enter a duplicate ({','.join([f'{d}sample_every' for d in self.get_run_type(export_group).duplicates])}): "))

    def get_yields(self, target_name, cutoff=1, overreach=False, verbose=False, sample_every=-1):
        target_graph = self.targets[target_name]['graph']
        all_data = [
            r.analyseClusterYield(target_name, target_graph, cutoff=cutoff, verbose=verbose, overreach=overreach,
                                  sample_every=sample_every) for r in self.flat_runs()]
        return pd.concat(all_data)

    def get_flat_yields(self, target_name, cutoff=1, overreach=False, verbose=False, pop_cols=[], filters={},
                        sample_every=-1):
        data = self.get_yields(target_name, cutoff=cutoff, verbose=verbose, overreach=overreach,
                               sample_every=sample_every)
        data = data.reset_index()

        cols = [c for c in list(data.columns) if c not in pop_cols]
        data = data[cols]
        for filterkey in filters:
            data = data[data[filterkey] == filters[filterkey]]
        return data

    def clear_cluster_cats_files(self, target_name):
        '''
        Clears cluster category calculations files!!!
        WARNING: that's a lot of lost calculations!!!!
        '''

        for rs in self.flat_runs():
            rs.clear_cluster_cats_files(target_name)

    def clear_yield_files(self, target_name):
        for rs in self.flat_runs():
            rs.clear_yield_files(target_name)

    def get_stats(self, target_name, cutoff=1, overreach=False, verbose=False, grouping_cols=['nt', 'temp', 'time'],
                  sample_every=-1):
        data = self.get_flat_yields(target_name, cutoff=cutoff, overreach=overreach, verbose=verbose, pop_cols=['tidx'],
                                    sample_every=sample_every)
        # data = data.dropna()
        gb = data.groupby(grouping_cols)
        avgs = gb.mean()
        avgs["yield_min"] = gb.min()['yield']
        avgs["yield_max"] = gb.max()['yield']
        avgs["yield_stdev"] = gb.std()['yield'].fillna(0)
        avgs = avgs.drop(["duplicate"], axis=1)
        return avgs

    def get_all_sizes(self):
        return pd.concat([r.getTotalSizes() for r in self.flat_runs()])


class PatchyRunType:
    def __init__(self, r, root_dir):
        self.root_dir = root_dir
        self.name = r['name'] if 'name' in r else r['exportGroupName']
        self.num_assemblies = int(r['num_assemblies'])
        self.cube_type_levels = [int(lvl) for lvl in r['rule_levels']] if 'particle_type_levels' not in r else [int(lvl)
                                                                                                                for lvl
                                                                                                                in r[
                                                                                                                    'particle_type_levels']]
        self.temperatures = [float(k) for k in r['temperatures'] if r['temperatures'][k]]
        self.narrow_types = r['narrow_types']
        self.duplicates = range(r['num_duplicates'])
        self.particle_density = r['particle_density']
        self.run_results = []
        self.run_results_map = {}
        for (d_idx, d) in enumerate(self.duplicates):
            for (nt_idx, nt) in enumerate(self.narrow_types):
                for t in self.temperatures:
                    self.run_results.append(
                        PatchyRunResult(self.root_dir + os.sep + "{}_duplicate_{}".format(self.name, d), d, nt, t,
                                        self.particle_density, self.num_assemblies))
                    self.run_results_map[(d, nt, t)] = self.run_results[len(self.run_results) - 1]

    def get_run(self, duplicate, nt, temperature):
        return self.run_results_map[(duplicate, nt, temperature)] if (duplicate, nt,
                                                                      temperature) in self.run_results_map else None

    def graph_cluster_sizes_temperature(self, narrow_type, duplicates=None, ax=plt.axes(), step_interval=25):
        if duplicates is None:
            duplicates = self.duplicates

        colors = cm.get_cmap('Accent')
        all_xs, all_ys, all_cs, all_counts = [], [], [], []
        for (tidx, t) in enumerate(self.temperatures):
            for d in duplicates:
                r = self.get_run(d, narrow_type, t)
                xs, ys, counts = r.cluster_sizes(np.arange(stop=r.num_cluster_timepoints(), step=step_interval))
                all_xs.append(xs)
                all_ys.append(ys)
                all_counts.append(counts)
                all_cs.append([colors.colors[tidx] for _ in xs.shape[0]])
                # r.graphClusterSizes(ax, np.arange(stop=r.num_cluster_timepoints(), step=25), color=colors.colors[tidx])
        ax.scatter(all_xs, all_ys, s=all_counts, c=all_cs)

    def graph_cluster_yields_temperature(self, narrow_type, duplicates=None, ax=plt.axes(), step_interval=25):
        if duplicates is None:
            duplicates = self.duplicates
        data = self.get_yields(0.75, step_interval, narrow_types=[narrow_type])
        data = data.loc[[x in duplicates for x in data['duplicate']]]
        yields = [data.loc[data['temp'] == t]['yield'] for t in self.temperatures]
        times = [data.loc[data['temp'] == t]['time'] for t in self.temperatures]
        ax.plot(times, yields)


def is_results_directory(p):
    """
    Returns true if the provided directory is a valid Patchy Run Result directory
    TODO: more clauses
    """
    # if path doesn't exist or is not a directory, return false
    if not path.exists(p) or not path.isdir(p):
        return False
    else:
        if not path.exists(path.join(p, "patchy_export_setup.json")):
            return False
    return True


class BadTargetException(Exception):
    def __init__(self, target_name, result_set):
        self.target_name = target_name;
        self.result_set_name = result_set.export_name
        self.result_set_targets = result_set.target_names()

    def __str__(self):
        return f"Yield computation target '{self.target_name}' not in {self.result_set_name} targets {','.join(self.result_set_targets)}"
