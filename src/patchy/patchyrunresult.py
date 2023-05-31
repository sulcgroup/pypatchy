import os
import pickle
from multiprocessing import Pool
import oxpy
import numpy as np
import pandas as pd
import time
from analyseClusters import *
# definitely one of the weirder import commands I've used
from glob import glob

from util import *

# hardcoding this because reading it from the input file would be difficult I've literally never changed it
OXDNA_PRINT_CLUSTERS_EVERY = 1e7

# analysis status codes
ASC_NO_DATA = 0
ACS_NO_PICKLES = 1
ASC_PICKLE_CLUSTERS = 2

CLUSTERS_PICKLE = "clusters.pickle"

class PatchyRunResult:
    def __init__(self, pathroot, duplicate, nt, temp, density, num_assemblies):
        self.path_root = pathroot
        self.name = "_".join(pathroot.split(os.sep)[-1].split("_")[:-2])
        self.duplicate_number = duplicate
        self.narrow_type_number = nt
        self.temperature = temp
        self.particle_density = density
        self.num_assemblies = num_assemblies
        self.clusterfile = pathroot + get_cluster_file_name()

        self.cluster_graphs = None
        
        
        with open(self.get_path() + os.sep + get_init_top_file_name(), 'rt') as topfile:
            self.num_particles, self.num_particle_types = topfile.readline().split(" ")
            self.num_particle_types = int(self.num_particle_types)
            self.num_particles = int(self.num_particles)
            particles = topfile.readline().split(" ")
            self.particle_identities = [int(particle) for particle in particles]
    
    def status(self):
        if not os.path.isfile(self.get_cluster_txt_file()):
            return ASC_NO_DATA
        else:
            if not os.path.isfile(self.get_cluster_pickle()):
                return ACS_NO_PICKLES
            else:
                return ASC_PICKLE_CLUSTERS
        
    def get_cluster_graphs(self):
        if (self.cluster_graphs is None):
            if os.path.isfile(self.get_cluster_pickle()):
                with open(self.get_cluster_pickle(), 'wb') as f:
                    self.cluster_graphs = pickle.load(f)
            else:
                with open(self.get_cluster_txt_file()) as f:
                    self.cluster_graphs = [graphsFromClusters(line) for line in f]
                with open(self.path_root + CLUSTERS_PICKLE, 'wb') as f:
                    pickle.dump(self.cluster_graphs, f)
        return self.cluster_graphs
                
    def get_path(self):
        return os.sep.join([self.path_root,
                            "nt{}".format(self.narrow_type_number),
                            "T_{}".format(self.temperature)]) + os.sep
    
    def print_descriptors(self):
            return f"{self.name} duplicate {self.duplicate_number} nt={self.narrow_type_number} T={self.temperature}"

    def get_cluster_txt_file(self):
        return self.get_path() + get_cluster_file_name()

    def get_cluster_pickle(self):
        return self.get_path() + os.sep + CLUSTERS_PICKLE
    
    def get_cluster_cats_file(self, target_name):
        return self.get_path() + f"clustercats_{target_name}.pickle"
    
    def get_cluster_cats_metadata_file(self):
        return self.get_path() + "cluster_cats_metadata.pickle"
    
    def has_cluster_cats_file(self, target_name):
        return path.isfile(self.get_cluster_cats_file(target_name))
    
    def get_cluster_cats(self, target_name):
        with open(self.get_cluster_cats_file(target_name), 'rb') as f:
            return pickle.load(f)
        
    def get_num_cluster_cat_timepoints(self, target_name):
        if not path.isfile(self.get_cluster_cats_metadata_file()):
            return 0
        with open(self.get_cluster_cats_metadata_file(), 'rb') as f:
            metadata = pickle.load(f)
            return len(metadata["categorized_timepoints"][target_name]["timepoints"])
    
    def num_timepoints(self):
        return len(self.get_cluster_graphs())
    
    def max_time(self):
        return getVal(self.get_path() + "last_conf.dat", 't = ')
    
    def get_shape_name(self):
        return self.name[:self.name.find("_duplicate")] 
    
    def get_cluster_yields_input(self):
        with oxpy.Context():
            inp = oxpy.InputFile()
            inp.init_from_filename(self.get_path() + os.sep + "input")
            return inp
    
    ## NOTE: DO NOT USE THIS FUNCTION IT DOES NOT WORK
    def timepoint_interval(self):
        with oxpy.Context():
            input_file = oxpy.InputFile()
            input_file.init_from_filename(self.get_path() + "input")
            return int(input_file['data_output_1']['print_every'])
    
    def cluster_sizes_at(self, pt):
        return [g.order() for g in self.get_cluster_graphs()[pt]]

    def getTotalSizes(self):
        df = pd.DataFrame({
            "timepoint": np.array(range(self.num_timepoints())) * get_sample_every(),
            "total_size": [sum(self.cluster_sizes_at(pt)) for pt in range(self.num_timepoints())]
        })
        df["group"] = self.get_shape_name()
        df["nt"] = self.narrow_type_number
        df["temp"] = self.temperature
        df["duplicate"] = self.duplicate_number
        return df
    
    def clear_cluster_cats_files(self, target_name):
        '''
        Clears cluster category calculations files!!!
        WARNING: that's a lot of lost calculations!
        '''
        os.remove(self.get_cluster_cats_file(target_name))
        
        with open(self.get_cluster_cats_metadata_file(), 'rb+') as f:
            metadata = pickle.load(f)
            del metadata["categorized_timepoints"][target_name]
            f.seek(0)
            pickle.dump(metadata, f)
            
    def clear_yield_files(self, target_name):
        '''
        Removes yield cache files. Not as dangerous as the cluster categorization purge
        but still the potential to lose some calculations.
        '''
        for f in glob(self.get_path() + f"cluster_yields_{target_name}_C*_O*.pickle"):
            os.remove(f)
        
    def getClusterCategories(self, target_name, target_graph, verbose=False, parallel=False, sample_every=-1):
        """
        If clusters are already categorized, loads cluster categorization data from pickle file
        If not, categorizes clusters and pickles data
        Time indexes in the returened dataframe are in units of OXDNA_PRINT_CLUSTERS_EVERY * steps
        """
        if sample_every == -1:
            sample_every = get_sample_every()
        metadata_file = self.get_cluster_cats_metadata_file()
        if os.path.isfile(metadata_file):
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                if not target_name in metadata["categorized_timepoints"]:
                    metadata["categorized_timepoints"][target_name] = {
                        "timepoints": set()
                    }
        else:
            metadata = {
                "categorized_timepoints": {
                    target_name: {
                        "timepoints": set()
                    }
                }
            }
                
        filename = self.get_cluster_cats_file(target_name)
        # if has existing cluster cats, load
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                cluster_categories = pickle.load(f)
        # otherwise construct empty dataframe, settle in with beverage
        else:
            cluster_categories = pd.DataFrame(columns=["nt", "temp", "duplicate", "tidx", "clustercategory", "sizeratio"])
        # log tidxs processed
        tidxs_processed = metadata["categorized_timepoints"][target_name]["timepoints"]
        # check for missing timepoints
        if not tidxs_processed >= set(range(0, self.num_timepoints(), sample_every)):
            if verbose:
                print(f"Missing {len(set(range(0, self.num_timepoints(), sample_every)) - tidxs_processed)} timepoints... Categorizing...")
                tstart = int(time.time())
                print(f"Starting categorizing clusters {self.print_descriptors()}...")
            new_cluster_cats = []
            # loop timepoints
            for i, graphs in enumerate(self.get_cluster_graphs()):
                # if this timepoint need processing
                if i not in tidxs_processed and i % sample_every == 0:
                    if verbose:
                        print(f"Missing timepoint at timepoint {i}. Calculating...")
                    # loop graphs at timepoint
                    for graph in graphs:
                        new_cluster_cats.append(categorizeCluster(i, self, graph, target_graph))
                    # register processing at timepoint
                    tidxs_processed.add(i)
            cluster_categories = pd.concat([cluster_categories, pd.DataFrame(new_cluster_cats)])
            if verbose:
                print(f"Finished categorizing clusters in {int(time.time()) - tstart} seconds!")
                
            # save analysis results
            with open(filename, 'wb+') as f:
                pickle.dump(cluster_categories, f)
            metadata["categorized_timepoints"][target_name]["timepoints"] = tidxs_processed
            with open(metadata_file, 'wb+') as f:
                f.seek(0)
                pickle.dump(metadata, f)
        
        # return clusters at requested interval
        return cluster_categories.loc[cluster_categories['tidx'] % sample_every == 0]
    
    def analyseClusterYield(self, target_name, target_graph, cutoff=1, overreach=False, verbose=False, parallel=False, sample_every=-1):
        '''
        Computes yields of clusters in graph
        '''
        if sample_every == -1:
            sample_every = get_sample_every()
        cluster_yields_file = self.get_path() + f"cluster_yields_{target_name}_C{cutoff}_O{overreach}.pickle"
        # if os.path.isfile(cluster_yields_file):
        #     if verbose:
        #         print(f"Fetching yield data from file {cluster_yields_file}...")
        #     with open(cluster_yields_file, 'rb') as f:
        #         data = pickle.load(f)
        # else:
        if True:
            tstart = time.perf_counter()
            if verbose:
                print(f"Computing yields of {self.print_descriptors()} with target={target_name}, overreach={overreach}, cutoff={cutoff}...")
            # get parameters
            inputfile = self.get_cluster_yields_input()
            
            # dt = getVal(run_result.get_path() + "input", 'dt = ') # todo: switch to oxpy

            # figure out how long simulation ran
            # maxTimeStep = self.max_time()
            
            # 
            tlength = self.num_timepoints()
            cluster_categories = self.getClusterCategories(target_name, target_graph, verbose, parallel, sample_every)
            
            # create a 'default' version of the yield dataframe to serve for timepoints with no clusters
            data = pd.DataFrame(index=pd.MultiIndex.from_product([
                    [self.narrow_type_number], [self.temperature], [self.duplicate_number], np.arange(0, tlength, sample_every)
                    ], names=['nt', 'temp', 'duplicate', 'tidx']),
                                columns=['yield'])
            data['yield'] = 0
            data['shape'] = self.get_shape_name()
            data['num_assemblies'] = self.num_assemblies
            
            if len(cluster_categories.index) > 0:
                # filter out oversize clusters
                if not overreach:
                    cluster_categories = cluster_categories.loc[cluster_categories['clustercategory'].isin([ClusterCategory.SUBSET, 
                                                                                                            ClusterCategory.MATCH])]
                # filter clusters under cutoff
                if cutoff > 0:
                    cluster_categories = cluster_categories.loc[cluster_categories['sizeratio'] >= cutoff]

                cluster_categories.drop('clustercategory', axis=1, inplace=True)
                yields = cluster_categories.groupby(['nt', 'temp', 'duplicate', 'tidx']).sum()
                yields['yield'] = yields['sizeratio']
                data.update(yields)

                # get time in steps by multiplying the time index by the cluster print interval
                data['time'] = 0
                data['time'] = data['time'].astype('ulonglong')
                data['time'] = data.index.get_level_values("tidx") * OXDNA_PRINT_CLUSTERS_EVERY

            if verbose:
                print(f"{self.print_descriptors()} - Max yield: {data['yield'].max()}")
        
            with open(cluster_yields_file, 'wb+') as f:
                pickle.dump(data, f)
            if verbose:
                print(f"Computed yields in {time.perf_counter() - tstart}!");
        return data