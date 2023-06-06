import sys
sys.path.append('../src')
from patchy.util import *
from patchy.analysis.input_output import *
import argparse
import re

if __name__ == '__main__':
        # set up argument parser
    parser = argparse.ArgumentParser(prog="ComputeYields", description="Analyzes clusters.txt files produced by oxDNA's PLClusterTopology observable during execution of PatchShapeInteraction, producing pickled graph files. The graph files are in turn processed to produce yield data.")
    parser.add_argument("-p", "--path", action="store", required=True, help="The path at which to find the dataset to process")
    parser.add_argument("-t", "--target", action="store", required=False, default=None, help="The target topology (will automatically run calculations for all targets if none)")
    parser.add_argument("-i", "--interval", action="store", required=False, type=int, default=get_sample_every(), help="The interval at which to sample the timepoints (default: every 10 timepoints)")
    # parse args
    args = parser.parse_args()
    print(f"Computing yields of data in folder {args.path} against target {args.target} with interval {args.interval}")
    
    if not re.fullmatch(f"{re.escape(sims_root())}[^\/]*\/[\w_-]+_duplicate_\d+\/nt[0-4]\/T_[\.\d]*", args.path):
        raise BadSimulationDirException(args.path)
        
    dataset_name = args.path.split(os.sep)[-4]
    # load dataset metadata
    dataset = choose_results(dataset_name)
    
    if not dataset.has_target(args.target):
        raise BadTargetException(args.target, dataset)
    
    dirstructure = args.path.split(os.sep)
    temperature_dir = dirstructure[-1]
    temperature = float(temperature_dir[2:])
    nt_dir = dirstructure[-2]
    nt = int(nt_dir[2:])
    duplicate_dir = dirstructure[-3]
    duplicate_num = int(duplicate_dir[duplicate_dir.rindex("_")+1:])
    
    type_idx = dataset.export_group_names.index("_".join(duplicate_dir.split("_")[:-2]))
    
    runresult = dataset.get_run(type_idx, duplicate_num, nt, temperature)
    if args.interval > runresult.num_timepoints():
        print(f"Provided interval {args.interval} greater than number of timepoints {runresult.num_timepoints()}. Evaluating at first and last timepoint...")
        args.interval = runresult.num_timepoints() - 1
    
    if (args.target is not None):
        print(f"Categorizing clusters of {runresult.print_descriptors()} against target {args.target}...")
        runresult.getClusterCategories(args.target, dataset.targets[args.target]['graph'], verbose=True, parallel=False, sample_every=args.interval)
    else:
        for targetname in dataset.targets:
            print(f"Categorizing clusters of {runresult.print_descriptors()} against target {targetname}...")
            runresult.getClusterCategories(targetname, dataset.targets[targetname]['graph'], verbose=True, parallel=False, sample_every=args.interval)

    