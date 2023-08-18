from .patchyresults import *
from ...util import simulation_run_dir, get_analysis_params_file_name, simulation_analysis_dir


def choose_results(sim_name=None) -> PatchyRunSet:
    if sim_name is None:
        print("Available datasets:\n\t" + ',\n\t'.join(os.listdir(simulation_analysis_dir())))
        sim_name = input("Enter simulation name: ")

    if not os.path.isdir(simulation_analysis_dir() / sim_name):
        print(f"No simulation folder at {simulation_analysis_dir() / sim_name}. Exiting.")
        exit(0)

    analysis_params = {}
    try:
        with open(simulation_analysis_dir() / sim_name / get_analysis_params_file_name(), 'r') as f:
            analysis_params = json.load(f)
    except FileNotFoundError:
        print("No analpipe params file found. Continuing with default analpipe params...")
        dir_files = os.listdir(simulation_analysis_dir() / sim_name)
        analysis_params['targets'] = []
        for target_file in dir_files:
            m = re.match('target_(.+)\.json', target_file)
            if m:
                analysis_params[m.group(1)] = {
                    "file": target_file
                }
    if len(analysis_params['targets']) == 0:
        print("No valid topology files to use to analyze yields. Exiting.")
        exit(0)

    results = PatchyRunSet(simulation_analysis_dir() / sim_name, analysis_params)

    return results


def choose_target(results: PatchyRunSet):
    target_name = ""
    while target_name == "":
        target_name = input(
            f"Select target topology for {results.export_name} (options are: {','.join(results.targets)}):")
        if target_name not in results.targets:
            target_name = ""  # reset target name to empty so loop will continue until user inputs a valid target name
    return target_name


def choose_results_and_target():
    # construct a PatchySimResultSet object from the specified directory
    results = choose_results()
    target_name = input(f"Input the name of an analpipe target ({','.join(results.targets.keys())}): ")

    target = results.targets[target_name]
    return results.export_name, results, target_name, target


def print_all_results_status():
    dirs = os.listdir(simulation_analysis_dir())
    datasets_dirs = list(filter(lambda r: is_results_directory(simulation_analysis_dir() / r + os.sep), dirs))
    datasets = [choose_results(ds) for ds in datasets_dirs]
    datasets_info = pd.DataFrame({
        "Directory Name": list(datasets_dirs),
        "Export Name": [ds.name() for ds in datasets],
        "Has Analysis Params": [os.path.isfile(ds.root_dir + os.sep + get_analysis_params_file_name()) for ds in
                                datasets],
        "Num. Targets": [ds.num_targets() for ds in datasets],
        "Target Names": [",".join([t["name"] for t in ds.targets.values()]) for ds in datasets],
        "Narrrow Types": [",".join([f"{nt}" for nt in ds.get_narrow_types()]) for ds in datasets],
        "Temperatures": [",".join([f"{T}" for T in ds.get_temperatures()]) for ds in datasets],
        "Num. Cube Types": [len(ds.get_rule()) for ds in datasets],
        "Num. Simulations": [ds.run_count() for ds in datasets],
    })
    display(datasets_info)
