This repository contains a collection of scripts written by Josh Evans and Joakim Bohlin for Patchy Particles data.
At some point it will probably be merged into [polycubes-clone](https://github.com/sulcgroup/polycubes-clone) and/or [patchy-particle-utils](https://github.com/sulcgroup/patchy-particle-utils)
Requirements:
- oxpy
- altair
- numpy
- networkx
- pandas

TODO: seperate data processing scripts from figure-forming jupyter notebooks

# Analysis Notebooks
## Chart Cluster Sizes
## Compare Cutoffs
## compare_datasets.ipynb
Compaers a number of datasets which have the same analysis target (or at least analysis targets with the same name) and overlapping narrow types and temperatures.
## Compare Temperature
## generate_graphs.ipynb
Generates a set of graphs for a specific narrow type and a single dataset at the full range of temperatures, along with a chart showing the number of each cube type which
were present.
## Graph Duplicates
## Show Clusters

# The Analysis Pipeline
# compute_yields.py
`compute_yields.py` is a python script that can be run from shell that will run yield calculations for the patchy run result in a provided directory (directory should be the one with the oxdna `input` file). It will automatically load the result group from the `patchy_export_settings.json` and `analysis_params.json` files in the datset directory.
# compute_yields.sh
This shell script will start slurm jobs to compute the yields of a dataset. It takes the following parameters
- `d` the folder name of a dataset to analyze. 
- `t` the name of a target to use for the analysis
- `i` (optional) the frequency with which to run the analysis. If set to 1, all the cluster timepoints will be analyzed, however there are many cluster timepoints so this would be unwise. the default value is the value `sample_every` in `settings.cfg`
Run `print_all_results_status()` in a jupyter notebook to get options for dataset names and timepoints
Example: `./compute_yields.sh -d WTSolidCube_7Nov22 -t solidcube -i 10`

# Libraries
## util.py
## patchyresults.py
## patchyrunresult.py
## input_output.py
function `print_all_results_status`: Lists all the datasets in the analysis directory and prints info
function `choose_results`: Constructs a PatchyRunSet object. Can be given a string name. If no name is given, the function will list options and prompt the user to choose one.