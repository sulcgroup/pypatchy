#!/bin/bash

# example ./compute_yields.sh -d WTSolidCube_7Nov22 -t solidcube -r 1000 -c 0 -o False

# hardcoding directory here because I'm too lazy not to
srcdir="/scratch/jrevan21/analysis_space";

while getopts d:t:i: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        t) target=${OPTARG};;
        i) interval=${OPTARG};;
    esac
done


printf "Analyzing data for dataset %s with target %s at a timepoint interval of %d" $dataset $target $interval;

for directory in $srcdir/$dataset/*_duplicate_*/nt?/T_*;
do
    printf "Processing data at %s.\n" $directory >> slurm_out/logs/$dataset.txt; 
    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name="compute_yields" # Name of the job in the queue
#SBATCH --error="./slurm_out/comp_yield_job_%j.err"     # Name of stderr file
#SBATCH --out="./slurm_out/comp_yield_job_%j.out"        # Name of the stdout file
#SBATCH -p sulccpu1
#SBATCH -q sulcgpu1
#SBATCH -n 1
#SBATCH -t 6-00:00
module load anaconda/py3
source activate polycubes2
     
python3 compute_yields.py -p $directory -t $target -i $interval

exit 0
EOT

done