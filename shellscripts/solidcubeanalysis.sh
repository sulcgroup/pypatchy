#!/bin/bash

#SBATCH --job-name="compute_cube3x3_yields"
#SBATCH --error="./slurm_out/compute_cube3x3_yields.err"
#SBATCH --out="./slurm_out/compute_cube3x3_yields.out"
#SBATCH -p run
#SBATH -c 1
#SBATCH --ntasks-per-node=6
#SBATCH -t 48-00:00:00

source /home/joshua/.bashrc
source activate polycubes

for accuracy in $(seq 0.5 0.1 0.9); do
	python3 ~/git/patchy-analysis/compute_yields.py -d AlloSolidCube_X1_7Nov22 -t solidcube -f 1000 -c $accuracy -o False
	python3 ~/git/patchy-analysis/compute_yields.py -d AlloSolidCube_X3_Singlet_7Nov22 -t solidcube -f 1000 -c $accuracy -o False
	python3 ~/git/patchy-analysis/compute_yields.py -d AlloSolidCube_X5_1_7Nov22 -t solidcube -f 1000 -c $accuracy -o False
	python3 ~/git/patchy-analysis/compute_yields.py -d AlloSolidCube_X5_4_7Nov22 -t solidcube -f 1000 -c $accuracy -o False
done
