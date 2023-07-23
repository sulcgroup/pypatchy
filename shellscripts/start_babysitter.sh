#!/bin/bash
# This script launches a PatchySimulationEnsemble in the background with a nice level of 19

# Load the necessary modules
module load anaconda/py3
source activate polycubes

# Check that at least one argument is provided
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 specfile [simdate]"
  exit 1
fi

specfile=$1;

if [[ $# == 2 ]]; then
  simdate=$2
  nice -n 19 python <<EOT &
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
try:
  PatchySimulationEnsemble("$specfile", sim_date="$simdate").babysit()
except Exception as e:
  print(f"Error occurred: {e}")
  exit(1)
EOT
else
  nice -n 19 python <<EOT &
from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble
try:
  PatchySimulationEnsemble("$specfile").babysit()
except Exception as e:
  print(f"Error occurred: {e}")
  exit(1)
EOT
fi
