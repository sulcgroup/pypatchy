todo: write nice
todo: more deets

# Test Scripts #

## the following are good, please do not touch them! ##

- `test_prerequisites.py` contains tests for `oxpy`, `oat`, `ipy_oxDNA`, and the required pypatchy directory structue
- `test_polycube_lib.py` test scripts for basic polycube libs


## the following are WIP ##
- `test_patchy_io.py` standalone tests for patchy file formats
- `test_analysis.py` contains test of analysis functionality 
- `test_solver_lib.py` contains test files for SAT solver library etc.
- `test_energy_conservation.py` runs a bonded two-particle NVE trajectory (DetailedPatchySwapInteraction)
  and checks total energy doesn't drift, on both the CPU and CUDA backends. The CUDA case auto-skips
  unless `oxpy` was built with `-DCUDA=ON` -- see `build_cuda_oxdna.sh`.


# Additional Files #
# Test Data #
tests require data which i haven't included here, it's on the server which we also use for nanobase

i think? this means you need to be on campus or on the ASU VPN for these to work?

anyway required data will be downloaded automatically

## List of Remote Datasets ##
- 4x4 tile_onto
- 