import numpy as np

from pypatchy.patchy.param_val_types import StagedAssemblyParam, StageInfoParam
from pypatchy.patchy.particle_adders import RandParticleAdder
from pypatchy.patchy.simulation_ensemble import find_ensemble
from concurrent.futures import ThreadPoolExecutor, as_completed


e = find_ensemble(cfg="sbo_temp_ramped")

tmax = 1.6e10
# set of temperatures to iterate through
stages_temperatures = np.arange(0.0725, 0.0525, -0.0005)
stages_timepoints = np.linspace(0, tmax, num=len(stages_temperatures), endpoint=False)
assert len(stages_temperatures) == len(stages_timepoints)
e.server_settings.is_slurm = False
e.const_params["steps"] = tmax
e.const_params["run_serial"] = True
stages = dict()
for i, (T, step) in enumerate(zip(stages_temperatures, stages_timepoints)):
    stage_name = f"stage{i+1}"
    if i == 0:
        stages[stage_name] = StageInfoParam(stage_name,
                                            t=step,
                                            T=T,
                                            add_method=RandParticleAdder({
                                                ptype.name(): e.get_param(ptype.name())
                                                for ptype in e.sim_get_particles_set(()).particles()
                                            }))
    else:
        stages[stage_name] = StageInfoParam(stage_name,
                                            t=step,
                                            T=T)
e.add_params(STAGES_KEY=StagedAssemblyParam(
    staging_val_name="default",
    staging_info=stages
))
e.dump_metadata()
def run_single_sim(sim):
    while e.sim_get_next_stage(sim) is not None:
        e.start_simulation(sim)

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_single_sim, sim) for sim in e.ensemble()]
    for future in as_completed(futures):
        future.result()  # Raise any exceptions if they occurred
print("Simulations complete!")
