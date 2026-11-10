"""
Runs one HOOMD-engine stage from its directory:

    python -m pypatchy.patchy.hoomd_backend.run_stage <stage directory>

The directory, written by HOOMDEngine.build_stage(), holds everything the run needs: stage.json (resolved
parameters, the model and the stage's steps) and init.gsd (the starting configuration, body centres only). The
run writes trajectory.gsd (body centres every print_conf_interval steps), energy.txt (every print_energy_every
steps) and, once the stage's last step is reached, last.gsd. Nothing here reads the ensemble, so stages can be run
by anything that can run Python: in-process subprocesses, batch workers or sbatch jobs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Union

from .model import WCA_CUTOFF_FACTOR, RigidPatchyModel

STAGE_SPEC_FILE = "stage.json"
INIT_FILE = "init.gsd"
TRAJECTORY_FILE = "trajectory.gsd"
LAST_FILE = "last.gsd"
ENERGY_FILE = "energy.txt"
LOG_FILE = "log.log"


def read_spec(stage_dir: Union[str, Path]) -> dict[str, Any]:
    with (Path(stage_dir) / STAGE_SPEC_FILE).open() as f:
        return json.load(f)


def make_device(name: str):
    import hoomd
    if name == "gpu":
        return hoomd.device.GPU()
    if name == "cpu":
        return hoomd.device.CPU()
    raise ValueError(f"Unknown device {name!r}")


def build_simulation(spec: dict[str, Any], init_gsd: Union[str, Path], device=None):
    """
    Sets up (but doesn't run) the HOOMD simulation for a stage.
    :param spec: the stage's stage.json contents
    :param init_gsd: GSD file with the starting body centres (its last frame is used)
    :param device: HOOMD device, or None for the one the stage's parameters name
    :return: (hoomd.Simulation, dict of the operations it uses: rigid, nlist, wca, patch, langevin)
    """
    import hoomd
    from hoomd import md

    params = spec["params"]
    model = RigidPatchyModel.from_dict(spec["model"])
    if device is None:
        device = make_device(params["device"])
    sim = hoomd.Simulation(device=device, seed=params["seed"])
    sim.create_state_from_gsd(str(init_gsd), frame=-1)

    rigid = md.constrain.Rigid()
    for body in model.bodies:
        rigid.body[body["name"]] = {
            "constituent_types": [s["type"] for s in body["sites"]],
            "positions": [s["position"] for s in body["sites"]],
            "orientations": [(1., 0., 0., 0.)] * len(body["sites"]),
        }
    rigid.create_bodies(sim.state)

    nlist = md.nlist.Cell(buffer=params["nlist_buffer"], exclusions=("body",))
    wca = md.pair.LJ(nlist=nlist, default_r_cut=0., mode="shift")
    patch = md.pair.Gaussian(nlist=nlist, default_r_cut=0., mode="shift")
    types = model.hoomd_types()
    n_bodies = len(model.bodies)
    for a in range(len(types)):
        for b in range(a, len(types)):
            pair = (types[a], types[b])
            if a < n_bodies and b < n_bodies:
                sigma = model.wca_sigma(a, b)
                wca.params[pair] = dict(epsilon=model.particle_epsilon, sigma=sigma)
                wca.r_cut[pair] = WCA_CUTOFF_FACTOR * sigma
            else:
                wca.params[pair] = dict(epsilon=0., sigma=1.)
            eps = model.site_epsilon[a - n_bodies, b - n_bodies] if a >= n_bodies and b >= n_bodies else 0.
            if eps != 0.:
                patch.params[pair] = dict(epsilon=-eps, sigma=model.patch_sigma)
                patch.r_cut[pair] = model.patch_rcut
            else:
                patch.params[pair] = dict(epsilon=0., sigma=1.)

    langevin = md.methods.Langevin(filter=hoomd.filter.Rigid(("center", "free")),
                                   kT=params["kT"],
                                   default_gamma=params["gamma"],
                                   default_gamma_r=(params["gamma_r"],) * 3)
    integrator = md.Integrator(dt=params["dt"],
                               integrate_rotational_dof=True,
                               rigid=rigid,
                               forces=[wca, patch],
                               methods=[langevin])
    sim.operations.integrator = integrator
    return sim, {"rigid": rigid, "nlist": nlist, "wca": wca, "patch": patch, "langevin": langevin}


def run_stage(stage_dir: Union[str, Path], device=None):
    """
    Runs a stage from its directory (see the module docstring)
    """
    import hoomd
    from hoomd import md

    stage_dir = Path(stage_dir)
    spec = read_spec(stage_dir)
    params = spec["params"]
    start, end = int(spec["start_step"]), int(spec["end_step"])
    print(f"pypatchy HOOMD stage {spec.get('stage_name')} in {stage_dir}: steps {start} -> {end}", flush=True)
    print(f"HOOMD {hoomd.version.version}; parameters: {json.dumps(params)}", flush=True)
    if spec.get("defaulted"):
        print(f"Parameters left at their defaults: {', '.join(spec['defaulted'])}", flush=True)

    sim, ops = build_simulation(spec, stage_dir / INIT_FILE, device)
    if sim.timestep != start:
        raise RuntimeError(f"{INIT_FILE} is at step {sim.timestep}, but the stage starts at step {start}")
    # GPU devices list the GPUs they use; CPU devices have no such attribute
    print(f"Device: {getattr(sim.device, 'devices', type(sim.device).__name__)}", flush=True)

    centres = hoomd.filter.Rigid(("center", "free"))
    if spec.get("thermalize"):
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.Tags(spec["thermalize"]), kT=params["kT"])
        print(f"Thermalized the momenta of {len(spec['thermalize'])} new particles", flush=True)

    # potential energy is summed over every particle (the sites carry the patch energies); temperatures and
    # kinetic energies over the bodies, whose degrees of freedom the integrator counts
    pe = md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    thermo = md.compute.ThermodynamicQuantities(filter=centres)
    sim.operations.computes.extend([pe, thermo])

    traj = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(params["print_conf_interval"]),
                           filename=str(stage_dir / TRAJECTORY_FILE),
                           filter=centres,
                           mode="wb",
                           dynamic=["property", "momentum"])
    sim.operations.writers.append(traj)

    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(sim, quantities=["timestep", "tps"])
    # the two computes contribute different quantities, so their log keys don't collide
    logger.add(pe, quantities=["potential_energy"])
    logger.add(thermo, quantities=["kinetic_temperature", "kinetic_energy"])
    energy_out = (stage_dir / ENERGY_FILE).open("w")
    table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(params["print_energy_every"]),
                              logger=logger, output=energy_out)
    sim.operations.writers.append(table)

    try:
        sim.run(end - start, write_at_start=True)
        traj.flush()
        # write last.gsd atomically: its existence (at the end step) is what marks the stage done
        tmp = stage_dir / (LAST_FILE + ".tmp")
        hoomd.write.GSD.write(state=sim.state, filename=str(tmp), filter=centres, mode="wb", precision="double")
        os.replace(tmp, stage_dir / LAST_FILE)
    finally:
        energy_out.close()
    print(f"Finished at step {sim.timestep} ({sim.tps:.1f} steps/s)", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    run_stage(argv[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
