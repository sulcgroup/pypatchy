"""
Conversion between pypatchy scenes (PLPSimulation) and HOOMD GSD frames.

Frames hold rigid body centres only, one per pypatchy particle, in scene order (GSD index = particle uid); HOOMD
recreates the constituent sites from the body definitions when a stage starts. pypatchy boxes span [0, L) while
HOOMD's are centred on the origin, so positions are shifted by L/2 (and wrapped into the box).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np

from ..pl.plparticle import PLPatchyParticle
from ..pl.plparticleset import PLParticleSet
from ..pl.plscene import PLPSimulation
from .model import (RigidPatchyModel, angmom_quat_from_body, body_angmom_from_quat, matrix_from_quat,
                    quat_from_matrix, rotation_matrix)


def scene_to_frame(scene: PLPSimulation, model: RigidPatchyModel, step: int):
    """
    :return: (gsd.hoomd.Frame holding the scene's particles as body centres, GSD indices of the particles that
        have no velocity or angular momentum yet - e.g. ones just added - which a run should thermalize)
    """
    import gsd.hoomd

    particles = list(scene.particles())
    if [p.get_uid() for p in particles] != list(range(len(particles))):
        raise ValueError("Scene particle uids must be 0..N-1, in order, to be written to GSD")
    box = np.asarray(scene.box_size(), dtype=float)
    n = len(particles)

    frame = gsd.hoomd.Frame()
    frame.configuration.step = int(step)
    frame.configuration.box = [*box, 0., 0., 0.]
    frame.particles.N = n
    frame.particles.types = model.hoomd_types()
    frame.particles.typeid = np.array([model.body_index(p.get_type()) for p in particles], dtype=np.uint32)

    shifted = np.array([p.position() for p in particles], dtype=float).reshape(-1, 3) - box / 2
    image = np.round(shifted / box)
    frame.particles.position = shifted - image * box
    frame.particles.image = image.astype(np.int32)

    orientation = np.array([quat_from_matrix(rotation_matrix(p)) for p in particles]).reshape(-1, 4)
    frame.particles.orientation = orientation
    bodies = [model.bodies[i] for i in frame.particles.typeid]
    frame.particles.mass = np.array([b["mass"] for b in bodies], dtype=float)
    frame.particles.moment_inertia = np.array([b["moment_inertia"] for b in bodies], dtype=float).reshape(-1, 3)
    frame.particles.diameter = np.array([2 * b["radius"] for b in bodies], dtype=float)
    # body tags are left unset: Rigid.create_bodies() assigns them when the stage starts

    velocity = np.zeros((n, 3))
    angmom = np.zeros((n, 4))
    unthermalized = []
    for i, p in enumerate(particles):
        if p.v is None or p.L is None:
            unthermalized.append(i)
            continue
        velocity[i] = p.v
        angmom[i] = angmom_quat_from_body(orientation[i], np.asarray(p.L, dtype=float))
    frame.particles.velocity = velocity
    frame.particles.angmom = angmom
    return frame, unthermalized


def frame_to_scene(frame, particle_set: PLParticleSet, model: RigidPatchyModel) -> PLPSimulation:
    """
    Reads the body centres of a GSD frame into a scene (velocities and body-frame angular momenta included)
    """
    n_bodies = len(model.bodies)
    typeid = np.asarray(frame.particles.typeid)
    centres = np.nonzero(typeid < n_bodies)[0]
    box = np.asarray(frame.configuration.box[:3], dtype=float)

    scene = PLPSimulation()
    scene.set_time(int(frame.configuration.step))
    scene.set_particle_types(particle_set)
    scene.set_box_size(box)
    scene.compute_cell_size(n_particles=len(centres))
    scene.apportion_cells()

    position = np.mod(np.asarray(frame.particles.position, dtype=float)[centres] + box / 2, box)
    position[position >= box] = 0.
    orientation = np.asarray(frame.particles.orientation, dtype=float)[centres]
    velocity = np.asarray(frame.particles.velocity, dtype=float)[centres]
    angmom = np.asarray(frame.particles.angmom, dtype=float)[centres]
    types_by_id = {p.type_id(): p for p in particle_set.particles()}
    for uid, idx in enumerate(centres):
        ptype: PLPatchyParticle = types_by_id[model.bodies[typeid[idx]]["type_id"]]
        p = PLPatchyParticle(patches=ptype.patches(),
                             particle_name=f"{ptype.name()}_{uid}",
                             type_id=ptype.type_id(),
                             index_=uid,
                             position=position[uid],
                             radius=ptype.radius())
        rot = matrix_from_quat(orientation[uid])
        p.set_orientation(rot[:, 0], rot[:, 2])
        p.v = velocity[uid]
        p.L = body_angmom_from_quat(orientation[uid], angmom[uid])
        scene.add_particle(p)
    return scene


def frame_rotations(frame, indices: np.ndarray) -> np.ndarray:
    """
    :return: (n, 3, 3) lab-frame rotation matrices of the given particles of a GSD frame
    """
    from scipy.spatial.transform import Rotation
    q = np.asarray(frame.particles.orientation, dtype=float)[indices]
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 3, 3)


def frame_cluster_graphs(frame, model: RigidPatchyModel, bond_energy: float = 0.) -> list[nx.Graph]:
    """
    Finds the clusters of patch-bonded particles in a GSD frame (see RigidPatchyModel.patch_bonds for what counts
    as a bond).
    :return: one graph per cluster of two or more particles. Nodes are particle indices (= scene uids) with a
        "particle_type" attribute (the pypatchy type id); edges have "patches", a list of the bonded patch pairs
        ((particle, patch id), (particle, patch id)) - one per tooth for multidentate patches - and "energy", their
        summed patch-patch energy
    """
    typeid = np.asarray(frame.particles.typeid)
    centres = np.nonzero(typeid < len(model.bodies))[0]
    body_types = typeid[centres]
    positions = np.asarray(frame.particles.position, dtype=float)[centres]
    box = np.asarray(frame.configuration.box[:3], dtype=float)
    bonds = model.patch_bonds(body_types, positions, frame_rotations(frame, centres), box, bond_energy)

    G = nx.Graph()
    for i, site_i, j, site_j, energy in bonds:
        pair = ((i, model.bodies[body_types[i]]["sites"][site_i].get("patch_id", site_i)),
                (j, model.bodies[body_types[j]]["sites"][site_j].get("patch_id", site_j)))
        if G.has_edge(i, j):
            G.edges[i, j]["patches"].append(pair)
            G.edges[i, j]["energy"] += energy
        else:
            G.add_edge(i, j, patches=[pair], energy=energy)
    for n in G.nodes:
        G.nodes[n]["particle_type"] = model.bodies[body_types[n]]["type_id"]
    return [G.subgraph(component).copy() for component in nx.connected_components(G)]


def write_frame(path: Union[str, Path], frame):
    import gsd.hoomd
    with gsd.hoomd.open(str(path), "w") as f:
        f.append(frame)


def read_frame(path: Union[str, Path], index: int = -1):
    """
    :return: frame `index` of a GSD file (default: the last one)
    :raises IndexError: if the file has no such frame
    """
    import gsd.hoomd
    with gsd.hoomd.open(str(path), "r") as f:
        if len(f) == 0:
            raise IndexError(f"{path} has no frames")
        return f[index]


def frame_steps(path: Union[str, Path]) -> list[int]:
    """
    :return: the step of every frame in a GSD file
    """
    import gsd.fl
    with gsd.fl.open(str(path), "r") as f:
        # GSD HOOMD schema: a chunk missing from a frame takes its value from frame 0, or the default (0)
        default = int(f.read_chunk(0, "configuration/step")[0]) \
            if f.nframes and f.chunk_exists(0, "configuration/step") else 0
        return [int(f.read_chunk(i, "configuration/step")[0]) if f.chunk_exists(i, "configuration/step")
                else default for i in range(f.nframes)]
