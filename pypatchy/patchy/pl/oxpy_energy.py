"""
Particle-pair energies and bonds for a PLPSimulation, computed by the actual oxDNA interaction (via oxpy)
instead of a Python re-implementation of it.

pypatchy used to carry its own Python copies of each oxDNA patchy potential (plpotential.py). Keeping them in sync
with the C++ plugins proved impractical, so energies now come straight from the plugin the simulation itself uses:
an `OxpyEnergyModel` says which interaction to load and with what parameters, and opening a session on a scene
writes that scene with the model's writer and loads it into an oxpy manager.

oxDNA keeps its configuration in process-wide singletons, so only one session can be open at a time.
"""
from __future__ import annotations

import copy
import itertools
import math
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .plscene import PLPSimulation
    from .plparticle import PLPatchyParticle
    from .plpatch import PLPatch

# parameters that pypatchy keeps in its param sets but that are not (or must not be) passed to an energy-only
# oxDNA run: pypatchy bookkeeping, outputs and backend selection
_EXCLUDED_INPUT_PARAMS = {
    "interaction_potentials",
    "backend", "backend_precision", "CUDA_list", "CUDA_device", "CUDA_sort_every", "cuda_mps",
    "use_edge", "max_density_multiplier",
    "trajectory_file", "lastconf_file", "energy_file", "log_file", "observables_file",
    "print_conf_interval", "print_energy_every", "no_stdout_energy",
    "topology", "conf_file", "steps", "restart_step_counter", "refresh_vel",
}

# the C++ plugins bond two patches only when they are within this distance of each other.
# raspberry: (alpha_p + alpha_q) * (ln(1.001) - ln(MIN_E)) ** 0.2, MIN_E = 1e-4 (RaspberryInteraction.cpp)
_RASPBERRY_REACH_FACTOR = (math.log(1.001) - math.log(1e-4)) ** 0.2
# flavio: hardcoded PATCHY_CUTOFF in PatchyShapeInteraction.h
_FLAVIO_PATCHY_CUTOFF = 0.18


class PatchBondAmbiguityWarning(UserWarning):
    """
    Raised when the patch-to-patch bonds inferred for a scene are not uniquely determined.

    oxpy reports energies per particle pair, so which patches form a bond between two bound particles is
    inferred from patch geometry (closest compatible patches within bonding reach). That inference is exact for
    clean bonds, but this warning flags every case where it may not be.
    """
    pass


class OxpyEnergyModel:
    """
    Picklable description of the oxDNA interaction used to compute a scene's energies.
    Holds no oxpy state itself: call `session()` to get energies for a particular scene.
    """
    patchy_format: str
    input_params: dict[str, Any]
    screen_radius: float
    _patch_reach: Union[float, None]
    repulsion_sites: Union[None, list[tuple[np.ndarray, float]]]
    type_rsite_lists: Union[None, dict[int, list[int]]]

    def __init__(self,
                 patchy_format: str,
                 input_params: dict[str, Any],
                 screen_radius: float = 3.0,
                 patch_reach: Union[float, None] = None,
                 repulsion_sites: Union[None, list[tuple[np.ndarray, float]]] = None,
                 type_rsite_lists: Union[None, dict[int, list[int]]] = None):
        """
        :param patchy_format: pypatchy writer key ("flavio", "lorenzo", "raspberry", ...) used to write scenes
        :param input_params: oxDNA input parameters for the interaction: interaction_type, plugin_search_path, T,
            and any model parameters (PATCHY_alpha, narrow_type, DPS_sigma_ss, ...). Output, backend and
            file-name parameters are ignored; energies are always computed on the CPU backend.
        :param screen_radius: particle pairs whose centers are further apart than this are assumed not to interact.
            oxpy does not expose the interaction cutoff, so this must be at least the largest pair cutoff of
            the model; the default (3.0) exceeds all of flavio, lorenzo and raspberry.
        :param patch_reach: max distance between two bonded patches. Inferred from the model when None.
        :param repulsion_sites: raspberry only: repulsion sites to write into the topology, if not the default
        :param type_rsite_lists: raspberry only: indices into repulsion_sites used by each particle type ID
            (default: every type uses every site)
        """
        self.patchy_format = patchy_format
        self.input_params = {key: val for key, val in input_params.items() if key not in _EXCLUDED_INPUT_PARAMS}
        assert "interaction_type" in self.input_params, "Energy model needs an `interaction_type`"
        self.screen_radius = screen_radius
        self._patch_reach = patch_reach
        self.repulsion_sites = repulsion_sites
        self.type_rsite_lists = type_rsite_lists

    def writer(self):
        from .patchyio import get_writer
        return get_writer(self.patchy_format)

    def patch_reach(self) -> float:
        """
        :return: the largest distance at which two patches can be bonded in this model
        """
        if self._patch_reach is not None:
            return self._patch_reach
        interaction = self.input_params["interaction_type"]
        if "Raspberry" in interaction:
            return 2 * float(self.input_params.get("PATCHY_alpha", 0.12)) * _RASPBERRY_REACH_FACTOR
        if "PatchyShape" in interaction:
            return _FLAVIO_PATCHY_CUTOFF
        if "PatchySwap" in interaction:
            return 1.5 * float(self.input_params.get("DPS_sigma_ss", 0.4))
        raise ValueError(f"Don't know the patch bonding reach of interaction `{interaction}`; "
                         f"pass patch_reach explicitly")

    def session(self,
                scene: PLPSimulation,
                extra_particles: Iterable[PLPatchyParticle] = ()) -> OxpyEnergySession:
        return OxpyEnergySession(self, scene, extra_particles)

    def __repr__(self):
        return f"OxpyEnergyModel({self.patchy_format}, {self.input_params.get('interaction_type')})"


class OxpyEnergySession:
    """
    Context manager holding an oxpy manager loaded with a scene. Particles are addressed by their pypatchy uid.

    Some plugins (raspberry, flavio with no_multipatch) keep patch bond "locks" between energy evaluations, so
    the energy of a pair can depend on which pairs were evaluated before it, exactly as within oxDNA itself.
    """
    _open: bool = False  # oxDNA state is global, so only one session may be open at once

    def __init__(self, model: OxpyEnergyModel, scene: PLPSimulation, extra_particles: Iterable[PLPatchyParticle] = ()):
        self.model = model
        self.scene = scene
        # particles that are written to oxDNA but not (yet) part of the scene, e.g. particles being placed
        self.extra_particles = list(extra_particles)
        self._tmpdir = None
        self._ctx = None
        self._manager = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> OxpyEnergySession:
        try:
            import oxpy
        except ImportError as e:
            raise ImportError("Computing patchy energies requires oxpy (the oxDNA python bindings)") from e
        if OxpyEnergySession._open:
            raise RuntimeError("An oxpy energy session is already open; oxDNA only supports one at a time")

        scene = self.scene
        all_particles = list(scene.particles()) + self.extra_particles
        uids = [p.get_uid() for p in all_particles]
        if len(set(uids)) != len(uids):
            raise ValueError("Particles passed to an energy session must have unique uids")
        unoriented = [p.get_uid() for p in all_particles if p.a1 is None or p.a3 is None]
        if unoriented:
            raise ValueError(f"Particles {unoriented[:10]} have no orientation (a1/a3 unset); "
                             f"set one (e.g. set_rotation / set_random_orientation) before computing energies")

        # build a scene copy sorted by particle type: some writers (raspberry, lorenzo) require type-sorted
        # particles, so sort here once and record the resulting order for every writer
        from .plscene import PLPSimulation
        ordered = sorted(all_particles, key=lambda p: p.get_type())  # stable: keeps scene order within a type
        self._uid_to_idx: dict[int, int] = {p.get_uid(): i for i, p in enumerate(ordered)}
        self._idx_to_uid: list[int] = [p.get_uid() for p in ordered]
        self._particles: dict[int, PLPatchyParticle] = {p.get_uid(): p for p in ordered}

        write_scene = PLPSimulation()
        write_scene.set_particle_types(scene.particle_types())
        write_scene.set_box_size(scene.box_size())
        write_scene.set_temperature(scene.T())
        write_scene.compute_cell_size(n_cells=1)
        write_scene.apportion_cells()
        box = np.asarray(scene.box_size(), dtype=float)
        for i, p in enumerate(ordered):
            p_copy = copy.deepcopy(p)
            p_copy.set_uid(i)
            # the scene's cell lists need in-box positions; wrapping is a no-op for oxDNA's periodic box
            wrapped = np.mod(np.asarray(p.position(), dtype=float), box)
            wrapped[wrapped >= box] = 0.
            p_copy.set_position(wrapped)
            write_scene.add_particle(p_copy)

        self._tmpdir = Path(tempfile.mkdtemp(prefix="pypatchy_oxpy_"))
        writer = self.model.writer()
        writer.set_directory(self._tmpdir)
        write_kwargs = {arg: f"{arg}.txt" for arg in writer.reqd_args()}
        write_kwargs["topology"] = "topology.top"
        write_kwargs["conf_file"] = "conf.dat"
        if self.model.repulsion_sites is not None or self.model.type_rsite_lists is not None:
            write_kwargs["scene_top"] = type(writer).RPatchyTopology(write_scene.particle_types(),
                                                                     write_scene.particle_type_counts(),
                                                                     self.model.repulsion_sites,
                                                                     self.model.type_rsite_lists)
        files = writer.write(write_scene, **write_kwargs)

        inputs: dict[str, Any] = {
            "sim_type": "MD",
            "dt": 0.001,
            "thermostat": "no",
            "verlet_skin": 0.05,
            "T": scene.T() if scene.T() > 0 else 0.1,
            **self.model.input_params,
            **dict(writer.get_input_file_data(write_scene)),
        }
        for key, val in files.items():
            if key in write_kwargs or key in ("topology", "conf_file"):
                path = Path(val)
                inputs[key] = str(path if path.is_absolute() else self._tmpdir / path)
        inputs.update({
            "backend": "CPU",
            # scenes built from polycubes / mgl files are often non-cubic
            "box_type": "orthogonal",
            "steps": 0,
            "restart_step_counter": 1,
            "trajectory_file": str(self._tmpdir / "trajectory.dat"),
            "lastconf_file": str(self._tmpdir / "last_conf.dat"),
            "energy_file": str(self._tmpdir / "energy.dat"),
            "log_file": str(self._tmpdir / "oxdna.log"),
            "no_stdout_energy": 1,
            "print_energy_every": 1000,
            "print_conf_interval": 1000000,
            "time_scale": "linear",
            "refresh_vel": 1,
        })
        with (self._tmpdir / "input").open("w") as f:
            for key, val in inputs.items():
                if isinstance(val, (bool, np.bool_)):
                    val = int(val)
                f.write(f"{key} = {val}\n")

        self._ctx = oxpy.Context(print_coda=False)
        self._ctx.__enter__()
        OxpyEnergySession._open = True
        # some plugins resolve files named inside the topology (e.g. lorenzo's patches_<type>.dat) against the
        # working directory, so load from inside the session directory
        cwd = os.getcwd()
        try:
            os.chdir(self._tmpdir)
            self._manager = oxpy.OxpyManager(str(self._tmpdir / "input"))
        except Exception:
            os.chdir(cwd)
            self.__exit__(None, None, None)
            raise
        os.chdir(cwd)
        self._ox_particles = self._manager.config_info().particles()
        self._interaction = self._manager.config_info().interaction
        self._box = np.array(self._manager.config_info().box_sides)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager = None
        self._ox_particles = None
        self._interaction = None
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
            self._ctx = None
        OxpyEnergySession._open = False
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        return False

    # ------------------------------------------------------------------ particle access
    def uids(self) -> list[int]:
        return list(self._idx_to_uid)

    def ox_particle(self, uid: int):
        return self._ox_particles[self._uid_to_idx[uid]]

    def update_particle(self, particle: PLPatchyParticle):
        """
        Pushes a pypatchy particle's current position and orientation into oxDNA
        """
        self._particles[particle.get_uid()] = particle
        ox_p = self.ox_particle(particle.get_uid())
        ox_p.pos = [float(x) for x in particle.position()]
        ox_p.set_orientation(_orientation_matrix(particle))

    def positions(self, uids: Iterable[int]) -> np.ndarray:
        return np.array([self._particles[uid].position() for uid in uids]).reshape(-1, 3)

    def neighbours(self, position: np.ndarray, candidates: Iterable[int]) -> list[int]:
        """
        :return: uids from `candidates` whose centers are within the screening radius of `position`
        """
        candidates = list(candidates)
        if not candidates:
            return []
        delta = self.positions(candidates) - np.asarray(position)
        delta -= self._box * np.round(delta / self._box)
        close = np.einsum("ij,ij->i", delta, delta) <= self.model.screen_radius ** 2
        return [uid for uid, c in zip(candidates, close) if c]

    # ------------------------------------------------------------------ energies
    def pair_energy(self, uid1: int, uid2: int) -> float:
        return self._interaction.pair_interaction(self.ox_particle(uid1), self.ox_particle(uid2))

    def energy_with(self, uid: int, others: Iterable[int]) -> float:
        """
        :return: summed interaction energy of particle `uid` with each particle in `others`
        """
        return sum(self.pair_energy(uid, other) for other in others if other != uid)

    def pair_energies(self, uids: Union[None, Iterable[int]] = None) -> dict[tuple[int, int], float]:
        """
        Evaluates every pair of the given particles (default: all scene particles) within the screening radius.
        :return: map of (uid1, uid2) -> energy, uid1 < uid2, for pairs with nonzero energy
        """
        uids = sorted(self._uid_to_idx if uids is None else uids, key=lambda u: self._uid_to_idx[u])
        self._interaction.begin_energy_computation()
        positions = self.positions(uids)
        energies = {}
        for i, uid in enumerate(uids[:-1]):
            delta = positions[i + 1:] - positions[i]
            delta -= self._box * np.round(delta / self._box)
            close = np.nonzero(np.einsum("ij,ij->i", delta, delta) <= self.model.screen_radius ** 2)[0]
            for j in close:
                other = uids[i + 1 + j]
                e = self.pair_energy(uid, other)
                if e != 0.:
                    energies[(min(uid, other), max(uid, other))] = e
        return energies

    def total_energy(self) -> float:
        """
        :return: total potential energy of the oxDNA system, computed by oxDNA itself
        """
        return self._manager.system_energy()


def _orientation_matrix(particle: PLPatchyParticle) -> np.ndarray:
    """
    oxDNA orientation matrix (columns a1, a2, a3, a2 = a3 x a1) of a pypatchy particle
    """
    a1 = np.asarray(particle.a1, dtype=float)
    a3 = np.asarray(particle.a3, dtype=float)
    return np.stack([a1, np.cross(a3, a1), a3], axis=1)


# ---------------------------------------------------------------------------------------- patch-level bonds
def patches_compatible(scene: PLPSimulation, patch1: PLPatch, patch2: PLPatch) -> bool:
    """
    Whether two patches can bind, according to the scene's interaction matrix if it has one,
    or the default color rule otherwise
    """
    intmat = scene.particle_types().interaction_matrix()
    if intmat is not None and len(intmat.intmap()) > 0:
        return intmat[patch1.color(), patch2.color()] != 0
    return patch1.can_bind(patch2)


def infer_patch_bonds(scene: PLPSimulation,
                      bonded_pairs: dict[tuple[int, int], float],
                      reach: float,
                      tie_tolerance: float = 0.1) -> list[dict]:
    """
    Works out which patches form the bonds between particle pairs that the interaction says are bound.

    Candidate bonds are compatible patch pairs within `reach` of each other on bound particle pairs. These are
    matched with a maximum-weight matching (closer pairs weigh more, as many bonds as possible) over the whole
    scene, so each patch takes part in at most one bond. Raises a single PatchBondAmbiguityWarning summarizing
    every case where the result is not clear-cut:
      - the pair is bound, but the particles have no compatible patches at all (no patch bond recorded)
      - the pair is bound, but no compatible patches are within reach (nearest compatible pair used instead)
      - near-tie: a chosen bond has a competing candidate (sharing one of its patches) at almost the same
        distance (within tie_tolerance * reach), so another pairing would fit the geometry just as well
      - the pair is bound, but all its candidate patches were matched to other particles (no patch bond recorded)

    :param bonded_pairs: map of (uid1, uid2) -> pair energy, for the pairs considered bound
    :return: list of bonds, each a dict with p1_uid, patch1 (patch id), p2_uid, patch2, distance, pair_energy,
        n_patch_bonds (number of patch bonds between that particle pair)
    """
    import networkx as nx

    box = scene.box_size()
    issues: dict[str, list[str]] = defaultdict(list)
    # candidate graph: nodes are (uid, patch id), edges are candidate patch bonds
    B = nx.Graph()
    pairs_with_candidates = set()
    for (uid1, uid2), pair_energy in bonded_pairs.items():
        p1, p2 = scene.get_particle(uid1), scene.get_particle(uid2)
        candidates = []
        for patch1, patch2 in itertools.product(p1.patches(), p2.patches()):
            if patches_compatible(scene, patch1, patch2):
                delta = p1.patch_position(patch1) - p2.patch_position(patch2)
                delta = delta - box * np.round(delta / box)
                candidates.append((float(np.linalg.norm(delta)), patch1.get_id(), patch2.get_id()))
        if not candidates:
            issues["bound particles with no compatible patches"].append(f"{uid1}-{uid2} (E={pair_energy:.3g})")
            continue
        candidates.sort()
        in_reach = [c for c in candidates if c[0] <= reach]
        if not in_reach:
            issues["bound particles with no compatible patches in bonding reach (closest pair used)"].append(
                f"{uid1}-{uid2} (closest {candidates[0][0]:.3g} > reach {reach:.3g})")
            in_reach = candidates[:1]
        pairs_with_candidates.add((uid1, uid2))
        for d, patch1, patch2 in in_reach:
            # positive weight, larger for closer patches
            B.add_edge((uid1, patch1), (uid2, patch2), weight=1. + max(reach, d) - d, distance=d,
                       pair=(uid1, uid2), pair_energy=pair_energy)

    # one patch, one bond: maximum-cardinality, maximum-weight matching per connected component
    matched: set[frozenset] = set()
    for component in nx.connected_components(B):
        sub = B.subgraph(component)
        edges = sub.edges() if sub.number_of_edges() == 1 else nx.max_weight_matching(sub, maxcardinality=True)
        matched.update(frozenset(e) for e in edges)

    bonds = []
    n_patch_bonds: dict[tuple[int, int], int] = defaultdict(int)
    for edge in matched:
        u, v = tuple(edge)
        data = B.edges[u, v]
        (uid_a, patch_a), (uid_b, patch_b) = (u, v) if u[0] == data["pair"][0] else (v, u)
        bonds.append({"p1_uid": uid_a, "patch1": patch_a, "p2_uid": uid_b, "patch2": patch_b,
                      "distance": data["distance"], "pair_energy": data["pair_energy"]})
        n_patch_bonds[data["pair"]] += 1
        # near-ties: an unmatched candidate sharing a patch with this bond, at almost the same distance
        for node, partner in ((u, v), (v, u)):
            for alt in B.neighbors(node):
                if alt == partner or frozenset((node, alt)) in matched:
                    continue
                d_alt = B.edges[node, alt]["distance"]
                if d_alt - data["distance"] < tie_tolerance * reach:
                    issues["near-tie between candidate patch pairings"].append(
                        f"particle {node[0]} patch {node[1]}: bonded to {partner} ({data['distance']:.3g}) "
                        f"vs {alt} ({d_alt:.3g})")
                    break
    for b in bonds:
        b["n_patch_bonds"] = n_patch_bonds[(b["p1_uid"], b["p2_uid"])]
    for pair in pairs_with_candidates:
        if n_patch_bonds[pair] == 0:
            issues["bound particles whose candidate patches all bonded elsewhere (no patch bond recorded)"].append(
                f"{pair[0]}-{pair[1]} (E={bonded_pairs[pair]:.3g})")

    if issues:
        summary = "; ".join(f"{len(examples)}x {kind} (e.g. {', '.join(examples[:3])})"
                            for kind, examples in issues.items())
        warnings.warn(f"Patch-level bonds are ambiguous: {summary}", PatchBondAmbiguityWarning, stacklevel=3)
    return bonds
