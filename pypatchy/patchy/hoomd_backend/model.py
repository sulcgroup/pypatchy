"""
The HOOMD engine's patchy particle model, and a numpy implementation of its energies.

Each pypatchy particle type is a HOOMD rigid body: a central particle carrying one massless constituent site per
patch, at the patch's (body-frame) position. Sites have one HOOMD type per patch colour (and strength), so colour
specificity is just a table of pair coefficients:

- particle centres repel with a WCA potential, sigma = r_i + r_j (the pypatchy particle radii),
  epsilon = `particle_epsilon`
- two sites attract when their patches can bind, with a shifted Gaussian well
  V(r) = -eps_ab * [exp(-r^2 / 2 patch_sigma^2) - exp(-patch_rcut^2 / 2 patch_sigma^2)],  r < patch_rcut
  eps_ab = patch_epsilon * w(c_a, c_b) * sqrt(s_a * s_b), where w is the particle set's interaction matrix
  entry (or 1 for colours that bind by pypatchy's default rule, if the set has no interaction matrix) and s are
  the patch strengths (multidentate conversion scales these per tooth)
- nothing else interacts (sites on the same body are excluded; sites don't feel other bodies' centres)

Patches are isotropic sites, so a single patch doesn't constrain torsion. Multidentate patches (several sites per
patch) do.

`RigidPatchyModel` holds the model as plain data (it is written into each stage's stage.json, and pickles), sets
up the HOOMD forces for a simulation, and implements the energy session protocol PLPSimulation uses for scene
operations (placing particles, resizing boxes, finding bonds) with the same potential in numpy.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..pl.plparticle import PLPatchyParticle
    from ..pl.plparticleset import PLParticleSet
    from ..pl.plscene import PLPSimulation

WCA_CUTOFF_FACTOR = 2 ** (1 / 6)
# body mass; constituent sites are massless
BODY_MASS = 1.0


# ---------------------------------------------------------------------------------------------- rotations
def rotation_matrix(particle: PLPatchyParticle) -> np.ndarray:
    """
    Lab-frame rotation matrix of a pypatchy particle: columns a1, a2, a3, so that the lab-frame position of a
    patch is rotation_matrix(p) @ patch.position() + p.position() (the same as PLPatchyParticle.patch_position)
    """
    if particle.a1 is None or particle.a3 is None:
        raise ValueError(f"Particle {particle.get_uid()} has no orientation (a1/a3 unset)")
    return particle.rotmatrix().T


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """
    :return: unit quaternion (w, x, y, z), HOOMD's convention, for rotation matrix m
    """
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, av = a[0], np.asarray(a[1:])
    bw, bv = b[0], np.asarray(b[1:])
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def angmom_quat_from_body(q: np.ndarray, l_body: np.ndarray) -> np.ndarray:
    """
    HOOMD stores angular momentum as a quaternion conjugate to the orientation, p = 2 q (0, L), where L is the
    body-frame angular momentum (the frame oxDNA, and so pypatchy's PLPatchyParticle.L, uses)
    """
    return 2 * quat_multiply(q, np.concatenate([[0.], l_body]))


def body_angmom_from_quat(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    inverse of angmom_quat_from_body: L = vec(conj(q) p) / 2
    """
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_multiply(q_conj, p)[1:] / 2


# ---------------------------------------------------------------------------------------------- the model
class RigidPatchyModel:
    """
    The HOOMD engine's patchy model for one particle set and set of model parameters. Plain data: see to_dict().
    """
    bodies: list[dict[str, Any]]  # one per particle type, in type id order
    sites: list[dict[str, Any]]  # one per site type: name, color, strength
    site_epsilon: np.ndarray  # (n_sites, n_sites) well depths, 0 where sites don't attract
    patch_sigma: float
    patch_rcut: float
    particle_epsilon: float

    def __init__(self,
                 bodies: list[dict[str, Any]],
                 sites: list[dict[str, Any]],
                 site_epsilon: Union[np.ndarray, list],
                 patch_sigma: float,
                 patch_rcut: float,
                 particle_epsilon: float):
        self.bodies = bodies
        self.sites = sites
        self.site_epsilon = np.asarray(site_epsilon, dtype=float).reshape(len(sites), len(sites))
        self.patch_sigma = float(patch_sigma)
        self.patch_rcut = float(patch_rcut)
        self.particle_epsilon = float(particle_epsilon)
        self._index()

    def _index(self):
        site_idx = {s["name"]: i for i, s in enumerate(self.sites)}
        self._type_index = {b["type_id"]: i for i, b in enumerate(self.bodies)}
        self._radii = np.array([b["radius"] for b in self.bodies], dtype=float)
        self._site_types = [np.array([site_idx[s["type"]] for s in b["sites"]], dtype=int) for b in self.bodies]
        self._site_pos = [np.array([s["position"] for s in b["sites"]], dtype=float).reshape(-1, 3)
                          for b in self.bodies]
        self._site_reach = np.array([np.linalg.norm(p, axis=1).max() if len(p) else 0. for p in self._site_pos])
        self._gauss_shift = math.exp(-self.patch_rcut ** 2 / (2 * self.patch_sigma ** 2))

    # -------------------------------------------------------------------------------------- construction
    @classmethod
    def from_particle_set(cls,
                          particle_set: PLParticleSet,
                          patch_epsilon: float,
                          patch_sigma: float,
                          patch_rcut: float,
                          particle_epsilon: float = 1.0) -> RigidPatchyModel:
        intmat = particle_set.interaction_matrix()
        use_intmat = intmat is not None and len(intmat.intmap()) > 0

        # one site type per (color, strength)
        keys = sorted({(patch.color(), float(patch.strength())) for patch in particle_set.patches()})
        per_color = {}
        for color, _ in keys:
            per_color[color] = per_color.get(color, 0) + 1
        sites, site_names = [], {}
        seen = {}
        for color, strength in keys:
            name = f"patch{color:+d}"
            if per_color[color] > 1:
                seen[color] = seen.get(color, 0) + 1
                name += f".{seen[color]}"
            site_names[(color, strength)] = name
            sites.append({"name": name, "color": color, "strength": strength})

        eps = np.zeros((len(sites), len(sites)))
        for i, a in enumerate(sites):
            for j, b in enumerate(sites):
                if use_intmat:
                    w = intmat[a["color"], b["color"]]
                else:
                    from ..pl.plpatch import PLPatch
                    w = 1. if PLPatch(color=a["color"]).can_bind(PLPatch(color=b["color"])) else 0.
                eps[i, j] = patch_epsilon * w * math.sqrt(a["strength"] * b["strength"])

        particles = sorted(particle_set.particles(), key=lambda p: p.type_id())
        names = [p.name() for p in particles]
        if len(set(names)) != len(names) or set(names) & set(site_names.values()):
            names = [f"P{p.type_id()}" for p in particles]
        bodies = []
        for name, p in zip(names, particles):
            radius = float(p.radius())
            bodies.append({
                "name": name,
                "type_id": int(p.type_id()),
                "radius": radius,
                "mass": BODY_MASS,
                # solid sphere
                "moment_inertia": [0.4 * BODY_MASS * radius ** 2] * 3,
                "sites": [{"type": site_names[(patch.color(), float(patch.strength()))],
                           "position": [float(x) for x in patch.position()]}
                          for patch in p.patches()],
            })
        return cls(bodies, sites, eps, patch_sigma, patch_rcut, particle_epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bodies": self.bodies,
            "sites": self.sites,
            "site_epsilon": self.site_epsilon.tolist(),
            "patch_sigma": self.patch_sigma,
            "patch_rcut": self.patch_rcut,
            "particle_epsilon": self.particle_epsilon,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RigidPatchyModel:
        return cls(**d)

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return f"RigidPatchyModel({len(self.bodies)} body types, {len(self.sites)} site types)"

    # -------------------------------------------------------------------------------------- lookups
    def body_type_names(self) -> list[str]:
        return [b["name"] for b in self.bodies]

    def hoomd_types(self) -> list[str]:
        """
        HOOMD particle types: body centres (typeid = position in self.bodies), then sites
        """
        return self.body_type_names() + [s["name"] for s in self.sites]

    def body_index(self, type_id: int) -> int:
        return self._type_index[type_id]

    def wca_sigma(self, i: int, j: int) -> float:
        return float(self._radii[i] + self._radii[j])

    def screen_radius(self) -> float:
        """
        largest centre-centre distance at which two particles can interact
        """
        wca = WCA_CUTOFF_FACTOR * 2 * self._radii.max()
        patch = 2 * self._site_reach.max() + self.patch_rcut
        return float(max(wca, patch))

    def patch_reach(self) -> float:
        """
        :return: the largest distance at which two patches can be bonded in this model
        """
        return self.patch_rcut

    # -------------------------------------------------------------------------------------- energies
    def wca(self, r: Union[float, np.ndarray], sigma: float) -> Union[float, np.ndarray]:
        r = np.asarray(r, dtype=float)
        sr6 = (sigma / np.where(r > 0, r, np.inf)) ** 6
        e = 4 * self.particle_epsilon * (sr6 * sr6 - sr6) + self.particle_epsilon
        return np.where(r < WCA_CUTOFF_FACTOR * sigma, e, 0.)

    def patch_well(self, r2: np.ndarray, eps: np.ndarray) -> np.ndarray:
        e = -eps * (np.exp(-r2 / (2 * self.patch_sigma ** 2)) - self._gauss_shift)
        return np.where(r2 < self.patch_rcut ** 2, e, 0.)

    def pair_energy(self,
                    i: int, pos_i: np.ndarray, rot_i: np.ndarray,
                    j: int, pos_j: np.ndarray, rot_j: np.ndarray,
                    box: np.ndarray) -> float:
        """
        Interaction energy of two particles.
        :param i, j: body indices (see body_index)
        :param pos_i, pos_j: centre positions
        :param rot_i, rot_j: lab-frame rotation matrices (see rotation_matrix)
        :param box: periodic box side lengths
        """
        d = pos_j - pos_i
        d = d - box * np.round(d / box)
        r = float(np.linalg.norm(d))
        e = float(self.wca(r, self.wca_sigma(i, j)))
        if r > self._site_reach[i] + self._site_reach[j] + self.patch_rcut:
            return e
        if len(self._site_types[i]) == 0 or len(self._site_types[j]) == 0:
            return e
        sites_i = self._site_pos[i] @ rot_i.T
        sites_j = self._site_pos[j] @ rot_j.T + d
        delta = sites_j[None, :, :] - sites_i[:, None, :]
        r2 = np.einsum("abk,abk->ab", delta, delta)
        eps = self.site_epsilon[np.ix_(self._site_types[i], self._site_types[j])]
        return e + float(self.patch_well(r2, eps).sum())

    def session(self,
                scene: PLPSimulation,
                extra_particles: Iterable[PLPatchyParticle] = ()) -> RigidPatchyEnergySession:
        return RigidPatchyEnergySession(self, scene, extra_particles)


class RigidPatchyEnergySession:
    """
    Energies of a scene under a RigidPatchyModel, computed in numpy. Implements the same protocol as
    OxpyEnergySession (particles addressed by pypatchy uid); unlike it, any number may be open at once.
    """

    def __init__(self, model: RigidPatchyModel, scene: PLPSimulation, extra_particles: Iterable[PLPatchyParticle] = ()):
        self.model = model
        self.scene = scene
        self.extra_particles = list(extra_particles)

    def __enter__(self) -> RigidPatchyEnergySession:
        particles = list(self.scene.particles()) + self.extra_particles
        uids = [p.get_uid() for p in particles]
        if len(set(uids)) != len(uids):
            raise ValueError("Particles passed to an energy session must have unique uids")
        unoriented = [p.get_uid() for p in particles if p.a1 is None or p.a3 is None]
        if unoriented:
            raise ValueError(f"Particles {unoriented[:10]} have no orientation (a1/a3 unset); "
                             f"set one (e.g. set_rotation / set_random_orientation) before computing energies")
        self._uids = uids
        self._row = {uid: i for i, uid in enumerate(uids)}
        self._types = np.array([self.model.body_index(p.get_type()) for p in particles], dtype=int)
        self._pos = np.array([p.position() for p in particles], dtype=float).reshape(-1, 3)
        self._rot = np.array([rotation_matrix(p) for p in particles]).reshape(-1, 3, 3)
        self._box = np.asarray(self.scene.box_size(), dtype=float)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def uids(self) -> list[int]:
        return list(self._uids)

    def update_particle(self, particle: PLPatchyParticle):
        row = self._row[particle.get_uid()]
        self._pos[row] = particle.position()
        self._rot[row] = rotation_matrix(particle)

    def positions(self, uids: Iterable[int]) -> np.ndarray:
        return self._pos[[self._row[u] for u in uids]].reshape(-1, 3)

    def neighbours(self, position: np.ndarray, candidates: Iterable[int]) -> list[int]:
        """
        :return: uids from `candidates` whose centres are within interaction range of `position`
        """
        candidates = list(candidates)
        if not candidates:
            return []
        delta = self.positions(candidates) - np.asarray(position)
        delta -= self._box * np.round(delta / self._box)
        close = np.einsum("ij,ij->i", delta, delta) <= self.model.screen_radius() ** 2
        return [uid for uid, c in zip(candidates, close) if c]

    def pair_energy(self, uid1: int, uid2: int) -> float:
        a, b = self._row[uid1], self._row[uid2]
        return self.model.pair_energy(self._types[a], self._pos[a], self._rot[a],
                                      self._types[b], self._pos[b], self._rot[b], self._box)

    def energy_with(self, uid: int, others: Iterable[int]) -> float:
        return sum(self.pair_energy(uid, other) for other in others if other != uid)

    def close_pairs(self, uids: Union[None, Iterable[int]] = None) -> list[tuple[int, int]]:
        """
        :return: pairs (uid1, uid2), uid1 < uid2, of the given particles (default: all) within interaction range
        """
        from scipy.spatial import cKDTree
        uids = sorted(self._uids if uids is None else uids)
        if len(uids) < 2:
            return []
        pos = np.mod(self.positions(uids), self._box)
        pos[pos >= self._box] = 0.
        pairs = cKDTree(pos, boxsize=self._box).query_pairs(self.model.screen_radius(), output_type="ndarray")
        return [(uids[i], uids[j]) if uids[i] < uids[j] else (uids[j], uids[i]) for i, j in pairs]

    def pair_energies(self, uids: Union[None, Iterable[int]] = None) -> dict[tuple[int, int], float]:
        """
        :return: map of (uid1, uid2) -> energy, uid1 < uid2, for pairs of the given particles (default: all scene
            particles) with nonzero energy
        """
        if uids is None:
            uids = [p.get_uid() for p in self.scene.particles()]
        energies = {}
        for u1, u2 in self.close_pairs(uids):
            e = self.pair_energy(u1, u2)
            if e != 0.:
                energies[(u1, u2)] = e
        return energies

    def total_energy(self) -> float:
        return sum(self.pair_energies().values())
