from __future__ import annotations

import copy
import math
from typing import Union

import numpy as np

from ..ensemble_parameter import PARTICLE_TYPES_KEY
from ..mgl import MGLParticle, MGLScene, MGLParticleSet
from .plparticle import PLPatchyParticle
from .plparticleset import PLParticleSet, PLSourceMap, MultidentateConvertSettings
from .plpatch import PLPatch
from .plscene import PLPSimulation
from ...patchy_base_particle import BaseParticleSet
from pypatchy.patchy.pl.patchyio import writer_options, get_writer
from ...polycubeutil.polycube_structure import PolycubeStructure
from ...polycubeutil.polycubesRule import PolycubesRule
from ...util import to_xyz, halfway_vector, normalize, get_input_dir


def to_PL(particle_set: BaseParticleSet,
          mdt_convert: Union[MultidentateConvertSettings, None] = None,
          num_teeth: int = 1,
          dental_radius: float = 0.) -> PLParticleSet:
    """
    I will freely admit I have no idea what "PL" means
    Whatever it means, this function converts an arbitrary particle set into
    PL particles and patches, applying multidentate where applicable.
    """
    # in the unlikely event this is already a PL particle set
    if isinstance(particle_set, PLParticleSet):
        if num_teeth == 1 and dental_radius == 0:
            return particle_set
        else:
            particles = particle_set
    elif isinstance(particle_set, PolycubesRule):
        particles = polycube_rule_to_PL(particle_set)
    elif all([isinstance(p, MGLParticle) for p in particle_set]):
        particles, _ = mgl_particles_to_pl(particle_set)
    else:
        raise Exception("Unrecognized particle types!!")

    # do multidentate convert
    if mdt_convert is None:
        mdt_convert = MultidentateConvertSettings(n_teeth=num_teeth, dental_radius=dental_radius)

    if mdt_convert.n_teeth > 1:
        particles = particles.to_multidentate(mdt_convert)

    assert particles.num_particle_types() == particle_set.num_particle_types(), \
        f"Bad conversion! Number of particle types produced {particles.num_particle_types()} not equal to number of particles" \
        f" in input particle set {particle_set.num_particle_types()}!"

    return particles


def polycube_rule_to_PL(particle_set: PolycubesRule) -> PLParticleSet:
    particles: PLParticleSet = PLParticleSet()
    # iter particles
    for particle in particle_set:
        particle_patches = []
        # convert to pl patch
        for patch in particle.patches():
            relPosition = patch.position() / particle.radius()
            pl_color = patch.colornum() - 20 if patch.colornum() < 0 else patch.colornum() + 20
            assert patch.get_id() == particles.num_patches() + len(particle_patches)
            a1 = relPosition / np.linalg.norm(relPosition)
            # torsion
            if patch.has_torsion():
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1,
                                  patch.alignDir())
            # no torsion
            else:
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1)
            particle_patches.append(plpatch)

        # convert to pl particle
        # reuse type ids here, unfortunately
        particle = PLPatchyParticle(type_id=particle.type_id(), particle_name=particle.name(),
                                    index_=particle.type_id())
        particle.set_patches(particle_patches)

        particles.add_particle(particle)
    return particles


def PL_to_rule(particles: list[PLPatchyParticle], ) -> Union[None, PolycubesRule]:
    """
    Tries to convert the provided set of particles and patches to a Polycubes Rule.
    TODO: include dynamic effects!!!! somehow.
    Args:
        particles: a list of PLPatchyParticle objects
    Returns:
        the set of particles provided as a Polycubes rule if possible. none if not possible
    """
    return PolycubesRule(rule_json=[
        {
            "patches": [
                {
                    "dir": to_xyz(np.round((patch.position() / particle.radius()))),
                    "alignDir": to_xyz(np.round(patch.a2())),
                    "color": patch.color() - 20 if patch.color() > 0 else patch.color() + 20,
                    "state_var": 0,
                    "activation_var": 0
                }
                for patch in particle.patches()]
        }
        for particle in particles])


def polycube_to_pl(polycube: PolycubeStructure,
                   mdt_convert: Union[MultidentateConvertSettings, None] = None,
                   nteeth=1,
                   dental_radius=0,
                   pad_cubes: float = 0.05,
                   pad_edges: float = 0.1) -> PLPSimulation:
    pl = PLPSimulation()
    # convert polycubes rule to multidentate patchy particles
    if mdt_convert is None:
        mdt_convert = MultidentateConvertSettings(n_teeth=nteeth, dental_radius=dental_radius)
    pl_types = to_PL(polycube.rule, mdt_convert)
    pl_types = pl_types.normalize()
    pl.set_particle_types(pl_types)
    mins = np.full(fill_value=np.inf, shape=3)
    maxs = np.full(fill_value=-np.inf, shape=3)
    # iter cubes in polycube
    cube_particles = []
    for cube in polycube.particles():
        pl_type: PLPatchyParticle = pl_types.particle(cube.get_type())
        particle = PLPatchyParticle(copy.deepcopy(pl_type.patches()),
                                    type_id=pl_type.type_id(),
                                    index_=cube.get_uid(),
                                    position=cube.position())
        # particle.a1 = POLYCUBE_NULL_A1
        # particle.a3 = POLYCUBE_NULL_A3
        particle.rotate(cube.rotation().as_matrix())
        particle.set_position(particle.position() * (1+pad_cubes))
        assert pl_type.matches(particle)
        cube_particles.append(particle)
        maxs = np.max([maxs, particle.position()], axis=0)
        mins = np.min([mins, particle.position()], axis=0)
    # compute box

    pad = (maxs - mins) * pad_edges + np.full(fill_value=1, shape=(3,))
    pl.set_box_size(maxs - mins + 2 * pad)

    pl.compute_cell_size(n_particles=len(cube_particles))
    pl.apportion_cells()
    for particle in cube_particles:
        particle.set_position(particle.position() - mins)
    pl.add_particles(cube_particles)


    # verify (actually please don't, this blows up the comptuer for large structures)
    # for cube1, cube2 in polycube.iter_bound_particles():
    #     assert pl.particles_bound(cube1.get_id(), cube2.get_id())
    #     cube_bindngs_count = len(list(polycube.iter_binding_patches(cube1, cube2)))
    #     pl_bindings_count = len(
    #         list(pl.iter_binding_patches(pl.get_particle(cube1.get_id()), pl.get_particle(cube2.get_id()))))
    #     # assert nteeth * cube_bindngs_count == pl_bindings_count

    return pl


class MGLPLSourceMap(PLSourceMap):
    # maps mgl particle color to PL type
    __color_type_map: dict[str, int]
    # maps mgl patch color string to PL patch color ints
    __patch_color_map: dict[str, int]

    def __init__(self,
                 src: MGLParticleSet,
                 colorTypeMap: dict[str, int],
                 patchColorMap: dict[str, int]):
        super().__init__(src)
        self.__color_type_map = colorTypeMap
        self.__patch_color_map = patchColorMap

    def particle_colormap(self) -> dict[str, int]:
        return self.__color_type_map

    def colored_particle_id(self, color: str) -> int:
        return self.__color_type_map[color]

    def colormap(self) -> dict[str, int]:
        return self.__patch_color_map

    # no need to do anything here
    def normalize(self) -> MGLPLSourceMap:
        return self


def mgl_particles_to_pl(mgl_particles: MGLParticleSet,
                        ref_scene: Union[MGLScene, None] = None) -> PLParticleSet:
    particle_type_list = []
    patch_uid = 0
    patch_color_map: dict[str, int] = {}
    color_counter = 1
    particle_type_colormap: dict[str, int] = {}
    for ptypeidx, mgl_ptype in enumerate(mgl_particles):
        pl_patches = []
        for mgl_patch in mgl_ptype.patches():
            # patch width information is inevitably lost
            # TODO: mgl -> pl mdt taking into account width

            # work out mgl patch colors
            # if color isn't in the map
            if mgl_patch.color() not in patch_color_map:
                # if the color is "dark[SOMETHING]"
                if mgl_patch.color().startswith("dark"):
                    color_str = mgl_patch.color()[4:]
                    # if the nondark version is in the map
                    if color_str in patch_color_map:
                        # add the dark version
                        patch_color_map[mgl_patch.color()] = -patch_color_map[color_str]
                    else:
                        # add the non-dark and dark version
                        patch_color_map[color_str] = color_counter
                        patch_color_map[mgl_patch.color()] = -color_counter
                        color_counter += 1
                else:
                    # just add the color
                    patch_color_map[mgl_patch.color()] = color_counter
                    color_counter += 1

            patch_color = patch_color_map[mgl_patch.color()]

            p = PLPatch(patch_uid,
                        patch_color,
                        mgl_patch.position(),
                        mgl_patch.position() / np.linalg.norm(mgl_patch.position()))
            pl_patches.append(p)
            patch_uid += 1
        pl_ptype = PLPatchyParticle(pl_patches,
                                    type_id=ptypeidx)
        particle_type_colormap[mgl_ptype.color()] = pl_ptype.get_type()
        particle_type_list.append(pl_ptype)

    pset = PLParticleSet(particle_type_list, MGLPLSourceMap(mgl_particles,
                                                            particle_type_colormap,
                                                            patch_color_map))

    # if we've provided a reference scene, use it to position A2 vectors (so we can convert multidentate later)
    if ref_scene is not None:
        # cry a lot
        handled_patches: set[int] = set()
        for (particle1, particle2) in ref_scene.iter_bound_particles():
            for patch1, patch2 in ref_scene.iter_binding_patches(particle1, particle2):
                # if we have handled all patches, just stop
                if len(handled_patches) == pset.num_patches():
                    break
                # get particle 1 type and rotation
                ptype1 = ref_scene.particle_types()[particle1.color()]
                p1rot = ref_scene.get_rot(particle1)

                # get particle 2 type and rotation
                ptype2 = ref_scene.particle_types()[particle2.color()]
                p2rot = ref_scene.get_rot(particle2)

                # get patch 1 and 2 type IDs
                ppatchtype1 = ptype1.patch(patch1.position() @ p1rot.T).get_id()
                ppatchtype2 = ptype2.patch(patch2.position() @ p2rot.T).get_id()
                if patch1.get_id() not in handled_patches and patch2.get_id() not in handled_patches:
                    # theta = math.pi - angle_between(patch1.position() @ particle1.rotation(),
                    #                                 patch2.position() @ particle2.rotation())
                    midvector = halfway_vector(patch1.position(),
                                               patch2.position())
                    # compute patch oris
                    patch1ori = normalize(np.cross(
                        patch1.position(),
                        midvector)
                    )
                    patch2ori = -normalize(np.cross(
                        patch2.position(),
                        midvector)
                    )

                    assert np.linalg.norm(patch1ori - patch2ori) < 1e-7, "Patch orientation vectors not orthogonal!"

                    pset.patch(ppatchtype1).set_a2(patch1ori @ p1rot.T)
                    pset.patch(ppatchtype2).set_a2(patch2ori @ p2rot.T)
                    handled_patches.add(ppatchtype1)
                    handled_patches.add(ppatchtype2)

                elif patch1.get_id() in handled_patches:
                    patch1ori = pset.patch(ppatchtype1).a2() @ p1rot.T
                    pset.patch(ppatchtype1).set_a2(patch1ori @ p2rot.T)

                    handled_patches.add(ppatchtype1)
                elif patch2.get_id() in handled_patches:
                    patch2ori = pset.patch(ppatchtype2).a2() @ p2rot.T
                    pset.patch(ppatchtype2).set_a2(patch2ori @ p1rot.T)

                    handled_patches.add(ppatchtype2)
        assert pset.num_patches() == len(handled_patches)

    return pset


def mgl_to_pl(mgl: MGLScene,
              pad_frac: float = 0.1) -> PLPSimulation:
    pl = PLPSimulation()
    pset = mgl_particles_to_pl(mgl.particle_types(), mgl)
    pset = pset.normalize()
    pl.set_particle_types(pset)

    mins = np.full(fill_value=np.inf, shape=3)
    maxs = np.full(fill_value=-np.inf, shape=3)
    pl.set_particle_types(pset)

    # convert scene
    for mgl_particle in mgl.particles():
        pl_type = pset.particle(pset.get_src_map().colored_particle_id(mgl_particle.color()))
        # particle = PLPatchyParticle(copy.deepcopy(pl_type.patches()),
        #                             type_id=pl_type.type_id(),
        #                             index_=mgl_particle.get_id(),
        #                             position=mgl_particle.position())
        particle: PLPatchyParticle = copy.deepcopy(pl_type)
        particle.set_uid(mgl_particle.get_id())
        particle.set_position(mgl_particle.position())
        # things get messy here, because we can't assume the mgl rotations are correct
        # in fact they're almost certainly not
        rot = pl_type.rotation_from_to(mgl_particle,
                                       pset.get_src_map().colormap())
        assert rot is not False, f"Cannot rotate particle {particle.get_id()} to match particle type {pl_type.type_id()}"
        particle.rotate(rot)
        pl.add_particle(particle)
        maxs = np.max([maxs, particle.position()], axis=0)
        mins = np.min([mins, particle.position()], axis=0)

    pad = (maxs - mins) * pad_frac + np.full(fill_value=1, shape=(3,))
    pl.translate(-mins + pad)
    pl.set_box_size(maxs - mins + 2 * pad)

    return pl


POLYCUBE_NULL_A1: np.ndarray = np.array([
    1,
    0,
    0
])
POLYCUBE_NULL_A3: np.ndarray = np.array([
    0,
    0,
    1
])


def load_pl_particles(**kwargs) -> PLParticleSet:
    """
    loads a particle sets from a dict, format negociable
    """
    try:
        if "format" in kwargs and kwargs["format"] in writer_options():
            writer = get_writer(kwargs["format"])
            writer.set_directory(get_input_dir())
            return writer.read_particle_types(**kwargs)  # todo: error handling
        else:
            assert PARTICLE_TYPES_KEY in kwargs and "patches" in kwargs, "No writer or particle/patches info specified!"
            particles_list = kwargs[PARTICLE_TYPES_KEY]
            patches_list = kwargs["patches"]
            # fix names
            for patch in patches_list:
                patch["type_id"] = patch["id"]
                del patch["id"]
                patch["relposition"] = patch["position"]
                del patch["position"]

            patches = [PLPatch(**p) for p in patches_list]
            particles_list = [PLPatchyParticle([patches[patch_id] for patch_id in p["patches"]],
                                               type_id=p["typex"])
                              for p in particles_list]
            return PLParticleSet(particles_list)
    except ValueError as e:
        raise ValueError(f"Invalid particle load info! {str(e)}")
