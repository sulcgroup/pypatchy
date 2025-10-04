from __future__ import annotations

import copy
from typing import Union

import numpy as np

from ..param_val_types import PARTICLE_TYPES_KEY
from ..mgl import MGLScene, MGLParticleSet
from .plparticle import PLPatchyParticle, PATCHY_NULL_A1, PATCHY_NULL_A3
from .plparticleset import PLParticleSet, PLSourceMap, MultidentateConvertSettings, decode_particles_dict
from .plpatch import PLPatch
from .plscene import PLPSimulation
from .patchyio import writer_options, get_writer
from ...polycubeutil.polycube_structure import PolycubeStructure
from ...polycubeutil.polycubesRule import PolycubesRule
from ...util import to_xyz, halfway_vector, normalize, get_input_dir


def polycube_rule_to_PL(particle_set: PolycubesRule) -> PLParticleSet:
    """
    converts a polycubes rule to a patchy particle set
    :param particle_set: the polycubes rule to convert to a particle set
    :return: the a patchy patchy particle set corresponding to the rule
    """
    particles: PLParticleSet = PLParticleSet()
    # iter particles
    for cube_type in particle_set:
        particle_patches = []
        # convert to pl patch
        for patch in cube_type.patches():
            relPosition = patch.position() / cube_type.radius()
            pl_color = patch.colornum() - 20 if patch.colornum() < 0 else patch.colornum() + 20
            assert patch.get_id() == particles.num_patches() + len(particle_patches)
            a1 = relPosition / np.linalg.norm(relPosition)
            # torsion
            if patch.has_torsion():
                # absolutely hate this
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1,
                                  patch.alignDir()) * np.array([PATCHY_NULL_A1,
                                                                -np.cross(PATCHY_NULL_A1, PATCHY_NULL_A3),
                                                                PATCHY_NULL_A3])
            # no torsion
            else:
                # absolutely hate this
                plpatch = PLPatch(patch.get_id(),
                                  pl_color,
                                  relPosition,
                                  a1) * np.array([PATCHY_NULL_A1,
                                                                -np.cross(PATCHY_NULL_A1, PATCHY_NULL_A3),
                                                                PATCHY_NULL_A3])
            particle_patches.append(plpatch)

        # convert to pl particle
        # reuse type ids here, unfortunately
        particle = PLPatchyParticle(type_id=cube_type.type_id(),
                                    particle_name=cube_type.name(),
                                    index_=cube_type.type_id())
        particle.set_patches(particle_patches)
        # apply starting rotation, for some reason
        # particle.normalize()

        particles.add_particle(particle)
    # FOR NOW polycubes rules always have opposite-color interactions
    for patch in particles.patches():
        c = patch.color()
        if c > 0:  # we don't need this line but it will save us from wasted func calls
            particles.interaction_matrix()[(c, -c)] = 1.0
    assert all([patch.color() in particles.interaction_matrix() for patch in particles.patches()])
    return particles


def PL_to_rule(particles: list[PLPatchyParticle], ) -> PolycubesRule:
    """
    Tries to convert the provided set of particles and patches to a Polycubes Rule.
    TODO: include dynamic effects!!!! somehow.
    todo: interaction matrix
    :param particles: a list of PLPatchyParticle objects
    :return: the set of particles provided as a Polycubes rule if possible. none if not possible
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
                   nteeth: Union[int, None] = None,
                   dental_radius: Union[float, None] = None,
                   pad_cubes: float = 0.05,
                   pad_edges: float = 0.1) -> PLPSimulation:
    """
    Converts a polycube to a patchy particle scene. Conversion parameters can be passed as a
    `MultidentateConvertSettings` object or as individual values
    :param polycube: the polycube structure to convert
    :param mdt_convert: conversion settings, as an object
    :param nteeth: number of teeth to create per face when performing a multidentate convert
    :param dental_radius: distance from the center of the face to the "tooth" patches when performing a multidentate convert
    :param pad_cubes: padding between cubes, relative to cube size
    :param pad_edges: padding along edges of simulation box, relative to box size
    """
    # construct empty patchy scene
    pl = PLPSimulation()
    # convert polycubes rule to patchy particles
    # convert to pl and then do multidentate, to maintain consistancy elsewhere
    pl_types = polycube_rule_to_PL(polycube.particle_types())
    # if we have multidentate settings
    if mdt_convert is not None or nteeth is not None:
        # backwards compatibility for passing tooth info directly
        if mdt_convert is None:
            # construct convert settings from nteeth and dental radius
            mdt_convert = MultidentateConvertSettings(n_teeth=nteeth, dental_radius=dental_radius)
        # else, mdt convert has already been set
        # convert
        pl_types = pl_types.to_multidentate(mdt_convert)
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
        particle.set_rot_identity()
        # for reasons I struggle to comprehend, the "default" pl particle rotation is not identity
        # polycube_rule_to_pl accounts for tis but to make our code work going fwd we need to adjust
        # again, brb gonna kms
        particle.a1 = PATCHY_NULL_A1
        particle.a3 = PATCHY_NULL_A3
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

    # # verify (actually please don't, this blows up the comptuer for large structures)
    # also it won't work bc we don't have any interaction potentials
    # for cube1, cube2 in polycube.iter_bound_particles():
    #     assert pl.particles_bound(cube1.get_uid(), cube2.get_uid()), f"Cubes {cube1.get_uid()} and {cube2.get_uid()} are bound in the polycube but not in the patchy scene"
    #     cube_bindngs_count = len(list(polycube.iter_binding_patches(cube1, cube2)))
    #     pl_bindings_count = len(
    #         list(pl.iter_binding_patches(pl.get_particle(cube1.get_uid()), pl.get_particle(cube2.get_uid()))))
    #     assert pl_bindings_count > 0, f"Particles {cube2.get_uid()} amd {cube2.get_uid()} are bound but no patch bond found!"
    #     if nteeth is not None:
    #         assert nteeth * cube_bindngs_count == pl_bindings_count,\
    #             f"Mismatch between number bonds between {cube1.get_uid()} and {cube2.get_uid()} " \
    #             f"from polycube to patchy."
    #     elif mdt_convert is not None:
    #         assert mdt_convert.n_teeth * cube_bindngs_count == pl_bindings_count,\
    #             f"Mismatch between number bonds between {cube1.get_uid()} and {cube2.get_uid()} " \
    #             f"from polycube to patchy."
    return pl


class MGLPLSourceMap(PLSourceMap):
    """
    A conversion object to mediate conversion between MGL and PL Patchy format
    I don't think this is currently used? TODO: write a test script for it
    """
    # maps mgl particle color to PL type
    __color_type_map: dict[str, int]
    # maps mgl patch color string to PL patch color ints
    __patch_color_map: dict[str, int]

    def __init__(self,
                 src: MGLParticleSet,
                 colorTypeMap: dict[str, int],
                 patchColorMap: dict[str, int]):
        """
        :param src: source MGL particle set
        :param colorTypeMap: mapping of colors in MGL to color int values
        :param patchColorMap:
        """
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
    """

    """
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

    pset = PLParticleSet(MGLPLSourceMap(mgl_particles,
                                        particle_type_colormap,
                                        patch_color_map), intmat=particle_type_list)

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
    """
    Converts an MGL scene to a patchy particle conf
    :param mgl: MGL scene to convert
    :param pad_frac: padding of box size
    """
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
        particle.set_uid(mgl_particle.get_uid())
        particle.set_position(mgl_particle.position())
        # things get messy here, because we can't assume the mgl rotations are correct
        # in fact they're almost certainly not
        rot = pl_type.rotation_from_to(mgl_particle,
                                       pset.get_src_map().colormap())
        assert rot is not False, f"Cannot rotate particle {particle.get_uid()} to match particle type {pl_type.type_id()}"
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
    loads a particle sets from a dict, format negotiable
    todo: write unit tests
    """
    try:
        if "format" in kwargs:
            if kwargs["format"] in writer_options():
                writer = get_writer(kwargs["format"])
                writer.set_directory(get_input_dir())
                return writer.read_particle_types(**{
                    key: kwargs[key] for key in kwargs if key != "format"
                })  # todo: error handling
            else:
                raise Exception(f"Invalid writer {kwargs['format']}")
        else:
            assert PARTICLE_TYPES_KEY in kwargs and "patches" in kwargs, "No writer or particle/patches info specified!"
            return decode_particles_dict(**kwargs)

    except ValueError as e:
        raise ValueError(f"Invalid particle load info! {str(e)}")

