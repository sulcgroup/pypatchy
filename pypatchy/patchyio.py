from __future__ import annotations

import copy
from typing import Union, IO
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import oxDNA_analysis_tools.UTILS.RyeReader as rr

from pypatchy.patchy.patchy_scripts import to_PL
from pypatchy.patchy.plpatchy import PLPatchyParticle, export_interaction_matrix, PLPatch, PLPSimulation
from pypatchy.patchy.stage import Stage
from pypatchy.patchy_base_particle import PatchyBaseParticleType, Scene
from pypatchy.patchy_base_particle import BaseParticleSet
from pypatchy.polycubeutil.polycubesRule import PolycubeRuleCubeType
from pypatchy.util import get_server_config, PATCHY_FILE_FORMAT_KEY


# this approach was suggested by chatGPT.
# I kind of hate it but if it works it works
def custom_formatter(x):
    if abs(x) < 1e-9:
        return "0"
    if x.is_integer():
        return str(int(x))
    else:
        return str(x)


class BasePatchyWriter(ABC):
    """
    
    """

    _writing_directory: Path

    def __init__(self):
        pass

    @abstractmethod
    def get_input_file_data(self,
                            scene: Scene,
                            **kwargs) -> list[tuple[str, str]]:
        """
        Returns a set of tuples of str,str which should be added
        to oxDNA input files
        """

    def set_write_directory(self, directory):
        self._writing_directory = directory

    def directory(self) -> Path:
        return self._writing_directory

    def file(self, filename) -> IO:
        return open(self.directory() / filename, "w")

    @abstractmethod
    def reqd_args(self) -> list[str]:
        pass

    @abstractmethod
    def write(self,
              scene: Scene,
              stage: Union[Stage, None] = None,
              **kwargs
              ) -> dict[str, str]:
        """
        Returns:
            a tuple the first element of which is a list of the paths of files this method just created,
            the second of which is a dict to append to oxdna input file
        """
        pass

    @abstractmethod
    def write_top(self, scene: Scene, top_path: Union[str, Path]):
        pass

    def write_conf(self, scene: Scene, p: Path):
        assert scene.get_conf() is not None
        conf = scene.get_conf()
        rr.write_conf(str(p), conf)

    @abstractmethod
    def read_scene(self,
                   top_file: Union[Path, str],
                   traj_file: Union[Path, str],
                   particle_types: BaseParticleSet) -> Scene:
        pass


class FWriter(BasePatchyWriter):
    """
    Class for writing files in the Flavian style, which uses particles.txt, patches.txt, and the Coliseum
    """

    def get_input_file_data(self,
                            scene: Scene,
                            **kwargs) -> list[tuple[str, str]]:

        return [
            ("patchy_file", "patches.txt"),
            ("particle_file", "particles.txt"),
            ("particle_types_N", f"{scene.num_particle_types()}"),
            ("patch_types_N", f"{scene.particle_types().num_patches()}")
        ]

    def write_top(self, scene: Scene, top_path: Union[str, Path]):
        if isinstance(top_path, str):
            top_path = str(top_path)

        particle_set = scene.particle_types()

        particle_type_counts = scene.particle_type_counts()

        total_num_particles = sum(particle_type_counts.values())

        with top_path.open("w") as top_file:
            # write top file
            # first line of file
            top_file.write(f"{total_num_particles} {particle_set.num_particle_types()}\n")
            # second line of file
            top_file.write(" ".join([
                f"{particle.type_id()} " * particle_type_counts[particle.type_id()] for particle in
                particle_set.particles()
            ]))


    def particle_type_string(self, particle: PLPatchyParticle, extras: dict[str, str] = {}) -> str:
        outs = 'particle_%d = { \n type = %d \n ' % (particle.type_id(), particle.type_id())
        outs = outs + 'patches = '
        for i, p in enumerate(particle.patches()):
            outs = outs + str(p.get_id())
            if i < len(particle.patches()) - 1:
                outs = outs + ','
        outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
        outs = outs + ' \n } \n'
        return outs

    def write(self,
              scene: Scene,
              stage: Union[Stage, None] = None,
              **kwargs) -> dict[str, str]:
        particle_fn = kwargs["particle_file"]
        patchy_fn = kwargs["patchy_file"]
        if stage is not None:
            init_top = stage.adjfn(kwargs["topology"])
            init_conf = stage.adjfn(kwargs["conf_file"])
        else:
            init_top = kwargs["topology"]
            init_conf = kwargs["conf_file"]
        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)

        particles = scene.particle_types()
        particle_type_counts = scene.particle_type_counts()

        plparticles = to_PL(particles, 1, 0)

        total_num_particles = sum(particle_type_counts.values())
        with self.file(particle_fn) as particles_file, \
                self.file(patchy_fn) as patches_file:

            # swrite particles and patches file
            for particle_patchy, particle_type in zip(plparticles, particles.particles()):
                # handle writing particles file
                for patch_idx, patch_obj in enumerate(particle_patchy.patches()):
                    # we have to be VERY careful here with indexing to account for multidentate simulations
                    # adjust for patch multiplier from multidentate

                    patches_file.write(self.save_patch_to_str(patch_obj))
                particles_file.write(self.particle_type_string(particle_patchy))

        self.write_top(scene, self.directory() / init_top)

        # write conf
        self.write_conf(scene, self.directory() / init_conf)

        return {
            "particle_file": particle_fn,
            "patchy_file": patchy_fn,
            "topology": init_top,
            "init_conf": init_conf
        }

    def reqd_args(self) -> list[str]:
        return ["patchy_file", "particle_file", "init_conf", "topology"]

    def save_patch_to_str(self, patch, extras={}) -> str:
        # print self._type,self._type,self._color,1.0,self._position,self._a1,self._a2

        fmtargs = {
            # "precision": 3,
            "separator": ",",
            "suppress_small": True,
            "formatter": {'float_kind': custom_formatter}
        }

        position_str = np.array2string(patch.position(), **fmtargs)[1:-1]
        a1_str = np.array2string(patch.a1(), **fmtargs)[1:-1]
        if patch.a2() is not None:  # tolerate missing a2s
            a2_str = np.array2string(patch.a2(), **fmtargs)[1:-1]
        else:
            # make shit up
            a2_str = np.array2string(np.array([0, 0, 0]), **fmtargs)[1:-1]

        outs = f'patch_{patch.type_id()} = ' + '{\n ' \
                                               f'\tid = {patch.type_id()}\n' \
                                               f'\tcolor = {patch.color()}\n' \
                                               f'\tstrength = {patch.strength()}\n' \
                                               f'\tposition = {position_str}\n' \
                                               f'\ta1 = {a1_str}\n' \
                                               f'\ta2 = {a2_str}\n'

        outs += "\n".join([f"\t{key} = {extras[key]}" for key in extras])
        outs += "\n}\n"
        return outs

    def read_scene(self,
                   top_file: Path,
                   traj_file: Path,
                   particle_types: BaseParticleSet) -> Scene:
        top_info, traj_info = rr.describe(str(top_file), str(traj_file))
        # only retrieve last conf
        conf = rr.get_confs(top_info, traj_info, traj_info.nconfs - 1, 1)[0]
        scene = PLPSimulation()
        scene.set_time(conf.time)
        scene.set_particle_types(particle_types)
        scene.set_box_size(conf.box)
        with top_file.open("r") as f:
            f.readline()
            ptypelist = [int(i) for i in f.readline().split()]
            for i, ptype_idx in enumerate(ptypelist):
                ptype: PatchyBaseParticleType = particle_types.particle(ptype_idx)
                pp = PLPatchyParticle(
                    patches=ptype.patches(),
                    particle_name=f"{ptype.name()}_{i}",
                    type_id=ptype_idx,
                    index_=i,
                    position=conf.positions[i, :],
                )
                pp.set_orientation(conf.a1s[i, :], conf.a3s[i, :])
                scene.add_particle(pp)
        return scene

    #
    # def read_top(self, fp: Path, scene: Scene):
    #     """
    #     Reads a Flavian-style topology file
    #     """
    #     with fp.open("r") as handle:
    #         lines = handle.readlines()
    #         vals = lines[0].split()
    #         N = int(vals[0])
    #         type_counts = []
    #         # for c in range(len(patches.keys())):
    #         particles = []  # [None for x in range(N)]
    #
    #         for pid, line in enumerate(lines[1:]):
    #             if len(line.split()) >= 2:
    #                 vals = line.split()
    #                 count = int(vals[0])
    #                 type_counts.append(count)
    #                 pcount = int(vals[1])
    #                 ptypes = vals[2]
    #                 ptypes = [int(x) for x in ptypes.strip().split(',')]
    #                 pgeometry = vals[3]
    #                 patches = []
    #                 index = 0
    #                 for i, p in enumerate(ptypes):
    #                     patch = PLPatch()
    #                     patch.set_color(p)
    #                     patch.init_from_dps_file(pgeometry, i)
    #                     patches.append(patch)
    #                 for j in range(count):
    #                     particle = PLPatchyParticle()
    #                     particle._type = pid
    #                     particle._patch_ids = ptypes
    #                     particle._patches = patches
    #                     particle.unique_id = index
    #                     index += 1
    #                     particles.append(particle)
    #
    #         scene.add_particles(particles)


class JWriter(BasePatchyWriter, ABC):
    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY,
                DENTAL_RADIUS_KEY,
                "patchy_file",
                "particle_file",
                "topology",
                "conf_file",
                "particle_types"
                ]

    def get_input_file_data(self,
                            scene: Scene,
                            **kwargs) -> list[tuple[str, str]]:
        return [
            ("patchy_file", "patches.txt"),
            ("particle_file", "particles.txt"),
            ("particle_types_N", str(scene.num_particle_types())),
            ("patch_types_N", str(scene.particle_types().num_patches() * kwargs[NUM_TEETH_KEY]))
        ]

    @abstractmethod
    def get_patch_extras(self, particle_type: PatchyBaseParticleType, patch_idx: int) -> dict:
        pass

    @abstractmethod
    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        pass

    def write(self,
              scene: Scene,
              stage: Union[Stage, None] = None,
              **kwargs
              ) -> dict[str, str]:

        # file info
        particle_fn = kwargs["particle_file"]
        patchy_fn = kwargs["patchy_file"]
        if stage is not None:
            init_top = stage.adjfn(kwargs["topology"])
            init_conf = stage.adjfn(kwargs["conf_file"])
        else:
            init_top = kwargs["topology"]
            init_conf = kwargs["conf_file"]
        particles_type_list: BaseParticleSet = kwargs["particle_types"]

        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)
        particles = scene.particle_types()

        pl_set = to_PL(particles,
                       kwargs[NUM_TEETH_KEY],
                       kwargs[DENTAL_RADIUS_KEY])

        self.write_conf(scene, self.directory() / init_conf)

        with self.file(init_top) as top_file, \
                self.file(particle_fn) as particles_file, \
                self.file(patchy_fn) as patches_file:

            # write particles and patches file
            for particle_patchy, particle_type in zip(pl_set.particles(), particles_type_list.particles()):
                # handle writing particles file
                for i, patch_obj in enumerate(particle_patchy.patches()):
                    # we have to be VERY careful here with indexing to account for multidentate simulations
                    # adjust for patch multiplier from multidentate
                    patch_idx = int(i / kwargs[NUM_TEETH_KEY])
                    extradict = self.get_patch_extras(particle_type, patch_idx)
                    patches_file.write(self.save_patch_to_str(patch_obj, extradict))
                particles_file.write(self.get_particle_extras(particle_patchy, particle_type))

        # shorthand b/c i don't want to mess w/ scene object passed as param

        scene_cpy = copy.deepcopy(scene)
        scene_cpy.set_particle_types(pl_set)

        self.write_top(scene, init_top)

        self.write_conf(scene, self.directory() / init_conf)

        return {
            "topology": init_top,
            "particle_file": particle_fn,
            "patchy_file": patchy_fn,
            "conf_file": init_conf
        }


# inherit from FWriter so can use class methods
class JFWriter(JWriter, FWriter):

    def get_particle_extras(self, plpartcle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        return self.particle_type_string(plpartcle)

    def get_patch_extras(self, particle_type: PolycubeRuleCubeType, patch_idx: int) -> dict:
        allo_conditional = particle_type.patch_conditional(
            particle_type.get_patch_by_idx(patch_idx), minimize=True)
        # allosteric conditional should be "true" for non-allosterically-controlled patches
        return {"allostery_conditional": allo_conditional if allo_conditional else "true"}


class JLWriter(JWriter):

    def particle_type_string(self, particle: PLPatchyParticle, extras: dict[str, str] = {}) -> str:
        outs = 'particle_%d = { \n type = %d \n ' % (particle.type_id(), particle.type_id())
        outs = outs + 'patches = '
        for i, p in enumerate(particle.patches()):
            outs = outs + str(p.get_id())
            if i < len(particle.patches()) - 1:
                outs = outs + ','
        outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
        outs = outs + ' \n } \n'
        return outs

    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        return self.particle_type_string(plparticle, {"state_size": particle_type.state_size()})

    def get_patch_extras(self, particle_type: PatchyBaseParticleType, patch_idx: int) -> dict:
        # adjust for patch multiplier from multiparticale_patchesdentate
        state_var = particle_type.patch(patch_idx).state_var()
        activation_var = particle_type.patch(patch_idx).activation_var()
        return {
            "state_var": state_var,
            "activation_var": activation_var
        }


class LWriter(BasePatchyWriter):
    def get_input_file_data(self, scene: Scene, **kwargs) -> list[tuple[str, str]]:
        return [("DPS_interaction_matrix_file", "interactions.txt")]

    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY, "topology"]

    def write(self,
              scene: Scene,
              stage: Union[Stage, None] = None,
              **kwargs
              ) -> dict[str, str]:
        particles = scene.particle_types()
        particle_type_counts = scene.particle_type_counts()

        particles, patches = to_PL(particles,
                                   kwargs[NUM_TEETH_KEY],
                                   kwargs[DENTAL_RADIUS_KEY])
        init_top = kwargs["topology"]
        init_conf = kwargs["init_conf"]
        particles_txts_files = []
        with self.file(init_top) as f:
            for particle in particles:
                # add particle file name to files list
                particles_txts_files.append(self.directory() / f"patches_{particle.type_id()}.dat")
                f.write(particle.export_to_lorenzian_patchy_str(
                    particle_type_counts[particle.type_id()],
                    self.directory()) + "\n")

        interactions_file = self.directory() / "interactions.txt"
        export_interaction_matrix(patches, interactions_file)

        self.write_conf(scene, self.directory() / init_conf)

        return {
            "conf_file": init_conf,
            "topology": init_top,
            # *particles_txts_files,
            "DPS_interaction_matrix_file": interactions_file
        }


__writers = {
    "flavio": FWriter(),
    "josh_flavio": JFWriter(),
    # "josh_lorenzo": JLWriter(),
    # "lorenzo": LWriter()
}


def get_writer(writer_key: Union[str, None] = None) -> BasePatchyWriter:
    if writer_key is None:
        writer_key = get_server_config()[PATCHY_FILE_FORMAT_KEY]
    return __writers[writer_key]


def register_writer(writer_name: str, writer_obj: BasePatchyWriter):
    __writers[writer_name] = writer_obj


NUM_TEETH_KEY = "num_teeth"
DENTAL_RADIUS_KEY = "dental_radius"
