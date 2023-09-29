from __future__ import annotations

from typing import Union, IO
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from pypatchy.patchy.patchy_scripts import to_PL
from pypatchy.patchy.plpatchy import PLPatchyParticle, export_interaction_matrix, PLPatch
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
                            particles: BaseParticleSet,
                            particle_type_counts: dict[int, int],
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
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs
              ) -> list[Path]:
        """
        Returns:
            a tuple the first element of which is a list of the paths of files this method just created,
            the second of which is a dict to append to oxdna input file
        """
        pass
    @abstractmethod
    def read_top(self, fp: Path, scene: Scene):
        pass

    @abstractmethod
    def write_top(self, fp: Path, scene: Scene):
        pass


class FWriter(BasePatchyWriter):
    """
    Class for writing files in the Flavian style, which uses particles.txt, patches.txt, and the Coliseum
    """

    def get_input_file_data(self, particles: BaseParticleSet, particle_type_counts: dict[int, int], **kwargs) -> list[
        tuple[str, str]]:

        return [
            ("patchy_file", "patches.txt"),
            ("particle_file", "particles.txt"),
            ("particle_types_N", f"{len(particles)}"),
            ("patch_types_N", f"{particles.num_patches()}")
        ]

    def write(self,
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs) -> list[Path]:
        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)
        plparticles, patches = to_PL(particles, 1, 0)

        total_num_particles = sum(particle_type_counts.values())
        with self.file("init.top") as top_file, self.file("particles.txt") as particles_file, self.file(
                "patches.txt") as patches_file:
            # first line of file
            top_file.write(f"{total_num_particles} {len(particles)}\n")
            # second line of file
            top_file.write(" ".join([
                f"{particle.type_id()} " * particle_type_counts[particle.type_id()] for particle in
                particles.particles()
            ]))

            for particle_patchy, particle_type in zip(plparticles, particles.particles()):
                # handle writing particles file
                for patch_idx, patch_obj in enumerate(particle_patchy.patches()):
                    # we have to be VERY careful here with indexing to account for multidentate simulations
                    # adjust for patch multiplier from multidentate

                    patches_file.write(self.save_patch_to_str(patch_obj))

        return [
                self.directory() / "init.top",
                self.directory() / "particles.txt",
                self.directory() / "patches.txt"
            ]


    def reqd_args(self) -> list[str]:
        return []

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

    def read_top(self, fp: Path, scene: Scene):
        """
        Reads a Flavian-style topology file
        """
        with fp.open("r") as handle:
            lines = handle.readlines()
            vals = lines[0].split()
            N = int(vals[0])
            type_counts = []
            # for c in range(len(patches.keys())):
            particles = []  # [None for x in range(N)]

            for pid, line in enumerate(lines[1:]):
                if len(line.split()) >= 2:
                    vals = line.split()
                    count = int(vals[0])
                    type_counts.append(count)
                    pcount = int(vals[1])
                    ptypes = vals[2]
                    ptypes = [int(x) for x in ptypes.strip().split(',')]
                    pgeometry = vals[3]
                    patches = []
                    index = 0
                    for i, p in enumerate(ptypes):
                        patch = PLPatch()
                        patch.set_color(p)
                        patch.init_from_dps_file(pgeometry, i)
                        patches.append(patch)
                    for j in range(count):
                        particle = PLPatchyParticle()
                        particle._type = pid
                        particle._patch_ids = ptypes
                        particle._patches = patches
                        particle.unique_id = index
                        index += 1
                        particles.append(particle)

            scene.add_particles(particles)

    def write_top(self, fp: Path, s: Scene):
        with fp.open("w") as handle:
            # write num particles, num types
            handle.write(f"{s.num_particles()} {s.num_particle_types()}")
            # write types
            handle.write(" ".join([str(p.get_type()) for p in s.particles()]))
            # write trailing newline
            handle.write("\n")

class JWriter(BasePatchyWriter, ABC):
    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY]

    def get_input_file_data(self,
                            particles: BaseParticleSet,
                            particle_type_counts: dict[int, int],
                            **kwargs) -> list[tuple[str, str]]:
        return [
                ("patchy_file", "patches.txt"),
                ("particle_file", "particles.txt"),
                ("particle_types_N", str(len(particles))),
                ("patch_types_N", str(particles.num_patches() * kwargs[NUM_TEETH_KEY]))
            ]

    @abstractmethod
    def get_patch_extras(self, particle_type: PatchyBaseParticleType, patch_idx: int) -> dict:
        pass

    @abstractmethod
    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        pass

    def write(self,
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs
              ) -> list[Path]:
        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchy.py)
        plparticles, patches = to_PL(particles,
                                     kwargs[NUM_TEETH_KEY],
                                     kwargs[DENTAL_RADIUS_KEY])

        total_num_particles = sum(particle_type_counts.values())
        with self.file("init.top") as top_file, self.file("particles.txt") as particles_file, self.file(
                "patches.txt") as patches_file:
            # first line of file
            top_file.write(f"{total_num_particles} {len(particles)}\n")
            # second line of file
            top_file.write(" ".join([
                f"{particle.type_id()} " * particle_type_counts[particle.type_id()] for particle in
                particles.particles()
            ]))

            for particle_patchy, particle_type in zip(plparticles, particles.particles()):
                # handle writing particles file
                for i, patch_obj in enumerate(particle_patchy.patches()):
                    # we have to be VERY careful here with indexing to account for multidentate simulations
                    # adjust for patch multiplier from multidentate
                    patch_idx = int(i / kwargs[NUM_TEETH_KEY])
                    extradict = self.get_patch_extras(particle_type, patch_idx)
                    patches_file.write(self.save_patch_to_str(patch_obj, extradict))

                particles_file.write(self.get_particle_extras(particle_patchy, particle_type))
        return [
                self.directory() / "init.top",
                self.directory() / "particles.txt",
                self.directory() / "patches.txt"
            ]



# inherit from FWriter so can use class methods
class JFWriter(JWriter, FWriter):

    def get_particle_extras(self, plpartcle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        return plpartcle.save_type_to_string()

    def get_patch_extras(self, particle_type: PolycubeRuleCubeType, patch_idx: int) -> dict:
        allo_conditional = particle_type.patch_conditional(
            particle_type.get_patch_by_idx(patch_idx), minimize=True)
        # allosteric conditional should be "true" for non-allosterically-controlled patches
        return {"allostery_conditional": allo_conditional if allo_conditional else "true"}


class JLWriter(JWriter):
    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PatchyBaseParticleType) -> str:
        return plparticle.save_type_to_string({"state_size": particle_type.state_size()})

    def get_patch_extras(self, particle_type: PatchyBaseParticleType, patch_idx: int) -> dict:
        # adjust for patch multiplier from multiparticale_patchesdentate
        state_var = particle_type.patch(patch_idx).state_var()
        activation_var = particle_type.patch(patch_idx).activation_var()
        return {
            "state_var": state_var,
            "activation_var": activation_var
        }


class LWriter(BasePatchyWriter):
    def get_input_file_data(self, particles: BaseParticleSet, particle_type_counts: dict[int, int], **kwargs) -> list[
        tuple[str, str]]:
        return [("DPS_interaction_matrix_file", "interactions.txt")]

    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY]

    def write(self,
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs
              ) -> list[Path]:
        particles, patches = to_PL(particles,
                                   kwargs[NUM_TEETH_KEY],
                                   kwargs[DENTAL_RADIUS_KEY])
        particles_txts_files = []
        with self.file("init.top") as f:
            for particle in particles:
                # add particle file name to files list
                particles_txts_files.append(self.directory() / f"patches_{particle.type_id()}.dat")
                f.write(particle.export_to_lorenzian_patchy_str(
                    particle_type_counts[particle.type_id()],
                    self.directory()) + "\n")

        interactions_file = self.directory() / "interactions.txt"
        export_interaction_matrix(patches, interactions_file)

        return [
                self.directory() / "init.top",
                *particles_txts_files,
                interactions_file
            ]



__writers = {
    "josh_flavio": JFWriter(),
    "josh_lorenzo": JLWriter(),
    "lorenzo": LWriter()
}


def get_writer(writer_key: Union[str, None] = None) -> BasePatchyWriter:
    if writer_key is None:
        writer_key = get_server_config()[PATCHY_FILE_FORMAT_KEY]
    return __writers[writer_key]


def register_writer(writer_name: str, writer_obj: BasePatchyWriter):
    __writers[writer_name] = writer_obj


NUM_TEETH_KEY = "num_teeth"
DENTAL_RADIUS_KEY = "dental_radius"
