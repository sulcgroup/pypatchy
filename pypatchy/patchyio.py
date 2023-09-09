from __future__ import annotations

from typing import Union, IO
from abc import ABC, abstractmethod
from pathlib import Path

from pypatchy.patchy.patchy_scripts import to_PL
from pypatchy.patchy.plpatchy import PLPatchyParticle, export_interaction_matrix
from pypatchy.patchy_base_particle import PatchyBaseParticleType
from build.lib.pypatchy.patchy_base_particle import BaseParticleSet
from pypatchy.polycubeutil.polycubesRule import PolycubeRuleCubeType
from pypatchy.util import get_server_config, PATCHY_FILE_FORMAT_KEY


class BasePatchyWriter(ABC):
    """
    
    """

    _writing_directory: Path

    def __init__(self):
        pass

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
              ) -> tuple[list[Path], dict[str, Union[str, float, int]]]:
        """
        Returns:
            a tuple the first element of which is a list of the paths of files this method just created,
            the second of which is a dict to append to oxdna input file
        """
        pass


class FWriter(BasePatchyWriter):
    def write(self,
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs) -> tuple[list[Path], dict[str, Union[str, float, int]]]:
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

                    patches_file.write(patch_obj.save_to_string())

        return (
            [
                self.directory() / "init.top",
                self.directory() / "particles.txt",
                self.directory() / "patches.txt"
            ],
            {
                "patchy_file": "patches.txt",
                "particle_file": "particles.txt",
                "particle_types_N": len(particles),
                "patch_types_N": particles.num_patches()
            }
        )

    def reqd_args(self) -> list[str]:
        return []


class JWriter(BasePatchyWriter, ABC):
    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY]

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
              ) -> tuple[list[Path], dict[str, Union[str, float, int]]]:
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
                    patches_file.write(patch_obj.save_to_string(extradict))

                particles_file.write(self.get_particle_extras(particle_patchy, particle_type))
        return (
            [
                self.directory() / "init.top",
                self.directory() / "particles.txt",
                self.directory() / "patches.txt"
            ],
            {
                "patchy_file": "patches.txt",
                "particle_file": "particles.txt",
                "particle_types_N": len(particles),
                "patch_types_N": particles.num_patches()
            }
        )


class JFWriter(JWriter):
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
    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY]

    def write(self,
              particles: BaseParticleSet,
              particle_type_counts: dict[int, int],
              **kwargs
              ) -> tuple[list[Path], dict[str, Union[str, float, int]]]:
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

        return (
            [
                self.directory() / "init.top",
                *particles_txts_files,
                interactions_file
            ],
            {
                "DPS_interaction_matrix_file": "interactions.txt"
            }
        )


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
