from __future__ import annotations

import copy
import itertools
import re
from typing import Union, IO, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import oxDNA_analysis_tools.UTILS.RyeReader as rr

from .plpatch import PLPatch
from .plparticle import PLPatchyParticle
from .plparticleset import PLParticleSet
from .plscene import PLPSimulation
from ...polycubeutil.polycubesRule import PolycubeRuleCubeType
from ...util import normalize, DENTAL_RADIUS_KEY, NUM_TEETH_KEY
from ...server_config import PatchyServerConfig, get_server_config


# this approach was suggested by chatGPT.
# I kind of hate it but if it works it works
def custom_formatter(x):
    if abs(x) < 1e-9:
        return "0"
    if x.is_integer():
        return str(int(x))
    else:
        return str(x)

# TODO: new PatchyBaseWritier class which we can extend to PLBaseWriter, MGLWriter, etc.

class PLBaseWriter(ABC):
    """
    Abstract base class from which all other writer class inherits
    """

    _writing_directory: Path
    _is_write_abs_paths = False

    def __init__(self):
        self._writing_directory = None

    def set_abs_paths(self, b: Union[bool, PatchyServerConfig]):
        if isinstance(b, bool):
            self._is_write_abs_paths = b
        else:
            self._is_write_abs_paths = b.absolute_paths

    def is_abs_paths(self):
        return self._is_write_abs_paths

    @abstractmethod
    def get_input_file_data(self,
                            scene: PLPSimulation,
                            **kwargs) -> list[tuple[str, str]]:
        """
        Returns a set of tuples of str,str which should be added
        to oxDNA input files
        """

    def set_directory(self, directory: Union[Path, str]):
        if isinstance(directory, Path):
            self._writing_directory = directory.expanduser()
        else:
            self.set_directory(Path(directory))

    def directory(self) -> Path:
        return self._writing_directory

    def file(self, filename, access="w") -> IO:
        assert access != "r" or (self.directory() / filename).exists()
        return open(self.directory() / filename, access)

    @abstractmethod
    def reqd_args(self) -> list[str]:
        """
        Returns a list of arguements that need to be passed to `write`
        """
        pass

    @abstractmethod
    def write(self,
              scene: PLPSimulation,
              **kwargs
              ) -> dict[str, str]:
        """
        Returns:
            a tuple the first element of which is a list of the paths of files this method just created,
            the second of which is a dict to append to oxdna input file
        """
        pass

    @abstractmethod
    def write_top(self, topology: PatchyTopology, top_path: Union[str, Path]):
        pass

    @abstractmethod
    def read_top(self, top_file: str) -> SWriter.SPatchyTopology:
        pass

    def write_conf(self, scene: PLPSimulation, p: Path):
        assert scene.get_conf() is not None
        conf = scene.get_conf()
        rr.write_conf(str(self.directory() / p), conf)

    @abstractmethod
    def read_particle_types(self, *args):
        """
        Reads particle type data without scene data or (ideally) .top file data
        The arguements taken by this method should vary by writer but should
        always be a subset of the return value of reqd_args()
        """
        pass

    @abstractmethod
    def read_scene(self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx=None) -> PLPSimulation:
        pass

    # @abstractmethod
    # def read(self) -> Union[Scene, None]:
    #     pass

    @abstractmethod
    def get_scene_top(self, s: PLPSimulation) -> PatchyTopology:
        pass

    class PatchyTopology(ABC):
        particle_ids: list[int]  # list of particles where each value is the particle's type ID

        @abstractmethod
        def particle_type_count(self, p) -> int:
            pass

        def get_particles_types(self) -> list[int]:
            return self.particle_ids

        @abstractmethod
        def num_particle_types(self) -> int:
            pass

        @abstractmethod
        def num_particles(self) -> int:
            pass


class FWriter(PLBaseWriter):
    """
    Class for writing files in the Flavian style, which uses particles.txt, patches.txt, and the Coliseum
    """

    def read_top(self, top_file: str) -> FWriter.PatchyTopology:
        with self.file(top_file, "r") as f:
            _, particles = f.readlines()
            particle_ids = [int(p) for p in particles.split()]
        return FWriter.FPatchyTopology(particle_ids)

    def get_scene_top(self, s: PLPSimulation) -> FPatchyTopology:
        return FWriter.FPatchyTopology(s.particles())

    def read_particle_types(self, patchy_file: str, particle_file: str) -> PLParticleSet:
        patches = self.load_patches(patchy_file)
        particles: list[PLPatchyParticle] = []

        with self.file(particle_file, "r") as f:
            lines = f.readlines()
            j = 0
            for line in lines:
                line = line.strip()
                # if line is not blank or a comment
                if len(line) > 1 and line[0] != '#':
                    if 'particle_' and '{' in line:
                        strargs = []
                        k = j + 1
                        while '}' not in lines[k]:
                            strargs.append(lines[k].strip())
                            k = k + 1
                        particle = self.particle_from_lines(strargs, patches)
                        particles.append(particle)
                j = j + 1
        particles.sort(key=lambda p: p.get_type())
        particle_set = PLParticleSet(particles)
        if not all([np.abs(p.rotmatrix() - np.identity(3)).sum() < 1e-6 for p in particles]):
            pass
            # particle_set.normalize()

        return particle_set

    def particle_from_lines(self,
                            lines: Iterable[str],
                            patches: list[PLPatch]) -> PLPatchyParticle:
        type_id = None
        patch_ids = None
        for line in lines:
            line = line.strip()
            if "=" in line and line[0] != '#':
                key, vals = line.split("=")
                key = key.strip()
                if key == "type":
                    vals = int(line.split('=')[1])
                    type_id = vals
                elif key == "patches":
                    vals = line.split('=')[1]
                    patch_ids = [int(g) for g in vals.split(',')]
        assert type_id is not None, "Missing type_id for particle!"  # not really any way to ID this one lmao
        assert patch_ids is not None, f"Missing patches for particle type {type_id}"
        return PLPatchyParticle([patches[i] for i in patch_ids], type_id=type_id)

    def get_input_file_data(self,
                            scene: PLPSimulation,
                            **kwargs) -> list[tuple[str, str]]:

        return [
            ("particle_types_N", f"{scene.num_particle_types()}"),
            ("patch_types_N", f"{scene.particle_types().num_patches()}")
        ]

    def write_top(self, topology: FWriter.PatchyTopology, top_path: str):

        with self.file(top_path) as top_file:
            # write top file
            # first line of file
            top_file.write(f"{topology.num_particles()} {topology.num_particle_types()}\n")
            # second line of file
            top_file.write(" ".join([str(particle) for particle in topology.particles()]))

    def particle_type_string(self, particle: PLPatchyParticle, extras: dict[str, str] = {}) -> str:
        outs = 'particle_%d = { \n type = %d \n ' % (particle.type_id(), particle.type_id())
        outs = outs + 'patches = '
        for i, p in enumerate(particle.patches()):
            outs = outs + str(p.type_id())
            if i < len(particle.patches()) - 1:
                outs = outs + ','
        outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
        outs = outs + ' \n } \n'
        return outs

    def write_particles_patches(self, particles: PLParticleSet, particle_fn: str, patchy_fn: str):
        with self.file(particle_fn) as particles_file, \
                self.file(patchy_fn) as patches_file:

            # swrite particles and patches file
            for particle_patchy in particles.particles():
                # handle writing particles file
                for patch_idx, patch_obj in enumerate(particle_patchy.patches()):
                    # we have to be VERY careful here with indexing to account for multidentate simulations
                    # adjust for patch multiplier from multidentate

                    patches_file.write(self.save_patch_to_str(patch_obj))
                particles_file.write(self.particle_type_string(particle_patchy))

    def write(self,
              scene: PLPSimulation,
              **kwargs) -> dict[str, str]:

        particle_fn = kwargs["particle_file"]
        patchy_fn = kwargs["patchy_file"]
        init_top = kwargs["topology"]
        init_conf = kwargs["conf_file"]
        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchylib.py)

        particles = scene.particle_types()
        particle_type_counts = scene.particle_type_counts()

        total_num_particles = sum(particle_type_counts.values())
        self.write_particles_patches(scene.particle_types(), particle_fn, patchy_fn)

        self.write_top(self.get_scene_top(scene), init_top)

        # write conf
        self.write_conf(scene, init_conf)

        if self.is_abs_paths():
            particle_fn = self.directory() / particle_fn
            patchy_fn = self.directory() / patchy_fn
            init_top = self.directory() / init_top
            init_conf = self.directory() / init_conf

        return {
            "particle_file": str(particle_fn),
            "patchy_file": str(patchy_fn),
            "topology": str(init_top),
            "init_conf": str(init_conf)
        }

    def reqd_args(self) -> list[str]:
        return ["patchy_file", "particle_file", "conf_file", "topology"]  # todo: topology and conf should be builtin

    def save_patch_to_str(self, patch: PLPatch, extras: dict = {}) -> str:
        # print self._type,self._type,self._color,1.0,self._position,self._a1,self._a2

        fmtargs = {
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

    def load_patches(self, filename: str, num_patches=0) -> list[PLPatch]:
        j = 0
        Np = 0
        patches = [PLPatch() for _ in range(num_patches)]
        fpath = self.directory() / filename
        assert fpath, f"No file called {filename}"

        with fpath.open("r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 1 and line[0] != '#':
                    if 'patch_' and '{' in line:
                        strargs = dict()
                        k = j + 1
                        while '}' not in lines[k]:
                            if not lines[k].isspace() and not line.startswith("#"):
                                key, val = lines[k].split("=")
                                key = key.strip()
                                strargs[key] = val
                            k = k + 1
                        patch = PLPatch()
                        # print 'Loaded patch',strargs
                        patch.init_from_string(strargs)
                        index = patch.type_id()
                        # flexable patch indexing
                        # probably not optimized speed wise but optimized for flexibility
                        if index >= len(patches):
                            patches += [None for _ in range(index - len(patches) + 1)]
                        patches[index] = patch
                        Np += 1
                j = j + 1

        if num_patches != 0 and Np != num_patches:
            raise IOError('Loaded %d patches, as opposed to the desired %d types ' % (Np, num_patches))
        return patches

    def read_scene(self, top_file: Path, traj_file: Path, particle_types: PLParticleSet, conf_idx=None) -> PLPSimulation:
        """
        Reads a patchy particle scene from files
        """
        top_file = self.directory() / top_file
        traj_file = self.directory() / traj_file
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

            scene.compute_cell_size(n_particles=len(ptypelist))
            scene.apportion_cells()
            for i, ptype_idx in enumerate(ptypelist):
                ptype: PLPatchyParticle = particle_types.particle(ptype_idx)
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

    # def read(self) -> Union[None, PLPSimulation]:
    #     assert (self.directory() / "input").exists(), "Can't read without input file"  # TODO: custom exception
    #     input_file = Input(str(self.directory()))
    #     # can't assume local setup has all values in ensemble json
    #     # handle absolute vs relative file paths
    #     patchy_file_path = input_file.input["patchy_file"]
    #     if Path(patchy_file_path).is_absolute():
    #         patchy_file_path = Path(patchy_file_path).suffix
    #     assert (self.directory() / patchy_file_path).exists(), "Missing patchy file!"
    #     particle_file_path = input_file.input["particle_file"]
    #     if Path(particle_file_path).is_absolute():
    #         particle_file_path = Path(particle_file_path).suffix
    #     assert (self.directory() / particle_file_path).exists(), "Missing particle file!"
    #     ptypes = self.read_particle_types(patchy_file_path,
    #                                       particle_file_path)
    #     # top and conf
    #     top_file = input_file.input["topology"]
    #     if Path(top_file).is_absolute():
    #         top_file = Path(top_file).suffix
    #     conf_file = input_file.input["conf_file"]
    #     if Path(conf_file).is_absolute():
    #         conf_file = Path(conf_file).suffix
    #     if s

    class FPatchyTopology(PLBaseWriter.PatchyTopology):

        nParticleTypes: int  # number of particle types
        topology_particles: list[int]  # list of particles, where each value is a type ID
        type_counts: dict[int, int]  # particle type ID -> count

        def __init__(self, top_particles: list[Union[PLPatchyParticle, int]]):
            self.topology_particles = []
            for p in top_particles:
                if isinstance(p, int):
                    self.topology_particles.append(p)
                elif isinstance(p, PLPatchyParticle):
                    self.topology_particles.append(p.type_id())
                else:
                    raise Exception()
            self.nParticleTypes = max(self.topology_particles) + 1
            self.type_counts = {i: 0 for i in range(self.nParticleTypes)}
            for p in self.topology_particles:
                self.type_counts[p] += 1

        def num_particle_types(self) -> int:
            return self.nParticleTypes

        def particle_type_counts(self) -> dict[int, int]:
            return self.type_counts

        def particles(self) -> Iterable[int]:
            return self.topology_particles

        def num_particles(self) -> int:
            return len(self.topology_particles)

        def particle_type_count(self, p) -> int:
            return self.type_counts[p]


class JWriter(PLBaseWriter, ABC):
    """
    Writer class for Josh's file formats (aka ones with allostery)
    This is a modifiecation of Lorenzo's or Flavio/Petr's formats, so this
    class is abstract
    """

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
                            scene: PLPSimulation,
                            **kwargs) -> list[tuple[str, str]]:
        return [
            ("particle_types_N", str(scene.num_particle_types())),
            ("patch_types_N", str(scene.particle_types().num_patches()))
        ]

    @abstractmethod
    def get_patch_extras(self, particle_type: PLPatchyParticle, patch_idx: int) -> dict:
        pass

    @abstractmethod
    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PLPatchyParticle) -> str:
        pass

    def write(self,
              scene: PLPSimulation,
              **kwargs
              ) -> dict[str, str]:

        # file info
        particle_fn = kwargs["particle_file"]
        patchy_fn = kwargs["patchy_file"]
        init_top = kwargs["topology"]
        init_conf = kwargs["conf_file"]
        particles_type_list: PLParticleSet = kwargs["particle_types"]

        # write top and particles/patches spec files
        # first convert particle json into PLPatchy objects (cf plpatchylib.py)

        pl_set = scene.particle_types()
        # kwargs[NUM_TEETH_KEY],
        # kwargs[DENTAL_RADIUS_KEY])

        self.write_conf(scene, init_conf)

        with self.file(particle_fn) as particles_file, \
                self.file(patchy_fn) as patches_file:

            # todo: allosteric hell world
            # write particles and patches file
            for particle_patchy, particle_type in zip(pl_set, particles_type_list.particles()):
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

        self.write_top(self.get_scene_top(scene), init_top)

        self.write_conf(scene, init_conf)

        if self.is_abs_paths():
            init_top = self.directory() / init_top
            particle_fn = self.directory() / particle_fn
            patchy_fn = self.directory() / patchy_fn
            init_conf = self.directory() / init_conf

        return {
            "topology": init_top,
            "particle_file": particle_fn,
            "patchy_file": patchy_fn,
            "conf_file": init_conf
        }

    @abstractmethod
    def save_patch_to_str(self, patch_obj: PLPatch, extradict: dict) -> str:
        pass


# inherit from FWriter so can use class methods
class JFWriter(JWriter, FWriter):
    """
    Flavio-style writer to export patches (implicit non-dynamic formulation of allostery)
    """

    def save_patch_to_str(self, patch_obj: PLPatch, extradict: dict) -> str:
        return FWriter.save_patch_to_str(self, patch_obj, extradict)

    def get_particle_extras(self, plpartcle: PLPatchyParticle, particle_type: PLPatchyParticle) -> str:
        return self.particle_type_string(plpartcle)

    def get_patch_extras(self, particle_type: PolycubeRuleCubeType, patch_idx: int) -> dict:
        allo_conditional = particle_type.patch_conditional(
            particle_type.get_patch_by_idx(patch_idx), minimize=True)
        # allosteric conditional should be "true" for non-allosterically-controlled patches
        return {"allostery_conditional": allo_conditional if allo_conditional else "true"}


class JLWriter(JWriter):
    """
    Lorenzo-style-ish patchy format but josh has messed with it
    this format uses particle type strings like Flavio's but with the dynamic model. kinda.

    Please don't use this format right now!!!
    """

    def read_particle_types(self, *args):
        pass

    def write_top(self, topology: LWriter.PatchyTopology, top_path: Union[str, Path]):
        pass

    def read_scene(self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx=None) -> PLPSimulation:
        pass

    def particle_type_string(self, particle: PLPatchyParticle, extras: dict[str, str] = {}) -> str:
        outs = 'particle_%d = { \n type = %d \n ' % (particle.type_id(), particle.type_id())
        outs = outs + 'patches = '
        for i, p in enumerate(particle.patches()):
            outs = outs + str(p.type_id())
            if i < len(particle.patches()) - 1:
                outs = outs + ','
        outs += "\n".join([f"{key} = {extras[key]}" for key in extras])
        outs = outs + ' \n } \n'
        return outs

    def get_particle_extras(self, plparticle: PLPatchyParticle, particle_type: PLPatchyParticle) -> str:
        return self.particle_type_string(plparticle, {"state_size": particle_type.state_size()})

    def get_patch_extras(self, particle_type: PLPatchyParticle, patch_idx: int) -> dict:
        # adjust for patch multiplier from multiparticale_patchesdentate
        state_var = particle_type.patch(patch_idx).state_var()
        activation_var = particle_type.patch(patch_idx).activation_var()
        return {
            "state_var": state_var,
            "activation_var": activation_var
        }


class LWriter(PLBaseWriter):
    """
    Class for writing data in Lorenzo's patch particle format
    """

    def read_top(self, top_file: str) -> LWriter.PatchyTopology:
        """
        WARNING: THIS METHOD RETURNS A PATCHYTOPOLOGY OBJECT WITHOUT PATCH COLORS
        """
        particle_types = []
        particle_type_counts = {}
        # load topology file, which contains particle type info and type counts
        with self.file(top_file, "r") as f:
            f.readline()
            for pid, line in enumerate(f):
                nInstances, nPatches, patchIDs, patchesfn = line.split()
                nPatches = int(nPatches)
                nInstances = int(nInstances)
                # patch color isn't included here, since bindings are defined by the interaction matrix
                patchIDs = [int(p) for p in patchIDs.split(",")]
                assert len(patchIDs) == nPatches, "Number of patches specified doesn't match length of patch list!"
                patches = []
                # load file with info about patches
                with self.file(patchesfn, "r") as patches_file:
                    # iter lines in patch files
                    for (patch_id, patch_line) in zip(patchIDs, patches_file):
                        # unfortunately patch_idx doesn't correspond to patch_id
                        patch_coords = np.array([float(i) for i in patch_line.split()])
                        patch_type = PLPatch(type_id=patch_id, relposition=patch_coords, a1=normalize(patch_coords))
                        patches.append(patch_type)
                particle_type = PLPatchyParticle(patches, type_id=pid, index_=pid)
                particle_types.append(particle_type)
                particle_type_counts[pid] = nInstances
        return LWriter.LPatchyTopology(PLParticleSet(particle_types), particle_type_counts)

    def get_scene_top(self, s: PLPSimulation) -> LPatchyTopology:
        return LWriter.LPatchyTopology(s.particle_types(), s.particle_type_counts())

    def read_particle_types(self, topology: str, DPS_interaction_matrix_file: str) -> PLParticleSet:
        top = self.read_top(topology)
        interaction_matrix = self.read_interaction_matrix(DPS_interaction_matrix_file)
        self.assign_colors(interaction_matrix, top.particle_types.patches())
        return top.particle_types

    def read_interaction_matrix(self, interaction_file: str) -> np.ndarray:
        pattern = r"patchy_eps\[(\d+)\]\[(\d+)\] = (\d+\.?\d*)"
        data = []
        max_index = 0
        with self.file(interaction_file, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                assert match, f"Malformed interaction file line {line}"
                extracted = tuple(map(int, match.groups()[:-1])), float(match.group(3))
                indices, value = extracted
                max_index = max(max_index, *indices)
                if indices[0] == indices[1]:
                    # TOOD: support
                    raise ValueError(f"Patch in line {line} interacts with itself, not currently supported!")
                data.append((indices, value))

        # Create an array with dimensions based on the maximum index
        array = np.zeros((max_index + 1, max_index + 1))  # +1 because indices are zero-based
        for (i, j), value in data:
            array[i, j] = array[j, i] = value

        return array

    def assign_colors(self, interaction_matrix: np.ndarray, patches: list[PLPatch]):
        num_patches = interaction_matrix.shape[0]
        assert num_patches == len(patches), f"Size of patch interaction matrix {num_patches} is not compatible " \
                                            f"with length of patches array {len(patches)}!"

        # colors = np.zeros(num_patches, dtype=int)  # Color 0 means uncolored

        # depth-first search which assigns colors
        def dfs(node: int, color: int):
            if patches[node].color() != 0 and patches[node].color() is not None:
                return patches[node].color() == color
            patches[node].set_color(color)
            for neighbor in range(num_patches):
                if interaction_matrix[node, neighbor] > 0:
                    if not dfs(neighbor, -color):
                        return False
            return True

        for patch in range(num_patches):
            if (patches[patch].color() == 0 or patches[patch].color() is None) and not dfs(patch, 1):
                raise ValueError("Cannot assign colors such that interacting patches add to zero")

    def write_top(self, topology: LPatchyTopology, top_file: str):
        with self.file(top_file) as f:
            f.write(f"{len(topology.particles())} {topology.particle_types.num_particle_types()}\n")
            for ptype in topology.particle_types:
                # add particle file name to files list
                # particles_txts_files.append(self.directory() / f"patches_{particle.type_id()}.dat")
                f.write(self.particle_type_str(ptype,
                                               topology.particle_type_count(ptype.type_id())) + "\n")

    def read_scene(self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx: Union[None, int] = None) -> PLPSimulation:
        top: LWriter.PatchyTopology = self.read_top(top_file)
        scene = PLPSimulation()
        scene.set_particle_types(particle_types)
        top_info, traj_info = rr.describe(str(self.directory() / top_file),
                                          str(self.directory() / traj_file))
        if conf_idx is None:
            conf = rr.get_confs(top_info, traj_info, traj_info.nconfs - 1, 1)[0]
        else:
            conf = rr.get_confs(top_info, traj_info, conf_idx, 1)[0]
        conf = rr.inbox(conf, center=False)
        assert ((conf.positions < conf.box) & (conf.positions >= 0)).all(), "Conf inbox did not inbox!"
        scene = PLPSimulation()
        scene.set_time(conf.time)
        scene.set_particle_types(particle_types)
        scene.set_box_size(conf.box)
        scene.compute_cell_size(n_particles=top.num_particles())
        scene.apportion_cells()
        for i, ptype_idx in enumerate(top.particle_ids):
            ptype: PLPatchyParticle = particle_types.particle(ptype_idx)
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

    def get_input_file_data(self, scene: PLPSimulation, **kwargs) -> list[tuple[str, str]]:
        return [("DPS_interaction_matrix_file", "interactions.txt")]

    def reqd_args(self) -> list[str]:
        return [NUM_TEETH_KEY, DENTAL_RADIUS_KEY, "topology", "conf_file", "DPS_interaction_matrix_file"]

    def write(self,
              scene: PLPSimulation,
              **kwargs
              ) -> dict[str, str]:
        assert self.directory() is not None, "No writing directory specified!"
        assert self.directory().exists(), f"Specified writing directory {str(self.directory())} does not exist!"

        particles: PLParticleSet = scene.particle_types()
        scene.sort_particles_by_type()

        init_top = kwargs["topology"]
        init_conf = kwargs["conf_file"]

        interactions_file = kwargs["DPS_interaction_matrix_file"]
        self.export_interaction_matrix(particles.patches(), interactions_file)

        self.write_conf(scene, init_conf)
        self.write_top(self.get_scene_top(scene), init_top)

        if self.is_abs_paths():
            init_conf = self.directory() / init_conf
            init_top = self.directory() / init_top
            interactions_file = self.directory() / interactions_file

        return {
            "conf_file": str(init_conf),
            "topology": str(init_top),
            "DPS_interaction_matrix_file": str(interactions_file)
        }

    def export_interaction_matrix(self, patches: list[PLPatch], filename: str):
        with self.file(filename, "w") as f:
            f.writelines(
                [
                    f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = {p1.strength()}\n"
                    for p1, p2 in itertools.combinations(patches, 2)
                    if p1.color() == -p2.color()
                ]
            )

    class LPatchyTopology(PLBaseWriter.PatchyTopology):
        """
        Lorenzian topology includes particle type info
        """
        particle_types: PLParticleSet
        type_counts: dict[int, int]

        def __init__(self, particle_types: PLParticleSet, particles: Union[list[int], dict[int, int]]):
            """
            Constructs a lorenzian-type patcy particle topology info object
            """
            self.particle_types = particle_types
            if isinstance(particles, list):
                self.particle_ids = particles
                self.type_counts = {}
                for p in particles:
                    if isinstance(p, PLPatchyParticle):
                        if p.type_id() not in self.type_counts:
                            self.type_counts[p.type_id()] = 0
                        self.type_counts[p.type_id()] += 1
                    else:
                        if p not in self.type_counts:
                            self.type_counts[p] = 0
                        self.type_counts[p] += 1
            else:
                assert isinstance(particles, dict), "Invalid type"
                self.type_counts = particles
                self.particle_ids = list(itertools.chain.from_iterable([
                    [ptype for _ in range(pcount)] for ptype, pcount in particles.items()
                ]))

        def particle_type_count(self, particle_id: int) -> int:
            return self.type_counts[particle_id] if particle_id in self.type_counts else 0

        def particles(self) -> list[int]:
            return self.particle_ids

        def num_particles(self) -> int:
            return len(self.particle_ids)

        def num_particle_types(self) -> int:
            return self.particle_types.num_particle_types()

    def particle_type_str(self, particle: PLPatchyParticle, nInstances: int) -> str:
        if self.is_abs_paths():
            patches_dat_filename = self.directory() / f"patches_{particle.type_id()}.dat"
        else:
            patches_dat_filename = f"patches_{particle.type_id()}.dat"
        particle_str = f"{nInstances} {particle.num_patches()} {','.join([str(pid) for pid in particle.patch_ids()])} {patches_dat_filename}"
        patches_dat_filestr = "\n".join(
            [np.array2string(patch.position(),
                             separator=" ",
                             suppress_small=True,
                             formatter={'float_kind': custom_formatter}
                             )[1:-1]
             for patch in particle.patches()]
        )
        with self.file(patches_dat_filename, "w") as f:
            f.write(patches_dat_filestr)
        return particle_str


class SWriter(PLBaseWriter):
    """
    Subhian patchy file writer
    """

    def get_input_file_data(self, scene: PLPSimulation, **kwargs) -> list[tuple[str, str]]:
        return []  # none

    def reqd_args(self) -> list[str]:
        return ["conf_file", "topology"]  # note

    def write(self, scene: PLPSimulation, **kwargs) -> dict[str, str]:
        """
        writes scene at specified stage to a file, returns a set of parameters to write to input file
        """
        top: SWriter.SPatchyTopology = self.get_scene_top(scene)
        self.write_top(top, kwargs["topology"])
        self.write_conf(scene, kwargs["conf_file"])
        return {
            "topology": kwargs["topology"],
            "conf_file": kwargs["conf_file"]
        }

    def write_top(self, topology: SPatchyTopology, top_path: Union[str, Path]):
        """
        writes the provided topology to a topology file
        """
        if isinstance(top_path, str):
            top_path = Path(top_path)
        with self.file(top_path, "w") as f:
            # first line: num particles, num strands(1), num particles (again)
            f.write(f"{topology.num_particles()} 1 {topology.num_particles()}\n")
            f.write("\n")

            # then write patch info (equivelant to patches.txt in flavian)
            for i, patch in enumerate(topology.particle_set().patches()):
                assert i == patch.get_id(), "Patch index does not match patch ID"
                patch_position = np.array2string(
                    patch.position(),
                    separator=" ",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                a1 = np.array2string(
                    patch.a1(),
                    separator=" ",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                a3 = np.array2string(
                    patch.a3(),
                    separator=" ",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                f.write(f"iP {i} {patch.color()} {patch.strength()} {patch_position} {a1} {a3}\n")

            f.write("\n")

            # then write particle types info (equivelant to particles.txt in flavian)
            for particle_type in topology.particle_set():
                f.write(
                    f"iC {particle_type.type_id()} {' '.join([f'{patch.get_id()}' for patch in particle_type.patches()])}\n")

            f.write("\n")

            # then write particle information
            for ptypeid in topology.list_particles():
                f.write(f"-3 0 {ptypeid} {topology.particle_set().particle(ptypeid).radius()}\n")

    def read_top(self, top_file: str) -> SPatchyTopology:
        with self.file(top_file, "r") as f:
            patches: list[PLPatch] = []
            header = f.readline()
            nparticles = int(header.split()[0])
            particle_types_info: list[tuple[int, list[int]]] = []
            type_counts: dict[int, int] = {}
            type_radii: dict[int, float] = {}  # subhajit makes my life difficult
            # can actually ignore this one
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                matches = [float(f) for f in re.findall(r"-?\d+\.?\d*", line)]
                # line describes patch
                if line.startswith("iP"):
                    i, color, strength, x, y, z, a1x, a1y, a1z, a3x, a3y, a3z = matches
                    a1 = np.array([a1x, a1y, a1z])
                    a3 = np.array([a3x, a3y, a3z])
                    a2 = np.cross(a1, a3)
                    patch = PLPatch(i,
                                    color,
                                    np.array([x, y, z]),
                                    a1,
                                    a2,
                                    strength)
                    assert (patch.a3() - a3 < 1e-6).all()
                    patches.append(patch)
                # line describes a particle
                elif line.startswith("iC"):
                    particle_type_id = int(matches[0])
                    patch_ids = [int(i) for i in matches[1:]]
                    particle_types_info.append((particle_type_id, patch_ids))
                # line hopefully particle instance?
                else:
                    category, _, particle_type_id, radius = matches  # TODO: tolerance for longer lines? do not.
                    assert category == -3, "Not a particle! Ask subhajit."
                    if particle_type_id in type_counts:
                        type_counts[particle_type_id] += 1
                        type_radii[particle_type_id] = radius
                    else:
                        type_counts[particle_type_id] = 1
        particle_types = [
            PLPatchyParticle(
                [patches[patch_id] for patch_id in patch_ids],
                type_id=particle_id,
                radius=type_radii[particle_id] if particle_id in type_radii else 0.5  # hate hate hate
            )
            for particle_id, patch_ids in particle_types_info
        ]
        ptypes = PLParticleSet(particle_types)
        top = SWriter.SPatchyTopology(ptypes, type_counts)
        assert top.num_particles() == nparticles, "Didn't load particles correctly, somehow"
        return top

    def read_particle_types(self, topology) -> PLParticleSet:
        return self.read_top(topology).particle_set()

    def read_scene(self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx=None) -> PLPSimulation:
        top = self.read_top(top_file)
        traj_file = self.directory() / traj_file
        # terrified of this line of code, i do not think the ryereader method will play nice with subhajit's format
        top_info, traj_info = rr.describe(str(self.directory() / top_file), str(traj_file))
        # only retrieve last conf
        conf = rr.get_confs(top_info, traj_info, traj_info.nconfs - 1, 1)[0]
        scene = PLPSimulation()
        scene.set_particle_types(top.particle_set())
        scene.set_time(conf.time)
        return scene

    def get_scene_top(self, s: PLPSimulation) -> SPatchyTopology:
        return SWriter.SPatchyTopology(s.particle_types(), s.particle_type_counts())

    class SPatchyTopology(PLBaseWriter.PatchyTopology):
        _particle_set: PLParticleSet
        _particle_type_counts: dict[int, int]

        def __init__(self, particle_set: PLParticleSet, particle_type_counts: dict[int, int]):
            self._particle_set = particle_set
            self._particle_type_counts = particle_type_counts

        def particle_type_count(self, p) -> int:
            return self._particle_type_counts[p]

        def num_particle_types(self) -> int:
            return len(self._particle_type_counts)

        def num_particles(self) -> int:
            return sum(self._particle_type_counts.values())

        def particle_set(self) -> PLParticleSet:
            return self._particle_set

        def list_particles(self) -> Iterable[int]:
            return itertools.chain.from_iterable([[itype for i in range(n)]
                                                  for itype, n in self._particle_type_counts.items()])


class MalformedSimulationException(BaseException):
    pass


__writers = {
    "flavio": FWriter(),
    "josh_flavio": JFWriter(),
    # "josh_lorenzo": JLWriter(),
    "lorenzo": LWriter(),
    "subhajit": SWriter()
}


def get_writer(writer_key: Union[str, None] = None) -> PLBaseWriter:
    if writer_key is None:
        writer_key = get_server_config().patchy_format
    return __writers[writer_key]


def register_writer(writer_name: str, writer_obj: PLBaseWriter):
    __writers[writer_name] = writer_obj


def writer_options() -> list[str]:
    return [*__writers.keys()]