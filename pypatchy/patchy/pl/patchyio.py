from __future__ import annotations

import copy
import itertools
import re
from math import sqrt
from typing import Union, IO, Iterable, Generator
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import oxDNA_analysis_tools.UTILS.RyeReader as rr
from Bio.SVDSuperimposer import SVDSuperimposer
from oxDNA_analysis_tools.UTILS.data_structures import Configuration, TrajInfo, TopInfo

from .plpatch import PLPatch, assign_colors_one_to_one, assign_colors
from .plparticle import PLPatchyParticle
from .plparticleset import PLParticleSet
from .plscene import PLPSimulation
from ...interaction_matrix import InteractionMatrix
from ...polycubeutil.polycubesRule import PolycubeRuleCubeType
from ...util import normalize, DENTAL_RADIUS_KEY, NUM_TEETH_KEY, selectColor
from ...server_config import PatchyServerConfig, get_server_config


# no class to read mgls since it isn't an oxdna format
def read_mgl(file_path: Union[Path, str], color_map: Union[dict[tuple[str, str], float], None]) -> PLPSimulation:
    assert color_map is not None, "We have not implemented automatic color matching!"
    if isinstance(file_path, str):
        file_path = Path(file_path)
    assert file_path.is_file()
    with file_path.open("r") as f:
        # first line of mgl defines box
        first_line = f.readline()
        # read box siZe
        box_size = np.array([float(x) for x in re.findall(r'\d+\.\d+', first_line)])
        # init mgl scene
        mgl = PLPSimulation()
        mgl.set_particle_types(PLParticleSet())
        # set mgl box size
        mgl.set_box_size(box_size)
        mgl.compute_cell_size(n_cells=1)
        mgl.apportion_cells()
        # each line after the first is a particle
        patch_idx = 0
        type_map: dict[str, PLPatchyParticle] = {}
        # each remaining line in the mgl is a particle

        color_list: list[str] = []

        for i, line in enumerate(f):
            # first part of an mgl line is the particle position, second is other info, third is patches
            particle_position, particle_data, patch_strs = re.split(r"[@M]", line)
            particle_position = np.array([float(x) for x in particle_position.split()])
            # radius, color
            r, particle_color = particle_data.split()
            r = float(r)
            assert particle_color.startswith("C")
            # particle color indicates type
            # read particle type string
            particle_color = particle_color[particle_color.find("[") + 1:particle_color.rfind("]")]
            patch_strs = patch_strs.split()
            assert len(patch_strs) % 5 == 0
            patches: list[PLPatch] = []
            # no delimeter between patches
            j = 0
            # read patches
            while j < len(patch_strs):
                # each patch is specied by 4 numbers and a string
                # the first three numbers specify the patch position, the fourth is the patch width,
                # the string is fifth and is the patch color
                patch_coords = np.array([float(x) for x in patch_strs[j:j + 3]])
                # read patch width for completness sake, but ignore it
                w = float(patch_strs[j + 3])
                patch_color: str = re.search(r"C\[(\w+)]", patch_strs[j + 4]).group(1)
                if patch_color not in color_list:
                    color_list.append(patch_color)
                color_idx = color_list.index(patch_color)
                # patch
                patch = PLPatch(mgl.particle_types().num_patches(),
                                color_idx,
                                patch_coords)
                # "a1" doesn't strictly exist in mgl but it's very useful to have anyway
                patch.set_a1(normalize(patch.position()))
                patches.append(patch)
                patch_idx += 1
                j += 5
            # if we have a type for this particle
            if particle_color in type_map:
                assert len(patches) == type_map[particle_color].num_patches()
                # compute matrix of particle positions
                patch_color_orders = [patch.color() for patch in patches]
                particle_patch_posmatrix = np.stack([
                    patch.position() for patch in patches
                ])
                patch_type_color_order = [patch.color() for patch in type_map[particle_color].patches()]
                ptype_pos_matrix = np.stack([np.zeros((3,)), *[
                    patch.position() for patch in
                    type_map[particle_color].patches()
                ]])
                # oh god we need to use a svd superimposer
                sup = SVDSuperimposer()
                for order in itertools.permutations(range(len(patches))):
                    # skip orderings that don't align
                    if all([c1 == c2 for c1, c2 in
                            zip(patch_color_orders, [patch_type_color_order[n] for n in order])]):
                        sup.set(
                            np.stack([
                                np.zeros((3,)),
                                *[
                                    patch.position() for patch in patches
                                ]]),
                            ptype_pos_matrix)
                        sup.run()
                        rms = sup.get_rms()
                        # sure, this is good rms
                        if rms < 0.01:
                            p_type = type_map[particle_color]
                            rot = sup.rot.T
                            break
                    raise Exception("Couldn't find correct rotation for particles")

            else:
                p_type = PLPatchyParticle(patches,
                                          radius=r,
                                          type_id=mgl.particle_types().num_particle_types(),
                                          particle_name=particle_color)
                type_map[particle_color] = p_type
                mgl.particle_types().add_particle(p_type)
                rot = np.identity(3)
            p = p_type.instantiate(mgl.num_particles())
            p.set_rotation(rot)
            p.set_position(particle_position)
            mgl.add_particle(p)
        # set interaction matrix
        mgl.particle_types().set_interactions(InteractionMatrix({
            (color_list.index(c1_str), color_list.index(c2_str)): strength
            for (c1_str, c2_str), strength in color_map.items()
        }))
        return mgl


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

    def file(self, filename: Union[Path, str], access="w") -> IO:
        # allow for passing filename as an absolute path, bypassing directory()
        if isinstance(filename, Path) and filename.is_absolute():
            return filename.open(access)
        else:
            if access == "r" and not (self.directory() / filename).exists():
                raise FileNotFoundError(f"No file called {filename} in directory {self.directory()}")
            return (self.directory() / filename).open(access)

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
        """
        writes a conf
        """
        assert scene.get_conf() is not None
        conf = scene.get_conf()
        rr.write_conf(str(self.directory() / p), conf)

    @abstractmethod
    def read_particle_types(self, **kwargs):
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

        def get_particle_type(self, particle_id: int) -> int:
            return self.particle_ids[particle_id]

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

    def read_particle_types(self, patchy_file: str, particle_file: str, topology=None, conf_file=None) -> PLParticleSet:
        """
        Parameters:
            particle_file file which contains particle type info
            patchy_file file which conains patch type info
            topology unused arg, reqd for some handling nonsense
            conf_file unused arg, reqd for some handling nonsense
        """
        # read patch data from patches.txt or equivalent2
        patches, interactions = self.load_patches(patchy_file)
        particles: list[PLPatchyParticle] = []

        # open particles.txt or equivalent
        with self.file(particle_file, "r") as f:
            # iter lines
            lines = f.readlines()
            j = 0
            for line in lines:
                line = line.strip()
                # if line is not blank or a comment
                if len(line) > 1 and line[0] != '#':
                    # if the line is the beginning of a particle type block
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

        return particle_set

    def particle_from_lines(self,
                            lines: Iterable[str],
                            patches: list[PLPatch]) -> PLPatchyParticle:
        type_id = None
        patch_ids = None
        # loop lines
        for line in lines:
            # strip extra spaces
            line = line.strip()
            # skip comments
            if "=" in line and line[0] != '#':
                # extract key and value from `key = value`
                key, vals = line.split("=")
                key = key.strip()  # remove extra spaces
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
        # open topology file
        with self.file(top_path) as top_file:
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

        particle_fn = kwargs["particle_file"] if "particle_file" in kwargs else "particles.txt"
        patchy_fn = kwargs["patchy_file"] if "patchy_file" in kwargs else "patches.txt"
        init_top = kwargs["topology"] if "topology" in kwargs else "topology.top"
        init_conf = kwargs["conf_file"] if "conf_file" in kwargs else "conf.dat"
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
            "conf_file": str(init_conf)
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

    def load_patches(self, filename: str, num_patches=0) -> tuple[list[PLPatch], InteractionMatrix]:
        """
        loads patches from a patches.txt file and designs an interaction matrix
        """
        j = 0
        Np = 0
        patches = [PLPatch() for _ in range(num_patches)]
        fpath = self.directory() / filename
        assert fpath, f"No file called {filename}"
        # setup counter for oclors
        # set up mapping for color read from file -> patches
        # interaction map
        interactions = InteractionMatrix()
        # open pathces.txt file
        with fpath.open("r") as f:
            lines = f.readlines()
            # iter lines
            for line in lines:
                line = line.strip()
                # skip comments and empty lines
                if len(line) > 1 and line[0] != '#':
                    # if line is beginning of a patch block
                    if 'patch_' and '{' in line:
                        # make dict to store patch attributes
                        strargs = dict()
                        k = j + 1
                        # iter rest of pach block
                        while '}' not in lines[k]:
                            # skip comments
                            if not lines[k].isspace() and not line.startswith("#"):
                                # unpack key-value pairs
                                key, val = lines[k].split("=")
                                key = key.strip()
                                strargs[key] = val
                            k = k + 1
                        patch = PLPatch()
                        # print 'Loaded patch',strargs
                        patch.init_from_string(strargs)
                        # we gotta do color stuff
                        # if patch is self-interacting
                        if 0 < patch.color() < 21:
                            # all interaction strengths will be 1, strength here is set by patches
                            interactions[(patch.color(), patch.color())] = 1.
                        else:
                            interactions[(patch.color(), -patch.color())] = 1.
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
        return patches, interactions

    def read_scene(self, top_file: Path, traj_file: Path, particle_types: PLParticleSet,
                   conf_idx=None) -> PLPSimulation:
        """
        Reads a patchy particle scene from files
        """
        # assert conf_idx is None
        # find topology file path
        top_file: Path = self.directory() / top_file
        # find conf file path
        traj_file: Path = self.directory() / traj_file
        # rye reader: describe top + dat
        top_info, traj_info = rr.describe(str(top_file), str(traj_file))
        # only retrieve last conf
        # todo: more error checking
        if conf_idx is None:
            conf = rr.get_confs(top_info, traj_info, traj_info.nconfs - 1, 1)[0]
        else:
            assert traj_info.nconfs > conf_idx > -1
            conf = rr.get_confs(top_info, traj_info, conf_idx, 1)[0]
        # inbox conf
        conf = rr.inbox(conf, center=False)

        # construct empty scene object
        scene = PLPSimulation()
        # set time
        scene.set_time(conf.time)
        # set particle types
        scene.set_particle_types(particle_types)
        # set box size
        scene.set_box_size(conf.box)

        # open topology file
        with self.file(top_file, "r") as f:
            # first line has # particle types, #particles (we already know this, so skip it)
            f.readline()
            # second line is list of type IDs
            ptypelist = [int(i) for i in f.readline().split()]
            # compute + apportion cells, before we add partcles
            scene.compute_cell_size(n_particles=len(ptypelist))
            scene.apportion_cells()
            for i, ptype_idx in enumerate(ptypelist):
                # get particle type
                ptype: PLPatchyParticle = particle_types.particle(ptype_idx)
                # instantiate particle instance
                pp = PLPatchyParticle(
                    patches=ptype.patches(),
                    particle_name=f"{ptype.name()}_{i}",
                    type_id=ptype_idx,
                    index_=i,
                    position=conf.positions[i, :],
                )
                # set instance orientation from conf data
                pp.set_orientation(conf.a1s[i, :], conf.a3s[i, :])
                # todo: momentum et al?
                # add particle
                scene.add_particle(pp)
        return scene

    class FPatchyTopology(PLBaseWriter.PatchyTopology):

        nParticleTypes: int  # number of particle types
        type_counts: dict[int, int]  # particle type ID -> count

        def __init__(self, top_particles: list[Union[PLPatchyParticle, int]]):
            self.particle_ids = []
            for p in top_particles:
                if isinstance(p, int):
                    self.particle_ids.append(p)
                elif isinstance(p, PLPatchyParticle):
                    self.particle_ids.append(p.type_id())
                else:
                    raise Exception()
            self.nParticleTypes = max(self.particle_ids) + 1
            self.type_counts = {i: 0 for i in range(self.nParticleTypes)}
            for p in self.particle_ids:
                self.type_counts[p] += 1

        def num_particle_types(self) -> int:
            return self.nParticleTypes

        def particle_type_counts(self) -> dict[int, int]:
            return self.type_counts

        def particles(self) -> Iterable[int]:
            return self.particle_ids

        def num_particles(self) -> int:
            return len(self.particle_ids)

        def particle_type_count(self, p) -> int:
            return self.type_counts[p]


class JWriter(PLBaseWriter, ABC):
    """
    Writer class for Josh's file formats (aka ones with allostery)
    This is a modifiecation of Lorenzo's or Flavio/Petr's formats, so this
    class is abstract
    """

    def reqd_args(self) -> list[str]:
        return ["patchy_file",
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
        return LWriter.LPatchyTopology(PLParticleSet(particle_types, intmat=InteractionMatrix()), particle_type_counts)

    def get_scene_top(self, s: PLPSimulation) -> LPatchyTopology:
        return LWriter.LPatchyTopology(s.particle_types(), s.particle_type_counts())

    def read_particle_types(self, **kwargs) -> PLParticleSet:
        topology = kwargs["topology"]
        DPS_interaction_matrix_file = kwargs["DPS_interaction_matrix_file"]
        top = self.read_top(topology)
        # interaction matrices are encoded by patch type ID, rather than "color" so we kinda need to work backwards
        interaction_matrix = self.read_interaction_matrix(DPS_interaction_matrix_file)
        # recolor interaction matrix to minimize number of colors used
        interaction_matrix, new_colors = assign_colors(interaction_matrix, top.particle_types.patches())
        for color, patch in zip(new_colors, top.particle_types.patches()):
            patch.set_color(color)
        top.particle_types.set_interactions(interaction_matrix)
        return top.particle_types

    def read_interaction_matrix(self, interaction_file: str) -> InteractionMatrix:
        """
        reads an interactions file and returns an interaction matrix
        it's important to note that this will return an interaction matrix for
        PATCH ID PAIRS, NOT COLORS
        """
        pattern = r"patchy_eps\[-?(\d+)\]\[-?(\d+)\] = (\d+\.?\d*)"
        data: list[tuple[tuple[int, int], float]] = list()
        max_index = 0
        with self.file(interaction_file, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                assert match, f"Malformed interaction file line {line}"
                extracted = tuple(map(int, match.groups()[:-1])), float(match.group(3))
                indices, value = extracted
                max_index = max(max_index, *indices)
                # if indices[0] == indices[1]:
                #     # TOOD: support
                #     raise ValueError(f"Patch in line {line} interacts with itself, not currently supported!")
                data.append((indices, value))
        return InteractionMatrix(data)

    def write_top(self, topology: LPatchyTopology, top_file: str):
        with self.file(top_file) as f:
            f.write(f"{len(topology.particles())} {topology.particle_types.num_particle_types()}\n")
            for ptype in topology.particle_types:
                # add particle file name to files list
                # particles_txts_files.append(self.directory() / f"patches_{particle.type_id()}.dat")
                f.write(self.particle_type_str(ptype,
                                               topology.particle_type_count(ptype.type_id())) + "\n")

    def read_scene (self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx: Union[None, int] = None) -> PLPSimulation:
        top: LWriter.PatchyTopology = self.read_top(top_file)
        scene = PLPSimulation()
        scene.set_particle_types(particle_types)
        top_info, traj_info = rr.describe(str(self.directory() / top_file),
                                          str(self.directory() / traj_file))
        if conf_idx is None:
            conf = rr.get_confs(top_info, traj_info, traj_info.nconfs - 1, 1)[0]
        else:
            assert conf_idx < traj_info.nconfs, f"Trying to read conf {conf_idx} from traj at " \
                                                f"{str(self.directory() / traj_file)} but file only contains {traj_info.nconfs} confs!"
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
        return ["topology", "conf_file", "DPS_interaction_matrix_file"]

    def write(self,
              scene: PLPSimulation,
              **kwargs
              ) -> dict[str, str]:
        assert self.directory() is not None, "No writing directory specified!"
        assert self.directory().exists(), f"Specified writing directory {str(self.directory())} does not exist!"

        particles: PLParticleSet = scene.particle_types()
        scene.sort_particles_by_type()

        init_top = kwargs["topology"] if "topology" in kwargs else "topology.top"
        init_conf = kwargs["conf_file"] if "conf_file" in kwargs else "init.dat"

        interactions_file = kwargs["DPS_interaction_matrix_file"] if "DPS_interaction_matrix_file" in kwargs else "interactions.txt"
        # self.export_interaction_matrix(particles.interaction_matrix(), interactions_file)
        # self.export_interaction_matrix(particles.patches(), interactions_file)
        self.export_interaction_matrix(particles, interactions_file)

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

    # def export_interaction_matrix(self, mat: InteractionMatrix, filename: str):
    #     with self.file(filename, "w") as f:
    #         f.writelines(
    #             [
    #                 f"patchy_eps[{c1}][{c2}] = {strength}\n" for (c1, c2), strength in mat
    #                 # use geometric mean of patch strengths, best for backwards compatibility
    #                 # f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = {sqrt(p1.strength() * p2.strength()) * mat[p1.color(), p2.color()]}\n"
    #                 # for p1, p2 in itertools.combinations(patches, 2)
    #                 # if mat[p1.color(), p2.color()]
    #             ]
    #         )
    #
    # def export_interaction_matrix(self, patches: list[PLPatch], filename: str):
    #     with self.file(filename, "w") as f:
    #         f.writelines(
    #             [
    #                 f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = {p1.strength()}\n"
    #                 for p1, p2 in itertools.combinations(patches, 2)
    #                 if p1.color() == -p2.color()
    #             ]
    #         )
    def export_interaction_matrix(self, particle_types: PLParticleSet, filename: str):
        with self.file(filename, "w") as f:
            f.writelines(
                [
                    # use geometric mean of patch strengths, best for backwards compatibility
                    f"patchy_eps[{p1.type_id()}][{p2.type_id()}] = {sqrt(p1.strength() * p2.strength()) * particle_types.interaction_matrix()[p1.color(), p2.color()]}\n"
                    for p1, p2 in itertools.combinations(particle_types.patches(), 2)
                    if particle_types.interaction_matrix()[p1.color(), p2.color()]
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
        ptypes = PLParticleSet(intmat=particle_types)
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


class RWriter(PLBaseWriter):
    """
    writer for raspberry-type particles
    """

    def get_input_file_data(self, scene: PLPSimulation, **kwargs) -> list[tuple[str, str]]:
        return []  # no input file info

    def reqd_args(self) -> list[str]:
        return ["conf_file", "topology"]

    def write(self, scene: PLPSimulation, **kwargs) -> dict[str, str]:
        init_top = kwargs["topology"]
        init_conf = kwargs["conf_file"]
        # raspberry format requires that particles be sorted by type
        scene_sorted = copy.deepcopy(scene)
        scene_sorted.sort_particles_by_type()
        if "scene_top" in kwargs:
            scene_top = kwargs["scene_top"]
        else:
            scene_top = self.get_scene_top(scene_sorted)
        self.write_top(scene_top, top_path=init_top)
        self.write_conf(scene_sorted, init_conf)
        return {
            "topology": init_top,
            "conf_file": init_conf
        }

    def write_top(self, topology: RPatchyTopology, top_path: Union[str, Path]):
        with self.file(top_path) as f:
            # required for oxDNA topolgy standard: this header
            f.write(f"{topology.num_particles()} {topology.num_particle_types()}\n")
            f.write("# Patch types\n")
            # iter patches
            for i, patch in enumerate(topology.particle_types.patches()):
                assert i == patch.get_id(), "Patch index does not match patch ID"
                # convert patch position to string
                patch_position = np.array2string(
                    patch.position(),
                    separator=",",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                # convert patch a1 to string
                a1 = np.array2string(
                    patch.a1(),
                    separator=",",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                f.write(f"iP {i} {patch.strength()} {patch.color()} {patch_position} {a1}\n")
            f.write("\n# Repulsion points\n")
            for site_position, r in topology.particle_repulsion_sites:
                position = np.array2string(
                    np.array(site_position),
                    separator=",",
                    suppress_small=True,
                    formatter={'float_kind': custom_formatter}
                )[1:-1]
                f.write(f"iR {position} {r}\n")
            f.write("\n# Particle Types\n")
            for particle_type in topology.particle_types.particles():
                patch_ids_str = ','.join([str(patch.get_id()) for patch in particle_type.patches()])
                f.write(f"iC {particle_type.type_id()} {topology.particle_type_count(particle_type.type_id())} {patch_ids_str} {','.join([str(i) for i in range(len(topology.particle_repulsion_sites))])}\n")

    def read_top(self, top_file: str) -> RWriter.RPatchyTopology:
        particle_types = self.read_particle_types(top_file)
        repulsion_sites = []
        type_counts = dict()
        with self.file(top_file, 'r') as f:
            for line in f:
                if line.startswith("iC"):
                    _, type_id, count, _, _ = line.split()
                    type_counts[int(type_id)] = int(count)
                if line.startswith("iR"):
                    # read repulsion site info
                    _, position, radius = line.split()
                    position = np.array([float(i) for i in position.split(",")])
                    assert len(position) == 3
                    repulsion_sites.append((position, float(radius)))

        top = RWriter.RPatchyTopology(particle_types, type_counts)
        top.particle_repulsion_sites = repulsion_sites
        return top

    def read_particle_types(self, particle_type_file: str) -> PLParticleSet:
        particle_type_info = []
        patches = dict()
        with self.file(particle_type_file, 'r') as f:
            for line in f:
                if line.startswith("iC"):
                    # read particle ("corpsicule") type
                    _, type_id, _, patches_list, repulsion_sites = line.split()
                    type_id = int(type_id)
                    patches_list = [int(i) for i in patches_list.split(",")]
                    repulsion_sites = [int(i) for i in repulsion_sites.split(",")]
                    particle_type_info.append((type_id, patches_list, repulsion_sites))


                elif line.startswith("iP"):
                    # read patch info
                    _, type_id, strength, color, position, a1 = line.split()
                    type_id = int(type_id)
                    strength = float(strength)
                    color = int(color)
                    position = np.array([float(i) for i in position.split(",")])
                    assert len(position) == 3
                    a1 = np.array([float(i) for i in a1.split(",")])
                    assert len(a1) == 3
                    patches[type_id] = PLPatch(type_id,
                                           color,
                                           position,
                                           a1,
                                           strength=strength)
        # todo: read interaction matrix
        # assert all particle types have same repulsion sites
        # todo: impl different shapes?
        assert all([tuple(rss) == tuple(particle_type_info[0][2]) for _, _, rss in particle_type_info])
        return PLParticleSet(particles=[
            PLPatchyParticle([patches[i] for i in patches_list],
                             type_id=type_id)
            for type_id, patches_list, repulsion_sites in particle_type_info
        ])

    def read_scene(self, top_file: Union[Path, str], traj_file: Union[Path, str], particle_types: PLParticleSet,
                   conf_idx=None) -> PLPSimulation:
        if conf_idx is None:
            conf_idx = 0 # ?
        topology = self.read_top(top_file)

        t, conf = rr.describe(str(self.directory() / top_file),
                           str(self.directory() / traj_file))
        conf = rr.get_confs(t, conf, conf_idx,1)[0]


        sim = PLPSimulation()
        sim.set_box_size(conf.box)
        sim.compute_cell_size(topology.num_particles())
        sim.apportion_cells()
        sim.set_particle_types(topology.particle_types)
        for i, ptype in enumerate(topology.iter_particles()):
            p = ptype.instantiate(i)
            sim.add_particle(p)
            p.set_position(conf.positions[i, :])
            p.a1 = conf.a1s[i, :]
            p.a3 = conf.a3s[i, :]
        return sim

    def get_scene_top(self, s: PLPSimulation) -> RPatchyTopology:
        return RWriter.RPatchyTopology(s.particle_types(), s.particle_type_counts())

    class RPatchyTopology(PLBaseWriter.PatchyTopology):
        # todo: better interaction site resolution
        # todo: actually optimize this for cube
        particle_repulsion_sites: list[tuple[list, float]]
        type_counts: dict[int, int]  # particle type ID -> count
        particle_types: PLParticleSet

        def __init__(self, particles: PLParticleSet, type_counts: dict[int, int]):
            self.type_counts = type_counts
            self.particle_types = particles
            self.particle_repulsion_sites = [
            (coords, 0.25) for coords in itertools.product(
                [0.25, -0.25],
                [0.25, -0.25],
                [0.25, -0.25]
            )
        ]

        def info(self, fp: str="") -> TopInfo:
            # can use dummy fp if it doesn't matter
            return TopInfo(fp, self.num_particles())

        def iter_particles(self) -> Generator[PLPatchyParticle, None, None]:
            """
            iterates particle TYPES by count
            """
            for particle_type in self.particle_types:
                for i in range(self.type_counts[particle_type.type_id()]):
                    yield particle_type

        def particle_type_count(self, p: int) -> int:
            return self.type_counts[p]

        def num_particle_types(self) -> int:
            return len(self.type_counts)

        def num_particles(self) -> int:
            return sum(self.type_counts.values())

    def write_as_mgl(self, fp: str, top: RPatchyTopology, traj: TrajInfo, interval: int=1):
        """
        writes the particle system as a mgl file
        this version writes repulsion sites as spheres, as opposed to the MGLParticle class
        methods
        """
        with self.file(fp) as f:
            for i in range(0, traj.nconfs, interval):
                # to prevent memory explosion read confs seperately
                conf = rr.get_confs(top.info(), traj, i, 1)[0]
                conf = rr.inbox(conf)
                self.write_mgl(f, top, conf)

    def write_mgl(self, file: IO, top: RWriter.RPatchyTopology, conf: Configuration):
        file.write(f".Box:{np.array2string(conf.box, precision=2, separator=',')[1:-1]}\n")
        for i, particle_type in enumerate(top.iter_particles()):
            p = copy.deepcopy(particle_type)
            p.set_position(conf.positions[i])
            p.a1 = conf.a1s[i]
            p.a3 = conf.a3s[i]

            # choose color
            particle_color = selectColor(particle_type.type_id(), fmt='arr')
            particle_color = ",".join([str(i) for i in particle_color])
            # write repulsion sites
            for (repulsion_pt_position, r) in top.particle_repulsion_sites:
                repulsion_pt_position = p.position() + repulsion_pt_position @ p.rotmatrix()
                repulsion_pt_position_str = np.array2string(repulsion_pt_position,
                                              suppress_small=True,
                                              precision=4,
                                              formatter={'float_kind': custom_formatter})[1:-1].strip()
                file.write(f"{repulsion_pt_position_str} @ {r} C[{particle_color}]\n")
            # write patches
            # mgl doens't like conic patches w/o mgl a central sphere, so we will have each patch have a dummy particle
            for i, patch in enumerate(particle_type.patches()):
                patch_color = selectColor(abs(patch.color()), saturation=65 if patch.color() > 0 else 25, fmt="arr")
                patch_color = ",".join([str(round(i, 1)) for i in patch_color])
                patch_width = 2 * 0.082 # todo: settable
                # best way to write patches is to make each patch a particle *ugh*
                position = np.array2string(p.patch_position(patch),
                                           suppress_small=True,
                                           precision=4,
                                           formatter={'float_kind': custom_formatter})[1:-1]
                width = np.array2string(p.patch_a1(patch) * patch_width,
                                        suppress_small=True,
                                        precision=4,
                                        formatter={'float_kind': custom_formatter})[1:-1]
                file.write(f"{position} @ 0 C[0,0,0] M {width} {patch_width * 4} C[{patch_color}]\n")


class MalformedSimulationException(BaseException):
    pass


__writers = {
    "flavio": FWriter,
    "josh_flavio": JFWriter,
    # "josh_lorenzo": JLWriter(),
    "lorenzo": LWriter,
    "subhajit": SWriter,
    "raspberry": RWriter
}


def get_writer(writer_key: Union[str, None] = None) -> PLBaseWriter:
    if writer_key is None:
        writer_key = get_server_config().patchy_format
    return __writers[writer_key]()


def register_writer(writer_name: str, writer_obj: PLBaseWriter):
    __writers[writer_name] = writer_obj


def writer_options() -> list[str]:
    return [*__writers.keys()]
