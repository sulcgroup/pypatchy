import tempfile
from pathlib import Path

import pytest

# ipy_oxDNA imports
from ipy_oxdna.structure_editor.dna_structure import load_dna_structure, rand_seq

# pypatchy imports
from pypatchy.patchy.dna_particle import DNAParticle
from pypatchy.patchy.pl.patchyio import get_writer
from pypatchy.patchy.pl.plparticle import PLPatchyParticle
from pypatchy.patchy.pl.plpotential import PLFRPatchyPotential, PLFRTorsionalPatchyPotential
from pypatchy.patchy.pl.plscene import PLPSimulation
from pypatchy.polycubeutil.polycube_structure import PolycubeStructure
from test_polycube_lib import ducky
from test_patchy_io import multidentate_patchy_ducky

@pytest.fixture
def temp_dir() -> str:
    # construct temporary directory to run

    with tempfile.TemporaryDirectory() as td:
        yield td

@pytest.fixture
def cube() -> DNAParticle:
    # assign strand IDs for patches
    # these strands will be 3' extended to produce single-stranded overhands which join the monomers
    patch_positions = [
        [128, 131, 130, 136],  # front
        [151, 155, 152, 160],  # bot
        [168, 163, 144, 156],  # left
        [146, 147, 149, 150],  # top
        [138, 140, 139, 141],  # right
        [142, 167, 143, 153]  # back
    ]
    # load a DNA structure to use as our monomer.
    # here we will load the files at ~/pypatchy/input/scenes/cube_monomer.top and ~/pypatchy/input/scenes/cube_monomer.
    # unlike the aforementioned load_polycube function this will NOT automatically prepent ~/.pypatchy/input
    dna = load_dna_structure(Path(__file__).parent / "test_files" / "dna_confs" / "structure_cube"/"cube_monomer.top",
                             Path(__file__).parent / "test_files" / "dna_confs" / "structure_cube" / "cube_monomer.dat")
    # create DNA monomer from DNA structure and patch positions
    dna_particle = DNAParticle(dna, patch_positions)
    return dna_particle


def icosahedron() -> DNAParticle:
    patchPositions = [
        [40, 21, 46],
        [28, 9, 32],
        [60, 75, 74],
        [146, 161, 160],
        [127, 132, 109],
        [138, 153, 175]
    ]

    # load dna icosahedron
    dna = load_dna_structure(Path(__file__).parent / "test_file" / "dna_confs" / "icosahedron" / "ico_monomer.top",
                             Path(__file__).parent / "test_file" / "dna_confs" / "icosahedron" /"ico_monomer.dat")
    return DNAParticle(dna, patchPositions)


def test_convert_wireframe_cube(temp_dir: str, cube:DNAParticle):
    from pypatchy.patchy.pl.plpatchylib import polycube_to_pl

    from ipy_oxdna.structure_editor.dna_structure import load_dna_structure, rc
    from pypatchy.patchy.dna_particle import DNAParticle
    from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter
    # from pypatchy.patchy.pl.patchio import get_writer
    from pypatchy.util import get_input_dir
    from pypatchy.polycubeutil.polycube_structure import load_polycube
    from pypatchy.util import get_output_dir
    from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings

    name = "wireframe"

    patch_positions = [
        [128, 131, 130, 136],  # front
        [151, 155, 152, 160],  # bot
        [168, 163, 144, 156],  # left
        [146, 147, 149, 150],  # top
        [138, 140, 139, 141],  # right
        [142, 167, 143, 153]  # back
    ]

    wireframe_polycube = load_polycube(Path(__file__).parent / "test_files" / "cube_structures" / "wireframecube.json")
    wireframe_multidentate = polycube_to_pl(wireframe_polycube,
                                            MultidentateConvertSettings(n_teeth=4,
                                            dental_radius=0.5)
                                            )
    wireframe_multidentate.add_potential(
        PLFRPatchyPotential(
            rmax=0.4 * 3,
            alpha=0.12,
        )
    )

    # validate converstion
    for cube1, cube2 in wireframe_polycube.iter_bound_particles():
        assert wireframe_multidentate.particles_bound(cube1.get_uid(), cube2.get_uid()), f"Cubes {cube1.get_uid()} and {cube2.get_uid()} are bound in the polycube but not in the patchy scene"
        # count polycube bindings between cube 1 and cube 2
        cube_bindngs_count = len(list(wireframe_polycube.iter_binding_patches(cube1, cube2)))
        # find corresponding patychy particles
        p1: PLPatchyParticle = wireframe_multidentate.get_particle(cube1.get_uid())
        p2: PLPatchyParticle = wireframe_multidentate.get_particle(cube2.get_uid())
        # count bindings between them
        pl_bindings = list(wireframe_multidentate.iter_binding_patches(p1, p2))
        assert len(pl_bindings) > 0, f"Particles {cube2.get_uid()} amd {cube2.get_uid()} are bound but no patch bond found!"
        assert 4 * cube_bindngs_count ==  len(pl_bindings),\
            f"Mismatch between number bonds between {cube1.get_uid()} and {cube2.get_uid()} " \
            f"from polycube to patchy."

    get_writer("flavio").set_directory(temp_dir)
    get_writer("flavio").write(wireframe_multidentate,
                               topology="topology.top",
                               conf_file="conformation.dat",
                               particle_file="particles.txt",
                               patchy_file="patches.txt")

    # load dna particles
    dna = load_dna_structure(get_input_dir() / "scenes/cube_monomer.top",
                             get_input_dir() / "scenes/cube_monomer.dat")
    # scene = polycube_to_pl(polycube, nteeth=4, dental_radius=0.25)
    dna_particle = DNAParticle(dna, patch_positions)

    converter = PatchyOrigamiConverter(wireframe_multidentate,
                                       padding=2,
                                       spacer_length=12,
                                       expected_num_edges=4 * wireframe_polycube.num_connections(),
                                       sticky_length=None,
                                       rel_rms_tolerance=1
                                       )
    # assign colors manually
    for icolor in converter.scene.particle_types().patch_colors():
        if icolor > 0:
            converter.assign_color_sequence(icolor, rand_seq(10))

    converter.assign_particles(dna_particle)
    converter.dump_monomers(name)
    converter.position_particles()
    converter.convert()
    converter.save_oxview(write_oxview_path=f"{temp_dir}/wireframe.oxview", color_by_type=True)
    pass

def test_convert_human(temp_dir: str, cube: DNAParticle):
    """
    tests convert a human polycube to dna origami
    """
    from pypatchy.patchy.pl.plpatchylib import polycube_to_pl

    from ipy_oxdna.structure_editor.dna_structure import load_dna_structure, rc
    from pypatchy.patchy.dna_particle import DNAParticle
    from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter
    # from pypatchy.patchy.pl.patchio import get_writer
    from pypatchy.util import get_input_dir
    from pypatchy.polycubeutil.polycube_structure import load_polycube
    from pypatchy.util import get_output_dir
    from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings

    name = "duck"

    patch_positions = [
        [128, 131, 130, 136],  # front
        [151, 155, 152, 160],  # bot
        [168, 163, 144, 156],  # left
        [146, 147, 149, 150],  # top
        [138, 140, 139, 141],  # right
        [142, 167, 143, 153]  # back
    ]

    human_polycube = load_polycube(Path(__file__).parent / "test_files" / "cube_structures" / "human.json")
    assert human_polycube.num_connections() == 12, f"Did not load polycube corrected! Expected number of cube-cube connections is 12, got {human_polycube.num_connections()}"
    human_multidentate = polycube_to_pl(human_polycube, MultidentateConvertSettings(n_teeth=4,
                                                                                    dental_radius=0.5))
    human_multidentate.add_potential(
        PLFRTorsionalPatchyPotential(
            rmax=0.4 * 3,
            alpha=0.12,
            narrow_type=2
        )
    )

    get_writer("flavio").set_directory(temp_dir)
    get_writer("flavio").write(human_multidentate,
                               topology="topology.top",
                               conf_file="conformation.dat",
                               particle_file="particles.txt",
                               patchy_file="patches.txt")

    # load dna particles
    dna = load_dna_structure(get_input_dir() / "scenes/cube_monomer.top",
                             get_input_dir() / "scenes/cube_monomer.dat")
    # scene = polycube_to_pl(polycube, nteeth=4, dental_radius=0.25)
    dna_particle = DNAParticle(dna, patch_positions)

    converter = PatchyOrigamiConverter(human_multidentate,
                                       padding=2,
                                       spacer_length=12,
                                       expected_num_edges=4 * human_polycube.num_connections(),
                                       sticky_length=9,
                                       rel_rms_tolerance=1
                                       # flexable_patch_distances=True
                                       )
    # converter.assign_color_sequence(22, "ATGCATCG", )

    converter.assign_particles(dna_particle)
    converter.dump_monomers(name)
    converter.position_particles()
    converter.convert()
    converter.save_oxview(write_oxview_path=f"{temp_dir}/human_dna.oxview", color_by_type=True)
    pass


def test_convert_swan(temp_dir: str, multidentate_patchy_ducky: PLPSimulation, cube: DNAParticle):
    from pypatchy.patchy.pl.plpatchylib import polycube_to_pl

    from ipy_oxdna.structure_editor.dna_structure import load_dna_structure, rc
    from pypatchy.patchy.dna_particle import DNAParticle
    from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter
    # from pypatchy.patchy.pl.patchio import get_writer
    from pypatchy.util import get_input_dir
    from pypatchy.polycubeutil.polycube_structure import load_polycube
    from pypatchy.util import get_output_dir
    from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings

    name = "duck"

    patch_positions = [
        [128, 131, 130, 136],  # front
        [151, 155, 152, 160],  # bot
        [168, 163, 144, 156],  # left
        [146, 147, 149, 150],  # top
        [138, 140, 139, 141],  # right
        [142, 167, 143, 153]  # back
    ]

    # load dna particles
    # scene = polycube_to_pl(polycube, nteeth=4, dental_radius=0.25)
    dna_particle = DNAParticle(cube, patch_positions)

    converter = PatchyOrigamiConverter(multidentate_patchy_ducky,
                                       padding=1.5,
                                       spacer_length=12,
                                       expected_num_edges=4*7,
                                       sticky_length=9,
                                       rel_rms_tolerance=1
                                       # flexable_patch_distances=True
                                       )
    start_color_code = 21
    converter.assign_particles(dna_particle)
    converter.dump_monomers(name)
    converter.position_particles()
    converter.convert()
    converter.save_oxview(write_oxview_path=f"{temp_dir}/ducky_dna.oxview", color_by_type=True)
    pass


# def test_generate_scene_convert(temp_dir: str, cube: DNAParticle):
#     # nessecary imports
#     from pathlib import Path
#
#     import numpy as np
#
#     from pypatchy.patchy.dna_particle import DNAParticle
#     from pypatchy.patchy.pl.patchyio import get_writer, LWriter
#     from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
#     from pypatchy.polycubeutil.polycubesRule import PolycubesRule
#
#     from pypatchy.patchy.pl.plpatchylib import polycube_rule_to_PL
#     rule = PolycubesRule(rule_str="1|||||_-1|||||")
#     pl_set = polycube_rule_to_PL(rule)
#     reader = get_writer("lorenzo")
#     top = LWriter.LPatchyTopology(pl_set, [1, 1])
#     (Path(temp_dir) / "patchy_scene").mkdir()
#
#     reader.set_directory(Path(temp_dir) / "patchy_scene")
#     reader.write_top(top, "particles.top")
#     reader.export_interaction_matrix(pl_set.interaction_matrix(), "interactions.txt")
#
#     # lookup patchy writer object
#     # our patchy files are in Lorenzo's format so use the build in Lorenzo-Writer
#     # (it can read or write particle data but today we are using it as a reader)
#     # honestly this should be a context manager, srry
#
#     # first load patchy particle set
#     # this varies a lot between patchy particle formats, this is only the correct method for Lorenzo-format
#     patchy_particle_set = reader.read_particle_types(topology="particles.top",
#                                                      DPS_interaction_matrix_file="interactions.txt", )
#     # specify settings that the program will use to convert the polycube to a patchy particle set
#     convert_settings = MultidentateConvertSettings(
#         n_teeth=4,  # each patch on the particle corresponds to 4 "multidentate" patches on the particles,
#         # which will be mapped to 4 strands on the DNA particle monomer (we'll load that later)
#         dental_radius=0.5,  # the distance in simulation units from the center of the patch to the "teeth"
#         torsion=False
#     )
#
#     # this is finicky unfortunately
#     # patches (even non-torsional) need an a2 vector to use as a basis for multidentate conversion but
#     # particle sets saved in the Lorenzo format are missing this
#     # however we can do it manually
#     for patch in patchy_particle_set.patches():
#         patch.set_a2(np.array([0, 1, 0]))
#     # convert particles to multidentate
#     patchy_particle_set = patchy_particle_set.to_multidentate(convert_settings)
#
#     scene = reader.read_scene(top_file="particles.top",  # topology file
#                               traj_file="last_conf.dat",  # here we will give it a single conf
#                               # but if we wanted we could pass a trajectory and a timestep
#                               particle_types=patchy_particle_set
#                               )
#
#     # if we wanted to convert this scene to multidentate we would do it here,
#     # but the scene is already multidentate
#
#     get_writer("flavio").set_directory("/home/josh/lab_data/test_l_particles")
#     get_writer("flavio").write(scene, topology="scene.top", conf_file="scene.conf", particle_file="particles.txt",
#                                patchy_file="patches.txt")
#
#     # imports
#     from ipy_oxdna.dna_structure import load_dna_structure
#     from pypatchy.util import get_input_dir
#     writer = get_writer("lorenzo")
#     writer.set_directory("/home/josh/git/pypatchy/tests/test_files/patchy_scenes/loro_eg")
#     scene = writer.read_scene("LORO.lorenzo.topology.top",
#                               "last_conf.dat",
#                               writer.read_particle_types("LORO.lorenzo.topology.top",
#                                                          "LORO.interaction_matrix.txt"))
#
#     # imports
#     from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter
#
#     converter = PatchyOrigamiConverter(scene,
#                                        sticky_length=10,  # length of the binding portion of single-stranded overhangs
#                                        # if we set this to a number, the program will generate a random
#                                        # sequence (not guaranteed to be orthogonal!!!!)
#                                        # if we set this to None you will have to manually enter all sequences
#                                        # below
#                                        spacer_length=12,  # poly-T spacer which will seperate the binding portion of the
#                                        # overhang from the rest of the staple strand
#                                        # there are a lot more optional args
#                                        )
#     converter.scale_factor = 48
#     # patchy color sequences start with 21 and -21 and proceed from there
#     converter.assign_color_sequence(21,  # color in patchy particle
#                                     "ATGCATCGAT",  # sequence
#                                     update_rc=True  # the converter will automatically assign -21 to the
#                                     # reverse compliment ATCGATGCAT (True is default value)
#                                     )
#
#     # set the converter to use our DNA particle from before
#     # if you want to have multiple distinct DNA monomers for different particle types,
#     # you can assign them here
#     converter.assign_particles(cube)
#
#     # position DNA particles
#     converter.position_particles()
#
#     # join them together
#     # can take a while
#     converter.convert(True)
#
#     # if you provide a relative path, pypatchy will place in the directory ~/.pypathy/output
#     # alternatively, if you provide an absolute path (beginning with "/") the program will use that directly
#     converter.save_top_dat(write_top_path=f"mystructure_dna.top",
#                            write_conf_path=f"mystructure_dna.conf")
#
#     converter.save_oxview(write_oxview_path="mystructure_dna.oxview",
#                           color_by_type=True)
#     converter.export_stickys_staples(Path(temp_dir) / "staples.xlsx")


def test_convert_3d_cross_polycube(temp_dir: str, cube: DNAParticle):
    pass

def test_convert_3d_cross_patchy():
    pass

def test_convert_tetrastack():
    pass

def test_subho_convert():
    pass