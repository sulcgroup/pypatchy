import tempfile

# ipy_oxDNA imports
from ipy_oxdna.structure_editor.dna_structure import load_dna_structure, rand_seq, recolor_top_dat
from ipy_oxdna.oxdna_simulation import Simulation

from pypatchy.patchy.dna_particle import DNAParticle
from pypatchy.patchy.patchy_origami_convert import PatchyOrigamiConverter
from pypatchy.patchy.pl.plparticle import PLPatchyParticle
from pypatchy.patchy.pl.plparticleset import MultidentateConvertSettings
from pypatchy.patchy.pl.plpatchylib import polycube_to_pl
from pypatchy.patchy.pl.plpotential import PLFRPatchyPotential
from pypatchy.polycubeutil.polycube_structure import load_polycube
from pypatchy.util import get_input_dir

# self-contained test/demonstration tah

def test_oxview_recolor():
    with tempfile.TemporaryDirectory() as td:
        name = "wireframe"

        patch_positions = [
            [128, 131, 130, 136],  # front
            [151, 155, 152, 160],  # bot
            [168, 163, 144, 156],  # left
            [146, 147, 149, 150],  # top
            [138, 140, 139, 141],  # right
            [142, 167, 143, 153]  # back
        ]

        # path to polycube file
        ducky_polycube = load_polycube("/home/josh/git/pypatchy/tests/test_files/cube_structures/duck_polycube.json")
        ducky_multidentate = polycube_to_pl(ducky_polycube,
                                            MultidentateConvertSettings(n_teeth=4,
                                                                            dental_radius=0.5)
                                            )
        ducky_multidentate.add_potential(
            PLFRPatchyPotential(
                rmax=0.4 * 3,
                alpha=0.12,
            )
        )

        # show_scene_multigraph(ducky_multidentate)

        # validate converstion
        for cube1, cube2 in ducky_polycube.iter_bound_particles():
            assert ducky_multidentate.particles_bound(cube1.get_uid(),
                                                      cube2.get_uid()), f"Cubes {cube1.get_uid()} and {cube2.get_uid()} are bound in the polycube but not in the patchy scene"
            # count polycube bindings between cube 1 and cube 2
            cube_bindngs_count = len(list(ducky_polycube.iter_binding_patches(cube1, cube2)))
            # find corresponding patychy particles
            p1: PLPatchyParticle = ducky_multidentate.get_particle(cube1.get_uid())
            p2: PLPatchyParticle = ducky_multidentate.get_particle(cube2.get_uid())
            # count bindings between them
            pl_bindings = list(ducky_multidentate.iter_binding_patches(p1, p2))
            assert len(
                pl_bindings) > 0, f"Particles {cube2.get_uid()} amd {cube2.get_uid()} are bound but no patch bond found!"
            assert 4 * cube_bindngs_count == len(pl_bindings), \
                f"Mismatch between number bonds between {cube1.get_uid()} and {cube2.get_uid()} " \
                f"from polycube to patchy."

        # load dna particles
        dna = load_dna_structure(get_input_dir() / "scenes/cube_monomer.top",
                                 get_input_dir() / "scenes/cube_monomer.dat")
        # scene = polycube_to_pl(polycube, nteeth=4, dental_radius=0.25)
        dna_particle = DNAParticle(dna, patch_positions)

        converter = PatchyOrigamiConverter(ducky_multidentate,
                                           padding=1.1,
                                           spacer_length=12,
                                           expected_num_edges=4 * ducky_polycube.num_connections(),
                                           sticky_length=None,
                                           rel_rms_tolerance=1
                                           )
        # assign colors manually
        # insert specific sequences here if you want
        for icolor in converter.scene.particle_types().patch_colors():
            if icolor > 0:
                converter.assign_color_sequence(icolor, rand_seq(10))

        converter.assign_particles(dna_particle)
        converter.dump_monomers(name)
        converter.position_particles()
        converter.convert()
        converter.save_oxview(write_oxview_path=f"{td}/ducky.oxview", color_by_type=True)

        converter.save_top_dat(f"{td}/init.top",
                               f"{td}/init.dat")

        # -------------- use this code to run a simulation and recolor the result ------------------------
        # ------- run a monte carlo relax ----------------------
        mc_relax = Simulation(file_dir=td, sim_dir=f"{td}/relax_mc/")
        mc_relax.build()
        mc_relax.input.swap_default_input("cpu_MC_relax")
        mc_relax.input["T"] = "30C"
        mc_relax.input["box_type"] = "orthogonal"
        mc_relax.input["steps"] = 1000
        mc_relax.input["print_conf_interval"] = 10
        mc_relax.input["print_energy_every"] = 10

        mc_relax.oat.generate_force(join=True)
        mc_relax.oxpy_run(subprocess=False)

        # top file
        recolor_top_source = mc_relax.sim_dir / mc_relax.input.top
        recolor_conf_source = mc_relax.sim_dir / mc_relax.input.get_last_conf()
        recolor_oxview_source = f"{td}/ducky.oxview"


        # # file path where you want to write your colored oxview
        recolor_write_to = f"{td}/ducky_relax_colored.oxview"

        # ------- use this if you have already run simulations -------------
        # # a .oxview file to use for base coloration information
        recolor_oxview_source = f"{td}/ducky.oxview"

        recolor_top_source = f"{td}/init.top"
        recolor_conf_source = f"{td}/init.dat"

        recolor_top_dat(
            recolor_oxview_source,
            recolor_top_source,
            recolor_conf_source,
            recolor_write_to
        )


    # def test_oxview_recolor():
    with tempfile.TemporaryDirectory() as td:
        name = "wireframe"

        patch_positions = [
            [128, 131, 130, 136],  # front
            [151, 155, 152, 160],  # bot
            [168, 163, 144, 156],  # left
            [146, 147, 149, 150],  # top
            [138, 140, 139, 141],  # right
            [142, 167, 143, 153]  # back
        ]

        wireframe_polycube = load_polycube(
            Path(__file__).parent / "test_files" / "cube_structures" / "duck_polycube.json")
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

        get_writer("flavio").set_directory(td)
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
        # insert specific sequences here if you want
        for icolor in converter.scene.particle_types().patch_colors():
            if icolor > 0:
                converter.assign_color_sequence(icolor, rand_seq(10))

        converter.assign_particles(dna_particle)
        converter.dump_monomers(name)
        converter.position_particles()
        converter.convert()
        converter.save_oxview(write_oxview_path=f"{td}/ducky.oxview", color_by_type=True)

        # create a directory to save raw
        converter.save_top_dat(f"{td}/init.top",
                               f"{td}/init.dat")

        # ------- run a monte carlo relax ----------------------
        mc_relax = Simulation(file_dir=td, sim_dir=f"{td}/relax_mc/")
        mc_relax.build()
        mc_relax.input.swap_default_input("CPU_mc_relax")
        mc_relax.input["T"] = "30C"
        mc_relax.input["steps"] = 100
        mc_relax.input["print_conf_interval"] = 10
        mc_relax.input["print_energy_every"] = 10

        mc_relax.oat.generate_force(join=True)
        mc_relax.input["box_type"] = "orthogonal"
        mc_relax.oxpy_run(subprocess=False)

        # ------- todo: mc relax? ------
        # top file
        recolor_top_source = f"{td}/relax_mc/init.top"
        recolor_conf_source = f"{td}/relax_mc/last_conf.dat"
        recolor_oxview_source = f"{td}/ducky.oxview"

        recolor_write_to = f"{td}/ducky_relax_colored.oxview"

        recolor_top_dat(
            recolor_oxview_source,
            recolor_top_source,
            recolor_conf_source,
            recolor_write_to
        )
