import sys
from .plpatchy import *
import shutil


# usage: flaviantolorenzian.py [patch file] [particle file] [top file]

def convert_flavian_to_lorenzian(_, patches_file, particles_file, top_file):
    patches = load_patches(patches_file)
    particles = load_particles(particles_file, patches)

    # read num particles_
    with open(top_file, 'r') as f:
        nParticles, nParticleTypes = f.readline().split()
        particle_ids = list(map(lambda x: int(x), f.readline().split()))

    # back up file
    shutil.copyfile(top_file, top_file + ".bak")

    with open(top_file, 'w+') as f:
        f.write(f"{nParticles} {nParticleTypes}\n")
        f.writelines([particle.export_to_lorenzian_patchy_str(particle_ids.count(particle.type())) + "\n"
                      for particle in particles])

    export_interaction_matrix(patches)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        convert_flavian_to_lorenzian(*sys.argv)
