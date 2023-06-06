import math
import sys
from patchy.plpatchy import *
from scipy.spatial.transform import Rotation as R

def convert_multidentate(particles, dental_radius, num_teeth, followSurf=False):
    new_particles = [None for _ in particles]
    patch_counter = 0
    new_patches = []
    for i_particle, particle in enumerate(particles):
        new_particle_patches = []
        for patch in particle.get_patches():
            teeth = [None for _ in range(num_teeth)]
            # "normalize" color by making the lowest color 0
            is_color_neg = patch.color() < 0
            colornorm = abs(patch.color()) - 21
            for tooth in range(num_teeth):
                # start with color
                c = colornorm * num_teeth + tooth + 21
                # grab patch position, a1, a2
                position = np.copy(patch.position())
                a1 = np.copy(patch.a1())
                a2 = np.copy(patch.a2())
                # theta is the angle of the tooth within the patch
                theta = tooth / num_teeth * 2 * math.pi
                if is_color_neg:
                    # opposite-color patches have to be rotated opposite directions
                    # b/c mirroring
                    theta *= -1
                    # set color sign
                    c *= -1
                r = R.identity()
                if followSurf:
                    # phi is the angle of the tooth from the center of the patch
                    psi = dental_radius / particle.radius()
                    psi_axis = np.cross(a1, a2)  # axis orthogonal to patch direction and orientation
                    # get rotation
                    r = R.from_matrix(rotation_matrix(psi_axis, psi))
                else:
                    # move tooth position out of center
                    position += a2 * dental_radius
                r = r * R.from_matrix(rotation_matrix(a1, theta))
                position = r.apply(position)
                a1 = r.apply(a1)
                # using torsional multidentate patches is HIGHLY discouraged but
                # this functionality is included for compatibility reasons
                a2 = r.apply(a2)
                teeth[tooth] = Patch(patch_counter, c, position, a1, a2, 1.0/num_teeth)
                patch_counter += 1
            # add all teeth
            new_particle_patches += teeth
        new_particles[i_particle] = PLPatchyParticle(type_id=particle.type(), index_=i_particle, radius=particle.radius())
        new_particles[i_particle].set_patches(new_particle_patches)
        new_patches += new_particle_patches
    return [new_particles, new_patches]


def convert_udt_files_to_mdt(_, patches_file, particles_file, dental_radius="0.5", num_teeth="4", follow_surf="false"):
    dental_radius = float(dental_radius)
    num_teeth = int(num_teeth)
    follow_surf = follow_surf.lower() == "true"
    patches = load_patches(patches_file)
    particles = load_particles(particles_file, patches)
    [new_particles, new_patches] = convert_multidentate(particles, dental_radius, num_teeth, follow_surf)

    new_particles_fn = f"{particles_file[:particles_file.rfind('.')]}_MDt.txt"
    new_patches_fn = f"{patches_file[:patches_file.rfind('.')]}_MDt.txt"
    with open(new_particles_fn, 'w') as f:
        for p in new_particles:
            f.write(p.save_type_to_string())

    with open(new_patches_fn, 'w') as f:
        for p in new_patches:
            f.write(p.save_to_string())

# usage: UDtoMDt [patch file] [particle file] [ [dental radius] [num teeth] [follow surface]]
if __name__ == "__main__":
    if len(sys.argv) > 2:
        convert_udt_files_to_mdt(*sys.argv)
