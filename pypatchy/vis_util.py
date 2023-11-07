import matplotlib.colors


def get_particle_color(ptypeidx: int):
    """
    returns an rgb color consistant with the usage in polycubes
    """
    hue = ptypeidx * 137.508
    return matplotlib.colors.hsv_to_rgb(((hue % 360) / 360, .5, .5))
