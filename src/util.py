import json
from os import path
import configparser
from colorsys import hsv_to_rgb
import numpy as np
import math


def get_root():
    return path.dirname(__file__)[:__file__.rfind("pypatchy") + len("pypatchy")]


cfg = configparser.ConfigParser()
cfg.read(path.join(get_root(), 'settings.cfg'))


def sims_root():
    return cfg['ANALYSIS']['simulation_data_dir']


def get_sample_every():
    return int(cfg['ANALYSIS']['sample_every'])


def get_cluster_file_name():
    return cfg['ANALYSIS']['cluster_file']


def get_export_setting_file_name():
    return cfg['ANALYSIS']['export_setting_file_name']


def get_init_top_file_name():
    return cfg['ANALYSIS']['init_top_file_name']


def get_analysis_params_file_name():
    return cfg['ANALYSIS']['analysis_params_file_name']


def get_server_config():
    return get_spec_json(cfg["SETUP"]["server_config"], "server_configs")


def get_param_set(filename):
    return get_spec_json(filename, "input_files")


def get_spec_json(name, folder):
    with open(f"{get_root()}/spec_files/{folder}/{name}.json") as f:
        return json.load(f)


class BadSimulationDirException(Exception):
    def __init__(self, p):
        self.p = p

    def __str__(self):
        return f"Path {self.p} does not have expected format for the patch to a patchy particles trial simulation"


def selectColor(number, saturation=50, value=65, fmt="hex"):
    hue = number * 137.508;  # use golden angle approximation
    if fmt == "hsv":
        return f"hsv({hue},{saturation}%,{value}%)";
    else:
        r, g, b = hsv_to_rgb(hue / 255, saturation / 100, value / 100)
        if fmt == "rgb":
            return f"rgb({r},{g},{b})"
        else:
            return f"#{hex(int(255 * r))[-2:]}{hex(int(255 * g))[-2:]}{hex(int(255 * b))[-2:]}"


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def to_xyz(vector):
    return {k: int(v) for k, v in zip(["x", "y", "z"], vector)}


def from_xyz(d):
    return np.array([d[k] for k in ["x", "y", "z"]])


def getRotations(ndim=3):
    rots = [
        # 2D rotations
        #np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    ]
    if ndim > 2:
        rots += [
            # 3D rotations
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
            np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
            np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
            np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
            np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
            np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
            np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        ]
    return rots


def getSignedAngle(v1, v2, axis):
    s = np.cross(v1, v2)
    c = v1.dot(v2)
    a = np.arctan2(np.linalg.norm(s), c)
    if not np.array_equal(s, axis):
        a *= -1
    return a

# function written by ChatGPT
def inverse_quaternion(q):
    w, x, y, z = q
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion norm is zero, can't calculate inverse.")
    q_conjugate = np.array([w, -x, -y, -z])
    q_inverse = q_conjugate / norm**2
    return q_inverse
