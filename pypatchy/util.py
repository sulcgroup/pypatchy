from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Union, Iterable, Any

from dateutil.relativedelta import relativedelta
from scipy.spatial.transform import Rotation as R
from itertools import groupby, combinations, chain
from pathlib import Path
import configparser
from colorsys import hsv_to_rgb
import numpy as np
import math

# global slurm job info cache
SLURM_JOB_CACHE: dict[int, dict[str, str]] = {}

dist = lambda a, b: np.linalg.norm(a - b)
normalize = lambda v: v / np.linalg.norm(v)

WRITE_ABS_PATHS_KEY = "absolute_paths"
MPS_KEY = "cuda_mps"

def get_local_dir() -> Path:
    return Path.home() / ".pypatchy/"


def get_input_dir() -> Path:
    return get_local_dir() / "input"


def lsin() -> list[str]:
    """
    Lists files and folders in `input` directory
    """
    return [*get_input_dir().iterdir()]


def get_output_dir() -> Path:
    return get_local_dir() / "output"


def get_log_dir() -> Path:
    return get_output_dir() / "logs"


cfg = configparser.ConfigParser()
assert (get_local_dir() / "settings.cfg").exists()
cfg.read(get_local_dir() / 'settings.cfg')


def simulation_run_dir() -> Path:
    return Path(cfg['ANALYSIS']['simulation_data_dir'])


def get_init_top_file_name() -> str:
    return cfg['ANALYSIS']['init_top_file_name']


def get_server_config() -> dict:
    return get_spec_json(cfg["SETUP"]["server_config"], "server_configs")


def is_server_slurm() -> bool:
    """
    Returns whether the server is a slurm server. Defaults to true for legacy reasons.
    """
    return "slurm_bash_flags" in get_server_config()


# TODO: slurm library? this project is experiancing mission creep
def get_slurm_bash_flags() -> dict[str, Any]:
    assert is_server_slurm(), "Trying to get slurm bash flags for a non-slurm setup!"
    return get_server_config()["slurm_bash_flags"]


def get_slurm_n_tasks() -> int:
    bashflags = get_slurm_bash_flags()
    if "n" in bashflags:
        return bashflags["n"]
    if "ntasks" in bashflags:
        return bashflags["ntasks"]
    return 1  # default value


def is_write_abs_paths() -> bool:
    return get_server_config()[WRITE_ABS_PATHS_KEY]

def is_mps() -> bool:
    return get_server_config()[MPS_KEY]

def get_param_set(filename) -> dict:
    return get_spec_json(filename, "input_files")


def get_spec_json(name, folder) -> dict:
    try:
        with open(f"{get_local_dir()}/spec_files/{folder}/{name}.json") as f:
            return json.load(f)
    except IOError as e:
        print(f"No file named {name} in {get_local_dir() / 'spec_files' / folder}!")


def is_sorted(target: Iterable[int]) -> bool:
    if not isinstance(target, list):
        target = list(target)
    return all([target[i - 1] < target[i] < target[i + 1] for i in range(1, len(target) - 1)])


class BadSimulationDirException(Exception):
    def __init__(self, p):
        self.p = p

    def __str__(self) -> str:
        return f"Path {self.p} does not have expected format for the patch to a patchy particles trial simulation"


def selectColor(number: int, saturation=50, value=65, fmt="hex") -> str:
    hue = number * 137.508  # use golden angle approximation
    if fmt == "hsv":
        return f"hsv({hue},{saturation}%,{value}%)"
    else:
        r, g, b = hsv_to_rgb(hue / 255, saturation / 100, value / 100)
        if fmt == "rgb":
            return f"rgb({r},{g},{b})"
        else:
            return f"#{hex(int(255 * r))[-2:]}{hex(int(255 * g))[-2:]}{hex(int(255 * b))[-2:]}"


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
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


def rotAroundAxis(patchPos, axis, angle):
    r = R.from_rotvec(angle * axis)
    return r.apply(patchPos)


def to_xyz(vector: np.ndarray) -> dict[str: int]:
    return {k: int(v) for k, v in zip(["x", "y", "z"], vector)}


def from_xyz(d: dict[str: int]) -> np.ndarray:
    return np.array([d[k] if k in d else 0 for k in ["x", "y", "z"]])


# TODO: test if getRotations and enumerateRotations have the same order!!!!

def getRotations(ndim=3) -> list[np.ndarray]:
    """
    Returns a list of rotation matrices for all possible

    """
    rots = [
        # 2D rotations
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
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


def enumerateRotations() -> dict[int: dict[int: int]]:
    """
    Returns:
        a list of mappings of direction indexes, representing all possible octahedral rotational
        symmetries (google it)

    """
    return {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        1: {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4},
        2: {0: 0, 1: 1, 2: 4, 3: 5, 4: 3, 5: 2},
        3: {0: 0, 1: 1, 2: 5, 3: 4, 4: 2, 5: 3},
        4: {0: 1, 1: 0, 2: 2, 3: 3, 4: 5, 5: 4},
        5: {0: 1, 1: 0, 2: 3, 3: 2, 4: 4, 5: 5},
        6: {0: 1, 1: 0, 2: 4, 3: 5, 4: 2, 5: 3},
        7: {0: 1, 1: 0, 2: 5, 3: 4, 4: 3, 5: 2},
        8: {0: 2, 1: 3, 2: 0, 3: 1, 4: 5, 5: 4},
        9: {0: 2, 1: 3, 2: 1, 3: 0, 4: 4, 5: 5},
        10: {0: 2, 1: 3, 2: 4, 3: 5, 4: 0, 5: 1},
        11: {0: 2, 1: 3, 2: 5, 3: 4, 4: 1, 5: 0},
        12: {0: 3, 1: 2, 2: 0, 3: 1, 4: 4, 5: 5},
        13: {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4},
        14: {0: 3, 1: 2, 2: 4, 3: 5, 4: 1, 5: 0},
        15: {0: 3, 1: 2, 2: 5, 3: 4, 4: 0, 5: 1},
        16: {0: 4, 1: 5, 2: 0, 3: 1, 4: 2, 5: 3},
        17: {0: 4, 1: 5, 2: 1, 3: 0, 4: 3, 5: 2},
        18: {0: 4, 1: 5, 2: 2, 3: 3, 4: 1, 5: 0},
        19: {0: 4, 1: 5, 2: 3, 3: 2, 4: 0, 5: 1},
        20: {0: 5, 1: 4, 2: 0, 3: 1, 4: 3, 5: 2},
        21: {0: 5, 1: 4, 2: 1, 3: 0, 4: 2, 5: 3},
        22: {0: 5, 1: 4, 2: 2, 3: 3, 4: 0, 5: 1},
        23: {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    }


# there may be a faster algorithm for this but frankly I don't care enough to find it
def rotidx(r: dict[int, dict[int, int]]) -> int:
    for key, value in enumerateRotations().items():
        if all(value[x] == r[x] for x in r):
            return key
    return -1  # invalid rotation dict


def getSignedAngle(v1: np.ndarray,
                   v2: np.ndarray,
                   axis: np.ndarray) -> float:
    """
    nightmare code
    computes the angle from vector v1 to vector v2 around axis... probably
    """
    s = np.cross(v1, v2)
    c = v1.dot(v2)
    a = np.arctan2(np.linalg.norm(s), c)
    if not np.array_equal(s, axis):
        a *= -1
    return a


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# function written by ChatGPT
def inverse_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion norm is zero, can't calculate inverse.")
    q_conjugate = np.array([w, -x, -y, -z])
    q_inverse = q_conjugate / norm ** 2
    return q_inverse


# shamelessly pilfered from https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def is_slurm_job() -> bool:
    return os.environ.get("SLURM_JOB_ID") is not None


def halfway_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Returns a unit vector halfway between two unit vectors a and b. """
    # Normalize a and b to be sure they're unit vectors
    a = normalize(a)
    b = normalize(b)

    # Check if a and b are opposite
    if np.allclose(a, -b):
        # Find an arbitrary vector not parallel to a
        if a[0] == 0 and a[1] == 0:
            c = np.array([0, 1, 0])
        else:
            c = np.array([-a[1], a[0], 0])
        return normalize(c)

    # Sum of a and b
    sum_vector = a + b
    return normalize(sum_vector)


def random_unit_vector() -> np.ndarray:
    """
    Generate a random unit vector in 3-space.
    WARNING: code came out of chatGPT!!!
    """
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = costheta

    return np.array([x, y, z])


def append_to_file_name(fn: str, extra: str) -> str:
    """
    appends an additonal string to a file name, before the extension
    """
    if fn.find(".") != -1:
        fn_pre = fn[:fn.find(".")]
        ext = fn[fn.find("."):]
        return fn_pre + "_" + extra + ext
    else:
        return fn + "_" + extra


# https://docs.python.org/3/library/itertools.html
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


PATCHY_FILE_FORMAT_KEY = "patchy_format"
