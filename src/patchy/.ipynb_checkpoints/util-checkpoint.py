import json
from os import path
import configparser
from colorsys import hsv_to_rgb


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
