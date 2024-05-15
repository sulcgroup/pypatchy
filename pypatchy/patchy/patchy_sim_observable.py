from __future__ import annotations

from typing import IO, Union, Any

from .stage import Stage
from ..util import get_spec_json


# DEPRECATED
# TODOL integrate with ipy_oxdna
class PatchySimObservable:
    """
    A class describing an oxDNA observable that can be added to a patchy particle interaction
    """
    observable_name: str
    file_name: str
    is_linear_tile: bool
    log_ppc: float  # ??
    log_n0: float  # ??
    log_fact: float  # ??
    print_every: int
    start_observe_stepnum: int
    stop_observe_stepnum: Union[int, None]
    only_write_last: bool
    update_name_with_time: bool
    cols: Any  # ????

    def __init__(self, **kwargs):
        # for all param meanings, see https://lorenzo-rovigatti.github.io/oxDNA/observables.html
        self.observable_name = kwargs["observable_name"]
        self.file_name: str = kwargs["name"]

        # optional params for nonlinear time sampling - not currently used
        self.is_linear_time = kwargs["linear"] if "linear" in kwargs else True
        if not self.is_linear_time:
            self.log_ppc = kwargs["log_ppc"]
            self.log_n0 = kwargs["log_n0"]
            self.log_fact = kwargs["log_fact"]
        else:
            self.print_every = kwargs["print_every"]

        # more optional params - currently unused
        self.start_observe_stepnum = kwargs["start_from"] if "start_from" in kwargs else 0
        self.stop_observe_stepnum = kwargs["stop_at"] if "stop_at" in kwargs else None
        self.only_write_last = kwargs["only_last"] if "only_last" in kwargs else False
        self.update_name_with_time = kwargs["update_name_with_time"] if "update_name_with_name" in kwargs else False

        self.cols = kwargs["cols"]  # abandon hope all ye who enter here

    def to_dict(self) -> dict:
        """
        Returns the observable as a python dict
        """
        # TODO: if using more params, update this
        return {
            "name": self.file_name,
            "print_every": self.print_every,
            "cols": self.cols
        }

    def write_input(self, input_file: IO, i: int, stage: Stage, analysis: bool = False):
        """
        Writes the observable to an oxDNA input file

        Args:
            input_file: an io object for file writing, directing to an oxdna input file
            i: the index of this observable (important because of how oxdna reads the input file)
            stage: the stage that we're writing an input file for
            analysis: if true, this input file is being written for `DNAanalysis` and not `oxdna`

        """
        if analysis:
            input_file.write(f"analysis_data_output_{i + 1} = " + "{\n")
        else:
            input_file.write(f"data_output_{i + 1} = " + "{\n")
        if stage.idx():
            input_file.write(f"\tname = {stage.name()}_{self.file_name}\n")
        else:
            input_file.write(f"\tname = {self.file_name}\n")
        # TODO TODO TODO!!! MAKE ABS PATHS WORK HERE

        input_file.write(f"\tprint_every = {self.print_every}\n")  # TODO: configure to deal with nonlinear time
        if self.start_observe_stepnum > 0:
            input_file.write(f"\tstart_from = {self.start_observe_stepnum}\n")
        if self.stop_observe_stepnum is not None:
            input_file.write(f"\tstop_at = {self.stop_observe_stepnum}\n")
        if self.only_write_last:
            input_file.write("\tonly_last = 1\n")
        if self.update_name_with_time:
            input_file.write("\tupdate_name_with_time = 1\n")

        for i_col, col in enumerate(self.cols):
            input_file.write(f"\tcol_{i_col + 1} = " + "{\n")
            for key, value in col.items():
                input_file.write(f"\t\t{key} = {value}\n")
            input_file.write("\t}\n")
        input_file.write("}\n")

    def write_input_dict(self,
                         input_dict: dict[str, Any],
                         i: int,
                         analysis: bool = False):
        """
        Edits an input dict to include

        Args:
            input_file: an io object for file writing, directing to an oxdna input file
            i: the index of this observable (important because of how oxdna reads the input file)
            stage: the stage that we're writing an input file for
            analysis: if true, this input file is being written for `DNAanalysis` and not `oxdna`

        """
        if analysis:
            key = f"analysis_data_output_{i + 1}"
        else:
            key = f"data_output_{i + 1}"
        input_dict[key] = {
            "name": self.file_name, # TODO: absoulte file pathing!!!
            "print_every": self.print_every,
            "stop_at": self.stop_observe_stepnum,
            "start_from": self.start_observe_stepnum,
            "only_last": 1,
            "update_name_with_time": 1,
            **{
                f"col_{i_col + 1}": col for i_col, col in enumerate(self.cols)
            }
        }
        # TODO TODO TODO!!! MAKE ABS PATHS WORK HERE


def observable_from_file(obs_file_name: str) -> PatchySimObservable:
    """
    Constructs a new PatchySimObservable object from a json file, assumed to be located in
    ~/.pypatchy/spec_files/observables

    Args:
        obs_file_name: a file name for the observable file

    Returns:
        a PatchySimObservable object made from the provided file

    """
    # standardize input
    if obs_file_name.endswith(".json"):
        obs_file_name = obs_file_name[:obs_file_name.rfind(".")]
    return PatchySimObservable(observable_name=obs_file_name, **get_spec_json(obs_file_name, "observables"))
