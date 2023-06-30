from __future__ import annotations

from ..util import get_spec_json


# TODO: integrate with oxpy.core.observables stuff

class PatchySimObservable:
    def __init__(self, **kwargs):
        # for all param meanings, see https://lorenzo-rovigatti.github.io/oxDNA/observables.html
        self.name: str = kwargs["name"]

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
        # TODO: if using more params, update this
        return {
            "name": self.name,
            "print_every": self.print_every,
            "cols": self.cols
        }


def observable_from_file(obs_file_name: str) -> PatchySimObservable:
    return PatchySimObservable(**get_spec_json(obs_file_name, "observables"))

