# executive decision: i have decided to put this class in its own file
# since (unlike the classes in tlm_data.py) it isn't a reimplementation of a pybind11 class
import pickle
from pathlib import Path
from typing import Union, BinaryIO

import libtlm
import json

from pypatchy.polycubeutil import tlm_data
from pypatchy.polycubeutil.tlm_data import TLMHistoryRecord


class TLMRecords:
    tlm_settings: libtlm.TLMParameters
    records: dict[int, TLMHistoryRecord]

    def __init__(self, fp: Union[Path, BinaryIO]):
        self.tlm_settings = None
        self.records = dict()
        if isinstance(fp, BinaryIO):
            self.__read_binary(fp)
        else:
            assert isinstance(fp, Path)
            assert fp.is_file()
            if fp.suffix == ".pickle":
                with fp.open("rb") as f:
                    self.__read_binary(f)
            else:
                assert fp.suffix == ".json", f"Illegal record file suffix {fp.suffix}"
                with fp.open("r") as f:
                    data = json.load(f)
                    records = [tlm_data.TLMHistoryRecord(**record) for record in data["records"]]
                    assert len(records) > 0, "Trying to read json file with no records!!"
                    self.records = {record.stepCount(): record for record in records}
                    self.tlm_settings = data["settings"]

    def __read_binary(self, stream: BinaryIO):
        try:
            self.tlm_settings = pickle.load(stream)
            for i in range(self.tlm_settings.num_steps // self.tlm_settings.history_interval):
                record: tlm_data.TLMHistoryRecord = pickle.load(stream)
                if record is not None:
                    self.records[record.stepCount()] = record

        except EOFError as e:
            raise e

    def export(self, fp: Path):
        assert fp.suffix == ".pickle", "Attempting to dump to non-pickle file"
        assert self.tlm_settings is not None
        with fp.open("wb") as f:
            pickle.dump(self.tlm_settings, f)
            for record in self.records.values():
                pickle.dump(record, f)