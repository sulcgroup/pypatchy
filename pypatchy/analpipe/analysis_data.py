from __future__ import annotations

import itertools
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd

TIMEPOINT_KEY = "timepoint"


class PipelineDataType(Enum):
    # raw data from trajectory.dat
    PIPELINE_DATATYPE_RAWDATA = 0
    # data from an observable
    PIPELINE_DATATYPE_OBSERVABLE = 1
    # data from a pandas dataframe
    PIPELINE_DATATYPE_DATAFRAME = 2
    # list of graphs
    PIPELINE_DATATYPE_GRAPH = 3
    # objects
    PIPELINE_DATATYPE_OBJECTS = 4


class OverlappingDataError(Exception):
    """
    Exception class for when you try to merge data with overlapping timepoints that conflict
    eg one PipelineData object says we have datum A at timepoint t, other one says we have datum B at timepoint t
    """

    _overlapping_data: tuple[PipelineData]

    def __init__(self, *args: PipelineData):
        self._overlapping_data = args
        self._overlapping_timepoints = itertools.accumulate(self.overlap_data(),
                                                            func=lambda a, b: np.intersect1d(a, b)[0])

    def overlap_data(self):
        return self._overlapping_data

    def overlapping_timepoints(self):
        return self._overlapping_timepoints

    def __str__(self):
        return f"Overlap between data! Overlapping timepoints {self.overlapping_timepoints()}"


class PipelineData(ABC):
    """
    Wrapper base-class for analysis data
    """

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def compare_tranges(self, tr: range) -> np.array:
        pass

    # TODO: make abstract and force more efficient impl
    def matches_trange(self, tr: range) -> bool:
        return self.compare_tranges(tr).all()

    @abstractmethod
    def trange(self) -> np.ndarray:
        pass

    @abstractmethod
    def cache_data(self, p: Path):
        pass

    @abstractmethod
    def load_cached_data(self, p: Path):
        pass

    @abstractmethod
    def __add__(self, other: PipelineData):
        pass


class PDPipelineData(PipelineData):
    """
    Data stored in a pandas dataframe
    """
    # the time range from begin of data to end
    # store in np.ndarray to handle missing data at timepoints
    # _trange array length should match length of unique timepoints
    _trange: np.ndarray

    data: pd.DataFrame

    def __init__(self, data: Union[None, pd.DataFrame], tr: Union[None, np.ndarray]):
        self._trange = tr
        self.data = data

    def get(self) -> pd.DataFrame:
        return self.data

    def compare_tranges(self, tr: Union[range, list, np.ndarray]) -> np.array:
        """
        Improved impl written by chatGPT
        Compares the range of the data stored in this object with the provided
        range object, optimized for large self._trange and tr.

        Return:
            a numpy array of booleans with length len(tr)
        """
        # Convert self._trange to a set for O(1) lookup if not already a set
        # This is beneficial if self._trange is used repeatedly for comparison
        if not isinstance(self._trange, set):
            _trange_set = set(self._trange)
        else:
            _trange_set = self._trange

        # Use vectorized operation for numpy arrays, else use a list comprehension
        if isinstance(tr, np.ndarray):
            # For numpy arrays, we can use np.isin for efficient element-wise comparison
            return np.isin(tr, _trange_set)
        else:
            # For lists or ranges, use a list comprehension with the set for fast lookup
            return np.array([t in _trange_set for t in tr])

    def missing_timepoints(self, tr: Union[range, list[Union[int, float]], np.ndarray]):
        """
        Compares the range of these data with the provided timepoints
        Returns an array of timepoints which are present in self's timepoint range but not in the
        provided timepoits
        """
        return np.array(list(set(self.trange()).difference(tr)))

    def trange(self) -> np.ndarray:
        return self._trange

    def cache_data(self, p: Path):
        with pd.HDFStore(str(p)) as f:
            f["data"] = self.data
            f["trange"] = pd.Series(self.trange())

    def load_cached_data(self, p: Path):
        # backwards compatibility with csv storage (mistakes were made)
        if not p.exists() and Path(str(p)[:str(p).rfind(".")] + ".csv").exists():
            self.data = pd.read_csv(Path(str(p)[:str(p).rfind(".")] + ".csv"))
            self._trange = self.data[TIMEPOINT_KEY].unique()
        else:
            with pd.HDFStore(str(p)) as hdfdata:
                self.data = hdfdata["data"]
                self._trange = np.array(hdfdata["trange"])

    def __add__(self, other: PDPipelineData) -> PDPipelineData:
        overlap, _, _ = np.intersect1d(self.trange(), other.trange())
        # if no overlap, we're cool
        if overlap.size == 0:
            return PDPipelineData(pd.concat([self.get(), other.get()], ignore_index=True),
                                  np.concatenate([self.trange(), other.trange()]))
        else:
            self_overlap = self.get()[self.get()[TIMEPOINT_KEY].isin(overlap)]
            other_overlap = other.get()[other.get()[TIMEPOINT_KEY].isin(overlap)]
            if self_overlap.equals(other_overlap):
                return PDPipelineData(pd.concat([self.get(), other.get()], ignore_index=True),
                                      np.concatenate([self.trange(), other.trange()]))
            else:
                raise OverlappingDataError(self, other)


def load_cached_pd_data(_, f: Path) -> PDPipelineData:
    assert f.is_file()
    data = PDPipelineData(None, None)
    data.load_cached_data(f)
    return data


# class ObservablePipelineData:
#     data: Any
#
#     def __init__(self, data, tr):
#         super(ObservablePipelineData, self).__init__(tr)
#         self.data = data
#
#     def get(self):
#         return self.data
#

class ObjectPipelineData(PipelineData):
    """
    Data composed of lists of graphs at each timepoint
    """

    # keys are timepoints, each value is
    data: dict[int, list[Any]]

    def __init__(self, data):
        assert isinstance(data, dict), "Invalid data arguement for ObjectPipelineData"
        self.data = data

    def get(self) -> dict[int, list[Any]]:
        return self.data

    def compare_tranges(self, tr: range) -> np.array:
        return np.array([t in self.data.keys() for t in tr])

    def trange(self) -> np.ndarray:
        return np.array(list(self.data.keys()))

    def cache_data(self, p: Path):
        with open(p, 'wb') as f:
            pickle.dump(self.data, f)

    def load_cached_data(self, p: Path):
        assert p.is_file()
        with open(p, 'rb') as f:
            self.data = pickle.load(f)

    def __add__(self, other: ObjectPipelineData) -> ObjectPipelineData:
        overlap = np.intersect1d(self.trange(), other.trange())
        # if no overlap, we're cool
        if len(overlap) == 0:
            return ObjectPipelineData({**self.get(), **other.get()})
        else:
            if all([self.get()[key] == other.get()[key] for key in overlap]):
                return ObjectPipelineData({**self.get(), **other.get()})
            else:
                raise OverlappingDataError(self, other)


class RawPipelineData(ObjectPipelineData):
    # TODO: anything at all
    pass


def load_cached_object_data(_, f: Path) -> ObjectPipelineData:
    assert f.is_file()
    with f.open("rb") as datafile:
        return pickle.load(datafile)


class MissingCommonDataError(Exception):
    """
    Exception to be thrown when two datasets are required for some operation but there's no overlap
    between the tranges of the datasets
    """
    _data_sets: tuple[PipelineData]

    def __init__(self, *args: PipelineData):
        self._data_sets = args

    def __str__(self):
        return "No timepoints overlapping between datasets!"
