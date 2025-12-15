"""
This module provides data structures and utilities for managing, caching,
and combining different types of analysis data used in the pypatchy analysis pipeline.
It defines base classes and subclasses for pandas-based, object-based, and raw data.

"""


from __future__ import annotations

import itertools
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Any, Optional

import numpy as np
import pandas as pd

TIMEPOINT_KEY = "timepoint"


class PipelineDataType(Enum):
    """
    Enumeration for data types that can be used in an analysis pipeline.


    Members:
        PIPELINE_DATATYPE_RAWDATA: Raw data from trajectory.dat.

        PIPELINE_DATATYPE_OBSERVABLE: Data derived from observables.

        PIPELINE_DATATYPE_DATAFRAME: Data stored in a pandas DataFrame.

        PIPELINE_DATATYPE_GRAPH: Data represented as a list of graphs.

        PIPELINE_DATATYPE_OBJECTS: Arbitrary Python objects (e.g., graphs, simulations).

    """
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
        self._overlapping_timepoints = list(itertools.accumulate(self.overlap_data(),
                                                            func=lambda a, b: np.intersect1d(a, b)[0]))

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
        """
        Return the stored data object (to be implemented by subclasses).
        """
        pass

    @abstractmethod
    def compare_tranges(self, tr: range) -> np.array:
        """

        :param tr: range of timepoints to compare against

        :returns: a numpy array of boolean corresponding to timepoints. each element is true if the object has data at that timepoint, false otherwise.

        """
        pass

    # TODO: make abstract and force more efficient impl
    def matches_trange(self, tr: range) -> bool:
        """
        :param tr: a range of timepoints
        :returns: true if the timepoints in the data match the provided arg, false otherwise.
        """
        return self.compare_tranges(tr).all()

    @abstractmethod
    def trange(self) -> np.ndarray:
        """
        :returns: the time range of the stored data
        """
        pass

    @abstractmethod
    def cache_data(self, p: Path):
        """
        saves data to a cache file (to be implemented by subclasses)
        """
        pass

    @abstractmethod
    def load_cached_data(self, p: Path):
        """
        loads data cached in a file (to be implemented by subclasses)
        """
        pass

    @abstractmethod
    def __add__(self, other: PipelineData):
        """
        Combines this set of data with another set of data (to be implemented by subclasses).
        """
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, slice, range]):
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

    def __init__(self, data: Optional[pd.DataFrame], tr: Optional[np.ndarray]):
        """
        Constructor
        """
        # require timepoint range to be
        self._trange = tr.astype(int) if tr is not None else None
        self.data = data

    def get(self) -> pd.DataFrame:
        """
        accessor for data as pandas dataframe
        :returns: data as a pandas dataframe
        """
        return self.data

    def __getitem__(self, item: Union[int, slice]):
        """

        """
        if isinstance(item, int):
            return PDPipelineData(self.data[self.data[TIMEPOINT_KEY] == item], np.array([item]))
        elif isinstance(item, slice) or isinstance(item, range):
            timepoints = np.arange(item.start, item.stop, item.step if item.step else 1)
            filtered_data = self.data[self.data[TIMEPOINT_KEY].isin(timepoints)]
            return PDPipelineData(filtered_data, timepoints)
        else:
            raise TypeError("Item must be int slice or range")


    def compare_tranges(self, tr: Union[range, list, np.ndarray]) -> np.array:
        """
        Improved impl written by chatGPT
        Compares the range of the data stored in this object with the provided
        range object, optimized for large self._trange and tr.

        :returns: a numpy array of boolean corresponding to timepoints. each element is true if the object has data at
        the corresponding timepoint, false otherwise.
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
        :returns: an array of timepoints which are present in self's timepoint range but not in the provided timepoints
        """
        return np.array(list(set(self.trange()).difference(tr)))

    def trange(self) -> np.ndarray:
        """
        :returns: the time range of the stored data
        """
        return self._trange

    def cache_data(self, p: Path):
        """
        saves the data to a cache file using pd.HDFStore
        :param p: Path to the data cache file
        """
        with pd.HDFStore(str(p)) as f:
            f["data"] = self.data
            f["trange"] = pd.Series(self.trange())

    def load_cached_data(self, p: Path):
        """
        loads data stored in a csv file or (more likely) and hdf5 file
        :param p: Path to the data cache file
        """
        # backwards compatibility with csv storage (mistakes were made)
        if not p.exists() and Path(str(p)[:str(p).rfind(".")] + ".csv").exists():
            self.data = pd.read_csv(Path(str(p)[:str(p).rfind(".")] + ".csv"))
            self._trange = self.data[TIMEPOINT_KEY].unique()
        else:
            with pd.HDFStore(str(p)) as hdfdata:
                self.data = hdfdata["data"]
                self._trange = np.array(hdfdata["trange"])

    def __add__(self, other: PDPipelineData) -> PDPipelineData:
        """
        :param other: PDPipelineData with additional pdndas dataframe data
        :returns: a PDPipelineData object containing the data in this and other
        """
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
    """
    :param f: Path to the data cache file
    :returns: a PDPipelineData object containing the the cache file
    """
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

    def __getitem__(self, item: Union[int, slice, range]) -> ObjectPipelineData:
        if isinstance(item, int):
            return ObjectPipelineData({item: self.get()[item]})
        elif isinstance(item, slice) or isinstance(item, range):
            try:
                return ObjectPipelineData({timepoint: self.get()[timepoint]
                      for timepoint in np.arange(start=item.start, stop=item.stop, step=item.step)})
            except KeyError as e:
                raise MissingDataError(np.array([timepoint
                                    for timepoint in np.arange(start=item.start, stop=item.stop, step=item.step)
                                        if timepoint not in self.get()]),
                                       self) from e
        else:
            raise TypeError(f"Invalid type {type(item)}")


class RawPipelineData(ObjectPipelineData):
    # TODO: anything at all
    pass

    def __getitem__(self, item: Union[int, slice, range]) -> RawPipelineData:
        """ reimpl to ensure correct type
        """
        if isinstance(item, int):
            return RawPipelineData({item: self.get()[item]})
        elif isinstance(item, slice) or isinstance(item, range):
            return RawPipelineData({timepoint: self.get()[timepoint]
                                       for timepoint in np.arange(start=item.start, stop=item.stop, step=item.step)})
        else:
            raise TypeError(f"Invalid type {type(item)}")


def load_cached_object_data(_, f: Path) -> ObjectPipelineData:
    assert f.is_file()
    with f.open("rb") as datafile:
        return pickle.load(datafile)

class MissingDataError(Exception):
    """
    Exception to be thrown when some required timepoints are missing from a dataset
    """
    _missing_timepoints: np.ndarray
    _data_set: PipelineData
    other:str # additional notes, for simulation info etc

    def __init__(self, missing_timepoints: np.ndarray, data_set: PipelineData):
        self._missing_timepoints = missing_timepoints
        self._data_set = data_set
        self.other = ""

    def missing_timepoints(self) -> np.ndarray:
        return self._missing_timepoints

    def __str__(self):
        return f"Missing timepoints: {self.missing_timepoints()}.\n Data set has timepoints: {self._data_set.trange()}\n{self.other}"

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
