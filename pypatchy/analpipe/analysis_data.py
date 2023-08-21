import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Any
import networkx as nx

import numpy as np
import pandas as pd

TIMEPOINT_KEY = "timepoint"


class PipelineDataType(Enum):
    # raw data from trajectory.dat - currently not used
    PIPELINE_DATATYPE_RAWDATA = 0
    # data from an observable
    PIPELINE_DATATYPE_OBSERVABLE = 1
    # data from a pandas dataframe
    PIPELINE_DATATYPE_DATAFRAME = 2
    # list of graphs
    PIPELINE_DATATYPE_GRAPH = 3


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
        Compares the range of the data stored in this object with the provided
        range object
        Return
            a numpy array of booleans with length len(tr)
        """
        return np.array([t in self._trange for t in tr])

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


def load_cached_pd_data(_, f: Path) -> PDPipelineData:
    assert f.is_file()
    data = PDPipelineData(None, None)
    data.load_cached_data(f)
    return data


class RawPipelineData:
    """
    Raw oxDNA data (eg not from an Observable). todo: flesh out
    """
    data: Any  # TODO: flesh out
    _trange = np.ndarray

    def __init__(self, data, tr):
        self.data = data
        self._trange = tr

    def get(self):
        return self.data

    def trange(self) -> np.ndarray:
        return self._trange

    def compare_tranges(self, tr: range) -> np.array:
        return all(t in self.trange() for t in tr)


#
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

class GraphPipelineData(PipelineData):
    """
    Data composed of lists of graphs at each timepoint
    """

    # keys are timepoints, each value is
    data: dict[int, list[nx.Graph]]

    def __init__(self, data):
        self.data = data

    def get(self) -> dict[int, list[nx.Graph]]:
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


def load_cached_graph_data(_, f: Path) -> GraphPipelineData:
    assert f.is_file()
    with f.open("rb") as datafile:
        return pickle.load(datafile)
