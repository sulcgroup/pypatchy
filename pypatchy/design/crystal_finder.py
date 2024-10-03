import itertools
from dataclasses import dataclass
from typing import Union

import pandas as pd

import libtlm
import numpy as np
from scipy.optimize import curve_fit

from pypatchy.polycubeutil.polycubesRule import PolycubesRule


# Define the sigmoid function
def sigmoid(x, L, k, x0, b):
    return -L / (1 + np.exp(-k * (x - x0))) + b


def tanh(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    return L * np.tanh(k * (x - x0)) + b


@dataclass
class CrystallizationTestParams:
    rule: PolycubesRule
    n_replicas: int
    data_point_interval: int
    n_unit_cells: int
    cube_type_counts: list[int]
    sim_n_steps: int
    torsion: bool = True

    total_type_counts = property(lambda self: [n * self.n_unit_cells for n in self.cube_type_counts])

# todo: make this a class so it can properly store temperature data
def find_crystal_temperature(tinter: float,
                             tmin: float,
                             tmax: float,
                             params: CrystallizationTestParams) -> tuple[
    Union[None, list[list[libtlm.TLMHistoryRecord]]], float]:
    assert len(params.cube_type_counts) == len(params.rule)
    T = (tmax + tmin) / 2  # find midpoint
    print(f"------------------------ Testing crystallization at temperature {T} -------------------")
    # run simulations
    polycube_results: list[list[libtlm.TLMHistoryRecord]] = libtlm.runSimulations(
        libtlm.TLMParameters(
            params.torsion,  # torsion
            True,  # deplete types
            T,  # temperature
            0.1,  # density - TODO: customize?
            str(params.rule),
            params.total_type_counts,  # type counts
            params.sim_n_steps,  # steps
            params.data_point_interval  # data point interval
        ),
        params.n_replicas, # number of replicase
        int(1e5) # interval to print energy to console
    )

    # convert results to pandas dataframe
    df = pd.DataFrame(itertools.chain.from_iterable([
        [{
            "sim": i,
            "tidx": tidx,
            "step": datapoint.stepCount(),
            "energy": datapoint.energy()
        } for tidx, datapoint in enumerate(sim_records) if datapoint.energy()]
        for i, sim_records in enumerate(polycube_results)
    ]))
    df.dropna(inplace=True)  # TODO: make not problem

    print(f"Computing best fit for {len(df['step'].unique())} steps and {len(polycube_results)} replicas.")
    # interpret data
    fit_results = []
    result_is_melt = False
    for sim in df['sim'].unique():
        sim_data = df[df['sim'] == sim]
        x_data = sim_data['step']
        y_data = sim_data['energy']

        # Initial guesses for L, k, and x0
        # Improve initial guesses
        L_guess = 100
        k_guess = - 1e-3
        x0_guess = x_data.mean()
        b_guess = 0.
        initial_guesses = (L_guess, k_guess, x0_guess, b_guess)

        # Perform the curve fitting
        try:
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=initial_guesses, maxfev=10000)
            # if the curve fit algorithm can't find ideal curve, it throws a runtimer error
        except RuntimeError as e:
            # if the curve fitter can't fit the data that means the thing is probably melted
            if str(e).startswith("RuntimeError: Optimal parameters not found:"):
                result_is_melt = True
                print("Could not fit data to signoidal curve. Conclusion: Melt")
                break
            else:
                raise e


        # Plot each fit
        # x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = sigmoid(x_data, *popt)

        ss_res = np.sum((y_data - y_fit) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        fit_results.append({
            'Sim': sim,
            'L': popt[0],
            'k': popt[1],
            'x0': popt[2],
            'b': popt[3],
            'R_squared': r_squared
        })
    fit_results = pd.DataFrame(fit_results)

    # TODO: we can do better?
    # todo: quantitiative measure of crystal-ness

    if result_is_melt or is_melt(fit_results):
        # if we've melted, lower temperature
        print(f"Rule melts at T={T}")
        tmax = T
    elif is_aggregate(fit_results):
        # if aggregat, raise temperature
        print(f"Rule aggregates at T={T}")
        tmin = T
    else:
        assert is_crystal(fit_results)
        print(f"Crystal at T={T}!")
        return polycube_results, T
    if (tmax - tmin) < tinter: # limit recursion depth
        print(f"Could not find crystallization temperature! Best temperature was T={T}")
        return polycube_results, T
    return find_crystal_temperature(tinter, tmin, tmax, params)


def is_aggregate(curve_fits: pd.DataFrame) -> bool:
    # if the sigmoid best fit has a midpoint before the start
    # that means it started forming immediately & thus aggregated
    # warning: melted fits will also do this!
    return curve_fits["x0"].mean() < 0


def is_melt(curve_fits: pd.DataFrame) -> bool:
    # if L is negligable that means melt
    fit_results_stats = curve_fits.describe()
    # return abs(curve_fits["L"].mean() / curve_fits["b"].mean()) < 0.01
    # alternatively: if curve fit is garbage, that means melt (energies are randomish)
    # historically all real melt curves I've observed have had R2 > 0.99
    return (curve_fits["R_squared"] < 0.95).any() and not is_aggregate(curve_fits)


def is_crystal(curve_fits: pd.DataFrame) -> bool:
    return not is_aggregate(curve_fits) and not is_melt(curve_fits)
