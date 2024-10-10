#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:16:24 2019
@author: joakim
"""
import argparse
import datetime
import json
import os
import sys
from multiprocessing import Pool

from pypatchy.util import get_input_dir

from pypatchy.design.sat.solution import SATSolution
from pypatchy.design.solve_utils import *
from pypatchy.design.sat.design_rule import Polysat
from pypatchy.design.solve_params import *


def smartEnumerate(xMax, yMax):
    return sorted(
        ((x, y) for x in range(1, xMax + 1) for y in range(1, yMax + 1)),
        key=lambda i: sum(i)
    )


# def fitsTopologies(topologies, rule, nChecks=32):
#     deterministic = True
#     bounded = True
#     top_counts = [0 for _ in topologies]
#     while nChecks > 0:
#         model = libpolycubes.getCoords(rule)
#         if model.shape[1] == 100:
#             bounded = False
#             deterministic = False
#         else:
#             idx = index_in(topologies, model)
#             if idx == -1:
#                 deterministic = False
#             else:
#                 top_counts[idx] += 1
#         nChecks -= 1
#     return all([i > 0 for i in top_counts]), top_counts, bounded, deterministic
#
#
# def index_in(topologies, model):
#     pass

# def gen_params(mins, maxs):
#     assert all(key in mins for key in maxs)
#     assert all(key in maxs for key in mins)
#     assert all(mins[key] <= maxs[key] for key in mins)
#     assert all(isinstance(mins[key], int) and isinstance(maxs[key], int) for key in mins)
#     keys = list(mins.keys())
#     ranges = [list(range(mins[key], maxs[key] + 1)) for key in keys]
#     prod = list(itertools.product(*ranges))
#     paramset = [{key: p[i] for i, key in enumerate(keys)} for p in prod]
#     paramset.sort(key=lambda x:np.prod([x[k] for k in x]))
#     return paramset
#
#
# def findSolutionsInRange(solveSpecPath, mins, maxs):
#     assert all(key in mins for key in maxs)
#     assert all(key in maxs for key in mins)
#     assert all([mins[key] == maxs[key] for key in mins if not isinstance(mins[key], int)])
#     string_params = {key: mins[key] for key in mins if isinstance(mins[key], str)}
#     paramsmap = gen_params({k:mins[k]for k in mins if isinstance(mins[k], int)}, {k: maxs[k] for k in maxs if isinstance(maxs[k], int)})
#     for paramset in paramsmap:
#         all_params = {**paramset, **string_params}
#         log(f"Trying to solve with parameters {' ,'.join([f'{key}: {all_params[key]}' for key in all_params])}.")
#         solve(solveSpecPath, paramset['nS'], paramset['nC'], all_params)

results = {}
finalResult = None
ongoing = 0


# def parallelFindMinimalRule(solve_params):
#     # Never need to check for more than the topology can specify
#     maxNT, maxNC = countParticlesAndBindings(solve_params.topology)
#     if not solve_params.nS:
#         maxCubeTypes = maxNT
#     if not solve_params.nC:
#         maxColors = maxNC
#     global ongoing
#     asyncResults = []
#     with multiprocessing.Pool(maxtasksperchild=1) as p:
#         for nCubeTypes, nColors in smartEnumerate(maxCubeTypes, maxColors):
#             subparams = copy.deepcopy(solve_params)
#             subparams.nS = nCubeTypes
#             subparams.nC = nColors
#             r = p.apply_async(
#                 findRuleFor,
#                 args=(subparams,),
#                 callback=log_result,
#                 error_callback=log_error
#             )
#             asyncResults.append(r)
#         while not finalResult:
#             pass
#         return finalResult


# def findRules(topPath, nCubeTypes='auto', nColors='auto', nSolutions='auto', nDim=3, torsionalPatches=True):
#     polyurl = "https://akodiat.github.io/polycubes?rule={}"
#     if nCubeTypes == 'auto' or nColors == 'auto':
#         if nSolutions == 'auto':
#             nSolutions = 100
#         r = [parallelFindMinimalRule(topPath, nSolutions=nSolutions, nDim=nDim, torsionalPatches=torsionalPatches)]
#     else:
#         if nSolutions == 'auto':
#             nSolutions = 1
#         sols = find_solution(topPath, nCubeTypes, nColors, nSolutions, nDim, torsionalPatches)
#         if sols == 'TIMEOUT':
#             log('Timed out')
#             return
#         r = [ruleToDec(rule) for rule in sols]
#     if len(r) > 0:
#         for rule in r:
#             log(polyurl.format(rule))
#             if nDim == 2:
#                 log(translateToPolyominoNotation(parseDecRule(rule)))
#         return r
#     else:
#         log('Sorry, no solution found', flush=True)
#         return
#

def solve(solve_params: SolveParams) -> Union[None, SATSolution]:
    """
    Args:
        solve_params: parameters for this solve attempt

    Returns:
        a SATSolution object, or None if no solution is found
    """
    tstart = datetime.now()
    soln = None

    logger = solve_params.get_logger()

    logger.info(f"{solve_params.nC} colors and {solve_params.nS} cube types")
    # try:
    mysat = Polysat(solve_params)
    mysat.init()
    logger.info(f"Constructed sat problem with {mysat.problem.num_variables()} variables and {mysat.problem.num_clauses()} clauses.")
    mysat.check_settings()
    good_soln = mysat.find_solution()

    if good_soln:
        logger.info(f"WORKING RULE: {good_soln.decRuleOld()}")
    else:
        logger.info("Could not find a working solution!")

    logger.info(f"Solved for {solve_params.nS} types and {solve_params.nC} colors in {datetime.now() - tstart}")
    return good_soln


# worker for multiprocessing
def solve_worker(params: tuple[int, int, int, dict, str], attach_parent=True):
    (idx, nS, nC, data, topname) = params

    solve_params = SolveParams(
        topname,
        topology=data['bindings'],
        nColors=nC,
        nSpecies=nS,
        **data)

    logger = setup_logger(solve_params.get_logger_name())
    if attach_parent:
        # add superlogger handler
        for main_logger_handler in logging.getLogger(f"{topname}_main").handlers:
            logger.addHandler(main_logger_handler)

    logging.getLogger(f"{topname}_main").info(f"Attaching process to solve "
                                              f"nS={nS}, nC={nC} "
                                              f"logged to file `{str(get_log_dir() / 'SAT' / f'{solve_params.get_logger_name()}.log')}`")

    logger.info(f"------------------COMPUTING: nS={nS}, nC={nC}--------------")

    solution = solve(solve_params)
    if isinstance(solution, SATSolution):
        solution.printToLog()
        if solution.has_coord_map():
            solution.exportScene(topname)
            # return when we find solution


def solve_multi(solve_params: dict,
                solve_name: str,
                checks: list[tuple[int, int]],
                num_cpus: int):
    """
    batches solve of structures

    """
    worker_params = [(idx, nS, nC, solve_params, solve_name)
                     for (idx, (nS, nC)) in enumerate(checks)]
    logging.getLogger(f"{solve_name}_main").info(
        f"Starting solve pool for {solve_name} with {num_cpus} processes, over {len(checks)} specs.")
    if num_cpus > 1:
        with Pool(processes=num_cpus) as pool:
            pool.map(solve_worker, worker_params)
    else:
        for p in worker_params:
            solve_worker(p)

def get_max_colors(solveSpec: dict) -> int:
    """
    Given a solve spec, returns the number of colors in the fully-addressable
     rule for the provided topology
    Args:
        solveSpec: a solve spec dict

    Returns: int, number of colors required for fully-addressable rule

    """
    nBind = len(solveSpec['bindings'])
    if 'extraConnections' in solveSpec:
        return nBind + len(solveSpec['extraConnections'])
    else:
        return nBind


def get_max_species(solveSpec: dict) -> int:
    """
    Given a solve spec, returns the number of cube types (species) in the fully
    addressable rule for the provided topology
    Args:
        solveSpec: a solve spec dict

    Returns: int, number of cube types reqd for fully addressable rule
    """

    return max(itertools.chain.from_iterable([[
        x for (i, x) in enumerate(b) if i % 2 == 0]
        for b in solveSpec['bindings']])) + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAT solver')
    parser.add_argument('solve_spec', type=str, help='Solve specification file name')
    parser.add_argument('--nS', type=int, help='Minimum number of species')
    parser.add_argument('--nC', type=int, help='Minimum number of colors')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')  # Add this line

    args = parser.parse_args()

    if args.solve_spec.find(".") == -1:
        args.solve_spec = args.solve_spec + ".json"

    try:
        with open(get_input_dir() / "topologies" / args.solve_spec, 'r') as f:
            filestr = f.read()

            data = json.loads(filestr)
            solve_params = SolveParams(
                args.solve_spec[args.solve_spec.rfind(os.sep)+1:args.solve_spec.find(".")],
                topology=data['bindings'],
                nColors=args.nC,
                nSpecies=args.nS,
                **data)
            setup_logger(solve_params.get_logger_name())
            if args.verbose:
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                solve_params.get_logger().addHandler(handler)

            solve_params.get_logger().info("Topology file: ")
            solve_params.get_logger().info(filestr)

            solve_name = args.solve_spec[:args.solve_spec.find(".")]

            solution = solve(solve_params)
            if isinstance(solution, SATSolution):
                solution.printToLog(logger=solve_params.get_logger())
                solution.exportScene(solve_name)
            else:
                solve_params.get_logger().info("No solution found!")

            solve_params.get_logger().info("Done!")

    except FileNotFoundError:
        logging.error(f"No file `{get_input_dir() / 'topologies' / args.solve_spec}`")
