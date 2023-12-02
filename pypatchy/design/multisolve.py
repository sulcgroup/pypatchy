import argparse
import itertools
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

from pypatchy.design import solve_params
from pypatchy.design.solve_exec import get_max_colors, get_max_species, solve_multi
import json

from pypatchy.util import get_input_dir
from solve_utils import setup_logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAT solver')
    parser.add_argument('solve_spec', type=str, help='Solve specification file name')
    parser.add_argument('--minS', type=int, default=1, help='Minimum number of species')
    parser.add_argument('--minC', type=int, default=1, help='Minimum number of colors')
    parser.add_argument('--maxS', type=int, default=0, help='Maximum number of species')
    parser.add_argument('--maxC', type=int, default=0, help='Maximim number of colors')

    parser.add_argument('--num_cpus', type=int, default=1, help='Number of CPUs available')

    parser.add_argument('--diff_limit', type=int, default=-1,
                        help="Limit on difference between nC and nS. The solver will skip specs where"
                             " |nC-nS| > diff_limit")

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.solve_spec.find(".") == -1:
        args.solve_spec = args.solve_spec + ".json"

    solve_spec_path: Path = get_input_dir() / "topologies" / args.solve_spec

    main_logger = setup_logger(f"{solve_spec_path.stem}_main")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if args.verbose:
        pass
        # solve_params.get_logger().addHandler(handler)

    try:
        with solve_spec_path.open('r') as f:
            filestr = f.read()
            main_logger.info("Topology file: ")
            main_logger.info(filestr)
            data = json.loads(filestr)
            if args.maxC == 0:
                args.maxC = get_max_colors(data)
            if args.maxS == 0:
                args.maxS = get_max_species(data)
        assert args.minS <= args.maxS, f"Minimum num species {args.minS} is greater than maximum number of species {args.maxS}!"
        assert args.minC <= args.maxC, f"Minimum num colors {args.minC} is greater than maximum number of colors {args.maxC}!"
        checks = list(itertools.product(range(args.minS, args.maxS + 1), range(args.minC, args.maxC + 1)))
        if args.diff_limit > -1:
            checks = [(s, c) for s, c in checks if abs(s - c) <= args.diff_limit]

        checks.sort(key=lambda x: x[0] + x[1])

        solve_name = args.solve_spec[:args.solve_spec.find(".")]
        solve_multi(data,
                    solve_name,
                    checks,
                    args.num_cpus)
        main_logger.info("Done!")

    except FileNotFoundError:
        main_logger.error(f"No file `{get_input_dir() / 'topologies' / args.solve_spec}`")
