from pypatchy.design.reduce_rule import RuleReducer
from ..polycubeutil.polycubesRule import PolycubesRule
from ..polycubeutil.polycube_structure import PolycubeStructure
from pypatchy.structure import TypedStructure
from .common_component import CommonComponent


class CannotDesignAMFAError(Exception):
    pass


def count_cc_type_divergences(cc: CommonComponent) -> int:
    return len(get_cc_type_divergences(cc))

def get_cc_type_divergences(cc: CommonComponent) -> set[int]:
    """
    Parameters:
        cc: a common component where structures are PolycubeStructure objects

    """
    # iter nodes in common component
    type_divergences = set()
    for n in cc.graph.nodes:
        # iter structures in cc
        cts = None
        for i, s in enumerate(cc.full_structures):
            assert isinstance(s, TypedStructure), \
                "Common component type divergence can't be assessed in an untyped structure!"
            # find node in structure
            snode = cc.node_in_structure(n, i)
            # node should always be in structure
            assert snode is not False
            # if no type has been determined at this point
            if cts is None:
                cts = s.particle_type(snode)
            # if type has been determined and this one doesn't match
            elif cts != s.particle_type(snode):
                # log type divergence
                type_divergences.add(n)
                # break
                break
    return type_divergences


class DesignAMFA:
    polycube1: PolycubeStructure
    polycube2: PolycubeStructure

    def __init__(self, *args: PolycubeStructure):
        self.polycube1, self.polycube2 = args
        self.rule = self.polycube1.rule
        assert self.rule == self.polycube2.rule

    def solve(self) -> PolycubesRule:
        # step 1: superimpose structures
        # try to find the transform that superimposes s0 onto s1 so that
        # the most cube types of the same type are in the same position
        best_cc = None
        polycube1_substructures = list(self.polycube1.substructures())
        polycube2_substructures = list(self.polycube2.substructures())
        pivot = None
        for s1, s2 in itertools.product(polycube1_substructures, polycube2_substructures):
            if len(s1) != len(s2):
                continue
            if best_cc is not None and len(s1) < len(best_cc):
                continue

            # construct iter structural homomorpism which maps s2 onto s1
            for homomorph in s2.homomorphisms(s1):
                # construct common component of s1 and s2
                cc = CommonComponent(s1,
                                     [self.polycube1, self.polycube2],
                                     [identity_homomorphism(s1), homomorph])
                if count_cc_type_divergences(cc) != 1:
                    continue
                new_pivot = cc.is_macoco()
                if new_pivot is not False:  # should be true if cc.is_macoco() == 0, which is a valid pivot point
                    type_divergences = get_cc_type_divergences(cc)
                    # if not all([
                    #     f.[new_pivot] in type_divergences
                    #     for s, f in zip(cc.full_structures, cc.homomorphisms)]
                    # ):
                    if new_pivot not in type_divergences:
                        continue
                    best_cc = cc
                    pivot = new_pivot
                    break

        if best_cc is None:
            raise CannotDesignAMFAError("No homomorphism exists to map polycube 1 onto polycube 2")

        # step 3: try to merge the two cube types
        cts = []
        # align cube types
        for i, f in enumerate(best_cc.homomorphisms):
            r, _ = f.as_transform()
            v = f.rmap_location(pivot)
            cube = f.source.get_cube(v)
            ct = cube.get_cube_type()
            # for p in ct.patches():
            #     f.
            # make rotations of ct line up with the common component
            ct.rotate(r @ cube.rotation().as_matrix())
            cts.append(ct)

        patches_to_merge = set()
        for delta, _ in enumerate(RULE_ORDER):
            if not all([ct.has_patch(delta) for ct in cts]):
                patches_to_merge.add(delta)

        reducer = RuleReducer(self.rule)
        reducer.merge_cube_types(cts[0],
                                 cts[1],
                                 ct1_to_incl=patches_to_merge,
                                 ct2_to_incl=patches_to_merge)
        return reducer.rule


import itertools
import json
import logging
import multiprocessing
import os
import sys
from typing import Generator

import networkx as nx
import numpy as np
import scipy.spatial.distance
from pypatchy.structure import Structure, StructuralHomomorphism, identity_homomorphism
from ..polycubeutil.polycubesRule import diridx, RULE_ORDER
from ..util import get_input_dir, rotidx

# from ..design.design_rule import Polysat

from .common_component import CommonComponent

from Bio.SVDSuperimposer import SVDSuperimposer


# class CommonComponentSAT(SATProblem):
#
#     """
#     given a component and a list of structures, uses SAT to find mappings of the component topology onto
#     the structure.

#     The resulting SAT problem should be satisfied and provide the mapping if the provided component is
#     a common component of the provided structure, and unsatisfiable otherwise
#     """
#
#     candidate: Structure # a strucure that is being tested for common componentness
#     structures: list[Structure]  # structures that the candidate should be a common component of
#     rotations: dict[dict[int, int]]
#     nP: int
#
#     def __init__(self,
#                  logger: logging.Logger,
#                  solve_timeeout: int,
#                  candidate: Structure,
#                  structures: list[Structure]):
#         super().__init__(logger, solve_timeeout)
#         self.candidate = candidate
#         self.structures = structures
#         self.rotations = enumerateRotations()
#         self.nP = 6
#
#     def init(self):
#         self.generate_constraints()
#
#     def generate_constraints(self) -> list[SATClause]:
#         constraints = []
#
#         # for each structure
#         for s in range(self.nS()):
#             # require that a 1:1 location map exists from our candidate to the structure
#             constraints.extend(self.gen_legal_locations(s))
#
#             # require that a valid rotation map exists from our candidate to this structure
#             constraints.extend(self.gen_legal_rotations(s))
#
#             # constraints.extend(self.gen_structure_bindings(s))
#
#             constraints.extend(self.gen_equivalance(s))
#         self.logger.info(f"Added {len(constraints)} total SAT clauses")
#         self.basic_sat_clauses.extend(constraints)
#         return constraints
#
#     def R(self, s: int, ri: int, rj: int) -> int:
#         assert -1 < s < self.nS(), "Structure index out of bounds"
#         assert -1 < ri < self.nR, "rotation index out of bounds"
#         return self.variable("R", s, ri, rj)
#
#     def V(self, s: int, li: int, lj: int) -> int:
#         assert -1 < s < self.nS(), "Structure index out of bounds"
#         assert -1 < li < len(self.candidate), "Candidate vertex inedx out of bounds"
#         assert -1 < lj < self.nLS(s), f"Structure {s} vertex index out of bounds"
#         return self.variable("V", s, li, lj)
#
#     # def S(self, s: int, li: int, ri: int, lj: int, rj: int) -> int:
#     #     assert -1 < s < self.nS()
#     #     assert -1 < li < self.nLS(s)
#     #     assert -1 < ri < self.nP
#     #     assert -1 < lj < self.nLS(s)
#     #     assert -1 < rj < self.nP
#     #     # li, ri, lj, rj is equivalent to lj, rj, li, ri so by
#     #     # having a single sat var for them we can cut our number of S vars in half
#     #     if lj > li:
#     #         x = li
#     #         li = lj
#     #         lj = x
#     #         x = ri
#     #         ri = rj
#     #         rj = x
#     #     return self.variable("S", li, ri, lj, rj)
#
#     def E(self,
#           cli: int, cri: int, clj: int, crj: int,
#           s: int,
#           sli: int, sri: int, slj: int, srj: int,
#           ) -> int:
#         assert -1 < cli < self.nLC()
#         assert -1 < cri < self.nP
#         assert -1 < clj < self.nLC()
#         assert -1 < crj < self.nP
#         assert -1 < s < self.nS()
#         assert -1 < sli < self.nLS(s)
#         assert -1 < sri < self.nP
#         assert -1 < slj < self.nLS(s)
#         assert -1 < srj < self.nP
#
#         if clj > cli:
#             x = cli
#             cli = clj
#             clj = x
#
#         if slj > sli:
#             x = cli
#             cli = clj
#             clj = x
#
#         return self.variable("E", cli, cri, clj, slj, s, sli, sri, slj, srj)
#
#     # def C(self, li: int, ri: int, lj: int, rj: int) -> int:
#     #     assert -1 < li < self.nLC()
#     #     assert -1 < ri < self.nP
#     #     assert -1 < lj < self.nLC()
#     #     assert -1 < rj < self.nP
#     #     # li, ri, lj, rj is equivalent to lj, rj, li, ri so by
#     #     # having a single sat var for them we can cut our number of S vars in half
#     #     if lj > li:
#     #         x = li
#     #         li = lj
#     #         lj = x
#     #         x = ri
#     #         ri = rj
#     #         rj = x
#     #     return self.variable("C", li, ri, lj, rj)
#
#     def gen_legal_rotations(self, s: int) -> list[SATClause]:
#         assert -1 < s < self.nS()
#         constraints = []
#         # each rotation index maps to one other rotation index
#         for ri in range(self.nP):
#             constraints.extend(self._exactly_one([
#                 self.R(s, ri, rj) for rj in range(self.nP)
#             ]))
#         # each rotation index maps from exactly one other rotation index
#         for rj in range(self.nP):
#             constraints.extend(self._exactly_one([
#                 self.R(s, ri, rj) for ri in range(self.nP)
#             ]))
#         # iterate possible permutations of 6 elements and forbid all permutations not in the
#         # rotation enumeration
#         # I hate this approach but it's 696 sat clauses, which is far from unworkable
#         allowed_rots = [tuple(r.values()) for r in enumerateRotations().values()]
#
#         for p in itertools.permutations(range(6)):
#             if p not in allowed_rots:
#                 # require at least one of the index mappings to be false
#                 constraints.append((-self.R(s, k, v) for k,v in enumerate(p)))
#
#
#         # # forbid non-rotational symmetries
#         # for rot_map in forbidden_symmetries():
#         #     # this symmetry is Forbidden
#         #     # which we do by requiring that at least one of the rotation map vars that encode
#         #     # it to be false
#         #     constraints.append((-self.R(s, r, rot_map[r]) for r in rot_map))
#         self.logger.info(f"Added {len(constraints)} rotation-mapping clauses for structure {s}")
#         return constraints
#
#     def gen_legal_locations(self, s: int) -> list[SATClause]:
#         assert -1 < s < self.nS()
#         struct = self.structures[s]
#         constraints: list[SATClause] = []
#         # for each loaction in the candidate component
#         for li in self.candidate.vertices():
#             # this vertex in the candidate component maps to exactly one vertex in `struct`
#             constraints.extend(
#                 self._exactly_one([
#                     self.V(s, li, lj) for lj in struct.vertices()])
#             )
#         # for each locaction in `struct`
#         for lj in struct.vertices():
#             # this vertex in struct is mapped to by at most one vertex in the candidate
#             constraints.extend(
#                 self._at_most_one(*[
#                     self.V(s, li, lj) for li in self.candidate.vertices()])
#             )
#         self.logger.info(f"Added {len(constraints)} location-mapping clauses for structure {s}")
#         return constraints
#
#     def gen_equivalance(self, s: int) -> list[SATClause]:
#         """
#         Generates a list of SAT clauses which require that the bindings which are present in
#         the candidate are also present in structure with index s
#         Args:
#             s: index of one of the structures
#
#         Returns:
#             a list of SAT clauses (int tuples)
#
#         """
#         clauses = []
#         struct = self.structures[s]
#
#         possible_struct_bindings = itertools.product(
#             range(self.nLS(s)),
#             range(self.nP),
#             range(self.nLS(s)),
#             range(self.nP)
#         )
#
#         # for each possible relationship between a binding in the candidate and the structure
#         for (cli, cri, clj, crj), (sli, sri, slj, srj) in itertools.product(self.candidate.bindings_list,
#                                                                             struct.bindings_list):
#             # extract edge variable
#             # edge (cli, cri, clj, crj) is equivalent to the edge (sli, sri, slj, srj) in s
#             v = self.E(cli, cri, clj, crj, s, sli, sri, slj, srj)
#             # equivalence
#             # E <-> R(s, cri, sri) ^ R(s, crj, srj) ^ V(s, cli, sli) ^ V(s, clj, slj)
#             # E (cli, cri, clj, crj, sli, sri, slj, srj) is equivalent to
#             # "The edge connecting face cri on location c
#             equiv_clauses = [
#                 (-v, -self.R(s, cri, sri)),
#                 (-v, -self.R(s, crj, srj)),
#                 (-v, -self.V(s, cli, sli)),
#                 (-v, -self.V(s, clj, slj)),
#                 (
#                     v,
#                     -self.R(s, cri, sri),
#                     -self.R(s, crj, srj),
#                     -self.V(s, cli, sli),
#                     -self.V(s, clj, slj)
#                 )
#             ]
#
#             clauses.extend(equiv_clauses)
#
#         # edge mapping should be 1 to 1
#         # for each binding in the candidate
#         for cli, cri, clj, crj in self.candidate.bindings_list:
#
#             # exactly one binding in the structure must be equivalent
#             clauses.extend(self._exactly_one([
#                 self.E(cli, cri, clj, crj, s, sli, sri, slj, srj)
#                 for sli, sri, slj, srj in struct.bindings_list]))
#
#         if len(self.candidate.bindings_list) > 1:
#             # for each binding in the structure
#             for sli, sri, slj, srj in struct.bindings_list:
#                 # at most one binding in the candidate is equivalent
#                 clauses.extend(self._at_most_one(*[
#                     self.E(cli, cri, clj, crj, s, sli, sri, slj, srj)
#                     for cli, cri, clj, crj in self.candidate.bindings_list]))
#
#         # for each set of bindings in the candidate binding set
#         # for cli, cri, clj, crj in self.candidate.bindings_list:
#             # require that one of the bindings in the structure corresponds
#             # don't have to require exactly one, pretty sure this is implicit in the problem
#
#             # clauses in disjunctive normal form (easier to make list this way)
#             # dnf_clauses = [(
#             #     self.R(s, cri, sri),
#             #     self.R(s, crj, srj),
#             #     self.V(s, cli, sli),
#             #     self.V(s, clj, slj)
#             # ) for sli, sri, slj, srj in struct.bindings_list]
#             #
#             # for clause in itertools.product(*dnf_clauses):
#             #     clauses.append(list(set(clause)))
#
#         self.logger.info(f"Added {len(clauses)} clauses to map structural bindings from candidate onto structure {s}")
#         return clauses
#
#     # def gen_structure_bindings(self, s: int) -> list[SATClause]:
#     #     """
#     #     constructs what's basically an adjacency matrix extended into direction-space dimensionality
#     #     for the graph of this structure
#     #     """
#     #     clauses: list[tuple[int, ...]] = []
#     #     # loop all possible edges (space in adjacency map)
#     #     bindlist: set[tuple[int, int, int, int]] = self.structures[s].bindings_list
#     #     for i, ri, j, rj in itertools.product(
#     #             range(self.nLS(s)),
#     #             range(self.nP),
#     #             range(self.nLS(s)),
#     #             range(self.nP)):
#     #         if j >= i:
#     #             continue  # skip redundant edges
#     #         # if (ignoring order) i binds to j via ri, rj is in this structure
#     #         if (i, ri, j, rj) in bindlist or (j, rj, i, ri) in bindlist:
#     #             clauses.append((self.S(s, i, ri, j, rj),))  # require var
#     #         else:
#     #             clauses.append((-self.S(s, i, ri, j, rj),))  # forbid var
#     #     # assert len(clauses) == self.nLS(s) * self.nLS(s) * self.nP * self.nP / 2
#     #     self.logger.info(f"Added {len(clauses)} binding clauses for structure {s}")
#     #     return clauses
#
#     def nS(self) -> int:
#         return len(self.structures)
#
#     def nLS(self, s: int) -> int:
#         return len(self.structures[s])
#
#     def nLC(self) -> int:
#         return len(self.candidate)
#
#     def output_cnf(self, out: Union[IO, None] = None) -> str:
#         """ Outputs a CNF formula """
#         num_vars = max(self.variables.values())
#         num_constraints = len(self.basic_sat_clauses)
#         outstr = "p cnf %s %s\n" % (num_vars, num_constraints)
#         # add basic clauses
#         for c in self.basic_sat_clauses:
#             outstr += ' '.join([str(v) for v in c]) + ' 0\n'
#
#         if out is not None:
#             out.write(outstr)
#         return outstr
#
#     def solve(self) -> Union[CommonComponent, None]:
#
#         formula = CNF(from_string=self.output_cnf())
#         tstart = datetime.now()
#         with Glucose4(bootstrap_with=formula.clauses) as m:
#             # if the solver has a timeout specified
#             if self.solver_timeout:
#                 timer = Timer(self.solver_timeout, interrupt, [m])
#                 timer.start()
#                 solved = m.solve_limited(expect_interrupt=True)
#                 timer.cancel()
#             else:
#                 solved = m.solve()
#             if solved:
#                 self.logger.info("Common component is valid!")
#                 # pysat returns solution as a list of variables
#                 # which are "positive" if they're true and "negative" if they're
#                 # false.
#                 model = m.get_model()
#                 # we can pass the model directly to the SATSolution constructor because it will
#                 # just check for positive variables to be present
#                 return self.soln_to_cc(frozenset(model))
#             else:
#                 self.logger.info("Common component is not valid.")
#                 return None
#
#     def soln_to_cc(self, sat_results: frozenset) -> CommonComponent:
#         struct_r_maps = [{} for _ in self.structures]
#         struct_l_maps = [{} for _ in self.structures]
#         # soln = self.get_solution(sat_results)
#         # for vname, vnum in soln:
#         #     m = re.match(r"R\((\d*),\s*(\d*),\s*(\d*)\)", vname)
#         #     if m:
#         #         species, ri, rj = m.groups()
#         #         assert int(ri) not in struct_r_maps[int(species)]
#         #         struct_r_maps[int(species)][int(ri)] = int(rj)
#         #         continue
#         #     m = re.match(r"V\((\d*),\s*(\d*),\s*(\d*)\)", vname)
#         #     if m:
#         #         species, li, lj = m.groups()
#         #         assert int(li) not in struct_l_maps[int(species)]
#         #         struct_l_maps[int(species)][int(li)] = int(lj)
#         #         continue
#
#         soln = self.get_solution_vars(sat_results)
#         for vnum, (species, ri, rj) in soln["R"]:
#             assert ri not in struct_r_maps[species]
#             struct_r_maps[species][ri] = rj
#             continue
#
#         for vnum, (species, li, lj) in soln["V"]:
#             assert li not in struct_l_maps[species]
#             struct_l_maps[species][li] = lj
#             continue
#
#         edge_vars = [
#             {
#                 "structure": v[4],
#                 "candidate edge": {
#                     "from": {
#                         "l": v[0], # location
#                         "r": v[1]  # face idx
#                     },
#                     "to": {
#                         "l": v[2], # location
#                         "r": v[3]  # face idx
#                     }
#                 },
#                 "structure edge": {
#                     "from": {
#                         "l": v[5], # location
#                         "r": v[6]  # face idx
#                     },
#                     "to": {
#                         "l": v[7], # location
#                         "r": v[8]  # face idx
#                     }
#                 }
#             }
#             for _, v in soln["E"]
#         ]
#
#         # find indeces of rotation map
#         rot_matches = [
#             [
#                 i for i, rot in self.rotations.items()
#                 if (rot_map_to_quat(rot) == rot_map_to_quat(r_map)).all()
#             ]
#             for r_map in struct_r_maps
#         ]
#         rmapidxs = [r[0] for r in rot_matches]
#
#         homomorphisms = [
#             StructuralHomomorphism(self.candidate, struct, rmapidx, lmap)
#             for struct, rmapidx, lmap in zip(self.structures, rmapidxs, struct_l_maps)
#         ]
#         return CommonComponent(self.candidate, self.structures, homomorphisms)

#
# def identify_common_component(logger: logging.Logger,
#               solve_timeout: int,
#               candidate: Structure,
#               structures: list[Structure]) -> Union[None, CommonComponent]:
#     print("----------------------------------------------")
#     logger.info(f"Initializing SAT problem for common component {candidate} of structures {','.join(str(s) for s in structures)}")
#     mysat = CommonComponentSAT(logger, solve_timeout, candidate, structures)
#     mysat.init()
#     logger.info(f"SAT solving...")
#     return mysat.solve()


def avg_2pt_dist(structure: Structure) -> float:
    # might be a way to optimize this with numpy
    a = structure.matrix()
    pairwise_distances = scipy.spatial.distance.pdist(a)
    return np.average(pairwise_distances)


def test_cc_permutation(parameters: tuple[np.ndarray, np.ndarray, tuple[int], tuple[int]]
                        ) -> tuple[tuple[int], dict[int, int], np.ndarray, bool]:
    candidate_matrix, structure_matrix, c, p = parameters
    # permute structure!
    permuted_structure_matrix = structure_matrix[p, :]
    mapping = np.array(c)[np.array(p)]
    result = ()
    # check for trivial solution
    if (permuted_structure_matrix == candidate_matrix).all():
        h = {i: n for i, n in enumerate(mapping)}
        rot = np.identity(3)
        result = p, h, rot, True
    else:
        sup = SVDSuperimposer()
        sup.set(permuted_structure_matrix, candidate_matrix)
        sup.run()
        # if two structures have a nonzero rms they are not the same. move on.
        if sup.get_rms() > 0:
            result = None, None, None, False
        else:
            # a note about indexing
            # column indexes in structure_matrix correspond to node numbers in the tuple c
            # if you rearrange structure_matrix by structure_matrix[p, :],
            # column indexes will now correspond to node numbers the array mapping (below)

            # get rotation & translation that maps candidate onto structure
            rot, tran = sup.get_rotran()
            # apply transformation
            transformed_candidate = candidate_matrix @ rot + tran
            # loop positions in candidate
            h = {i: n for i, n in enumerate(mapping)}
            # # enumerate mapping as key-value pairs
            # for i, n in enumerate(mapping):
            #     # grab position vector from structure matrix
            #     position = permuted_structure_matrix[i, :]
            #     # find match within transformed candidate matrix
            #     match_map = ~(transformed_candidate - position[np.newaxis, :]).any(axis=1)
            #     assert np.sum(match_map) < 2
            #     if np.sum(match_map) > 0:
            #         # index of column in transformed candidate matrix which matches reordered structure matrix
            #         idx = np.nonzero(match_map)[0][0]
            #         # map index in candidate matrix (corresponds to candidate structure node, we're not doing subgraphs)
            #         # to node in substructure.
            #         h[idx] = n
            result = p, h, rot, True
    return result


def identify_common_component(parametrs: tuple[logging.Logger,  # logger
                                               Structure,  # candidate
                                               list[Structure]  # structures
                            ]) -> Generator[CommonComponent, None, None]:
    logger, candidate, structures = parametrs

    candidate_avg_2pt = avg_2pt_dist(candidate)

    candidate_matrix = candidate.matrix()

    structural_homomorphisms = [[] for _ in structures]
    logger.info("Attempting to identify structural homomorphisms")
    # loop structures
    for w, structure in enumerate(structures):
        # structure that's identical to candidate will always be coolio
        if structure == candidate:
            structural_homomorphisms[w].append(identity_homomorphism(structure))
            continue
        # loop substructures of this structure
        # use itertools.combinations not itertools.permutations because subgraph will ignore order anyway

        for c in itertools.combinations(structure.graph.nodes, len(candidate)):
            # if the substructure is not connected, move on
            if not nx.is_strongly_connected(structure.graph.subgraph(c)):
                continue
            # first let's filter out obviously-wrong structures
            substructure = structure.substructure(c)
            # if the average distance between two points on the substructure is not the same
            # as the average distance between two points on the candidate, they won't be homomorphic. right?
            if abs(avg_2pt_dist(substructure) - candidate_avg_2pt) > 0.1:
                continue

            structure_matrix = substructure.matrix()

            # SVD superimposer is order dependant so loop permutations
            # gotta use range(len(c)) because c will often not be contiguous or have zero index
            def iter_h() -> Generator[tuple[np.ndarray, np.ndarray, tuple[int], tuple[int]], None, None]:
                for perm in itertools.permutations(range(len(c))):
                    yield candidate_matrix, structure_matrix, c, perm

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                for p, h, rot, tf in pool.imap_unordered(test_cc_permutation, iter_h()):
                    if tf:
                        assert len(h) == len(candidate)
                        # convert rotation to dict
                        rotmap = rotidx({
                            i: diridx(rot @ v) for i, v in enumerate(RULE_ORDER)
                        })
                        # construct structural homomorphism
                        logger.info("Found structural homomorphism!!!")
                        h = StructuralHomomorphism(candidate, structure, rotmap, h)
                        structural_homomorphisms[w].append(h)
                        break  # if we found a permutation that works we do not need to try more of them!
    for homo_group in itertools.product(*structural_homomorphisms):
        yield CommonComponent(candidate, structures, homo_group)


# class AMFASolver:
#     """
#     Designs allosteric multifarious assembly behavior for a set of structures
#     """
#
#     def __init__(self, name: str, **kwargs):
#         self.solve_name = name
#         assert "bindings" in kwargs
#         self.bindings = kwargs["bindings"]
#         self.solve_timeout = kwargs["solve_timeout"] if "solve_timeout" in kwargs else 10800  # default: 3 hrs
#         self.max_diff = None if "max_diff" not in kwargs else int(kwargs["max_duff"])
#
#         # construct graph of all nodes/edges
#         test_graph = nx.DiGraph()
#         for v1, d1, v2, d2 in self.bindings:
#             if v1 not in list(test_graph.nodes):
#                 test_graph.add_node(v1)
#             if v2 not in list(test_graph.nodes):
#                 test_graph.add_node(v2)
#             test_graph.add_edge(v1, v2, dirIdx=d1)
#             test_graph.add_edge(v2, v1, dirIdx=d2)
#
#         assert "extraConnections" not in kwargs  # i am begging you to Not
#         self.structures = []
#         # loop connected components
#         for c in nx.algorithms.strongly_connected_components(test_graph):
#             self.structures.append(Structure(graph=test_graph).substructure(c))
#
#         # TODO: feels like 2 is arbitrary even though I'm pretty sure I can't go higher
#         # at least for divergent AMFA
#         assert len(self.structures) == 2
#         # self.is_do_parallel = "parallel" in kwargs and kwargs["parallel"]
#         self.is_do_parallel = False
#
#         self.logger = setup_logger("amfa", get_log_dir() / "SAT" / "amfa.log")
#         self.logger.setLevel(logging.INFO)
#         self.logger.addHandler(logging.StreamHandler())
#
#     def identify_macguffins(self) -> Generator[tuple[CommonComponent, int], None, None]:
#         self.logger.info("Identifying macguffins")
#         for component in self.get_common_components():
#             self.logger.info(f"Checking if common component {str(component)} is a macguffin...")
#             pivot_point = component.is_macguffin()
#             if pivot_point > -1:
#                 self.logger.info(f"Identified pivot point at node {pivot_point}!")
#                 yield component, pivot_point
#             else:
#                 self.logger.info("Not a macguffin :(((")
#
#     def get_common_components(self) -> Generator[CommonComponent, None, None]:
#         self.logger.info("Identifying common components")
#         # use smallest structure to speed process
#         s0 = sorted(self.structures, key=lambda s: len(s))[0]
#
#         # include self in substructures, since other structures may be larger
#         candidates = [*s0.substructures(), s0]
#         candidates.sort(key=len, reverse=True)
#         sat_params = [
#             # I will be actually surprised if any take over 2 hrs to solve
#             (self.logger, candidate, self.structures)
#             for candidate in candidates
#         ]
#         sat_params.sort(key=lambda x: len(x[2]), reverse=True)
#         counter = 0
#         if self.is_do_parallel:
#             # parallel version
#             with multiprocessing.Pool() as pool:
#                 component_results = pool.map(identify_common_component, sat_params)
#         else:
#             # serial version
#             for p in sat_params:
#                 logging.info(f"Identidfying common component {str(p[1])} \
#                 of structures {','.join([str(s) for s in p[2]])}...")
#                 for c in identify_common_component(p):
#                     counter += 1
#                     logging.info(f"Found common component {str(c)}! Cumulative common component count {counter}.")
#                     yield c
#         self.logger.info(f"Found {counter} common components!")
#
#     def solve_amfa(self) -> Generator[PolycubesRule, None, None]:
#         """
#         This is where the fun begins!
#         Returns:
#             a PolycubesRule object designed with Allosteric Multifarious Assembly
#         """
#         for mg, pivot, mg_soln in self.get_macguffin_rules():
#             min_nC = mg_soln.num_colors + 1
#             min_nS = mg_soln.num_species + 1
#             # max num colors is the max num colors req'd to make the non-macguffin parts of the full structures
#             max_nC = mg_soln.num_colors + sum([mg.disjoint_n_edges(i) for i in range(mg.nstructures())])
#             # ditto for edges
#             max_nS = mg_soln.num_species + sum([mg.disjoint_n_nodes(i) for i in range(mg.nstructures())])
#             for ps in construct(self.solve_name, min_nC, max_nC, min_nS, max_nS, self.max_diff):
#                 # TODO: multifarious sovle tester!!
#                 # initialize SAT solver
#                 mysat = Polysat(ps)
#                 mysat.init()
#
#                 # require common component cube behavior
#                 for structure in mg.full_structures:
#                     pass
#
#                 overall_solution = mysat.find_solution(ps)
#                 if overall_solution:
#                     yield overall_solution.rule
#
#     def get_macguffin_rules(self) -> Generator[tuple[CommonComponent, int, SATSolution], None, None]:
#         # iterate macguffins
#         for mg, pivot in self.identify_macguffins():
#             max_num_colors = len(mg.bindings_list)
#             max_num_species = len(mg.graph.nodes)
#             solve_param_sets = construct(
#                 name="magcuffin",
#                 min_nC=1,
#                 max_nC=max_num_colors,
#                 min_nS=1,
#                 max_nS=max_num_species,
#                 max_diff=self.max_diff,
#                 bindings=mg.bindings_list,
#                 # 90 minutes seems safe enough if we're using
#                 solve_timeout=90 * 60,
#                 # use a "virtual nanoparticle" to make sure the pivot point is a unique particle type
#                 nanoparticles=[pivot])
#
#             for sat_solve_params in solve_param_sets:
#                 macguffin_rule = solve(sat_solve_params)
#                 if macguffin_rule:  # not None
#                     yield mg, pivot, macguffin_rule


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: amfaDesign.py [filename]")
    fpath = sys.argv[1]
    with open(get_input_dir() / "topologies" / fpath, 'r') as f:
        j = json.load(f)
    solver = DesignAMFA(fpath[fpath.rfind(os.sep) + 1:fpath.find(".")], **j)
    for soln in solver.solve():
        print(str(soln))
