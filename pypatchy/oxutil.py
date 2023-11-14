# # This file was written by god-knows-who god-knows-when for god-knows-what reason, god-knows-when
# # Some of this might be redundant with oxpy or oat
# # some of this might be candidates for merging with oxpy or oat
# # who knows? the universe is a complecated and confusing place
# from dataclasses import dataclass
# from pathlib import Path
# from typing import IO, Iterable, Union
#
# import numpy as np
# from collections import namedtuple
# import json
# from datetime import datetime
#
#
# # Absolutely hate these
# # shadow names from oxDNA_analysis_tools.UTIL.data_structures
# # they're ALMOST the same but NOT QUITE
#
# Base = namedtuple('Base', ['type', 'p3', 'p5'])
#
#
# # @dataclass
# # class Strand:
# #     id: int
# #     bases: list[Base]
# #
# #     def __rshift__(self, other: int):
# #         return Strand(self.id, [
# #            Base(base.type,
# #                 base.p3 + other if base.p3 != -1 else -1,
# #                 base.p5 + other if base.p5 != -1 else -1) for base in self.bases
# #         ])
# #
# # @dataclass
# # class Topology:
# #
# #     nbases: int
# #     nstrands: int
# #     strands: list
#
#
# def make_strand(sid: int, start_id: int, seq: str) -> Strand:
#     return Strand(sid, [
#         Base(seq[0], -1, start_id + 1),
#         *[
#             Base(x, i + start_id - 1, i + start_id + 1) for i,x in list(enumerate(seq))[1:-1]
#         ],
#         Base(seq[-1], len(seq) - 2, -1)
#     ])
#
#
# def get_seq(s: Strand) -> str:
#     return "".join([b.type for b in s.bases])
#
#
# def write_configuration_header(f: IO, conf):
#     f.write(f't = {int(conf.time)}\n')
#     f.write(f"b = {' '.join(conf.box.astype(str))}\n")
#     f.write(f"E = {' '.join(conf.energy.astype(str))}\n")
#
#
# def write_configuration(f: IO, conf):
#     """
#     Writes a conformation file I guess
#     """
#     for p, a1, a3 in zip(conf.positions, conf.a1s, conf.a3s):
#         f.write('{} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(' '.join(p.astype(str)), ' '.join(a1.astype(str)),
#                                                             ' '.join(a3.astype(str))))
#
#
# def read_top(path: Union[str, Path]):
#     """
#     Reads an oxDNA topology file???
#     Isn't there an oat method for this?
#     """
#     base2strand = {}
#     if isinstance(path, str):
#         path = Path(path)  # i swear i'm a real scientist
#     assert path.is_file()
#     with path.open("r") as file:
#         lines = file.readlines()
#     # get line info
#     nbases, nstrands = map(int, lines[0].split())
#     # generate placeholder for bases
#     strands = [Strand(i, []) for i in range(1, nstrands + 1)]
#     # generate the return object
#     top_info = Topology(nbases, nstrands, strands)
#     i = 0
#     for line in lines[1:]:
#         sid, t, p3, p5 = line.split()
#         sid, p3, p5 = map(int, [sid, p3, p5])
#         b = Base(t, p3, p5)
#         top_info.strands[sid - 1].bases.append(b)
#         base2strand[i] = sid - 1
#         i += 1
#     return top_info, base2strand
#
#
# def merge_tops(tops):
#     # as we can have arbitrary topologies
#     # we need to iterate over them to figure out the number of bases \ strands
#     nbases, nstrands = 0, 0
#     for ti in tops:
#         nbases += ti.nbases
#         nstrands += ti.nstrands
#
#         # generate placeholder for bases
#     strands = [Strand(strand_id, []) for strand_id in range(1, nstrands + 1)]
#     # generate the return object
#     top_info = Topology(nbases, nstrands, strands)
#     # now we have to update the bases
#     offset = 0
#     sid = 0
#     # fill in the bases with the new base \ strand offset
#     tl = len(tops)
#     for id, ti in enumerate(tops):
#         print(f"{id + 1}/{tl}", end="\r")
#         if id > 0:
#             offset += ti.nbases
#         for strand in ti.strands:
#             top_info.strands[sid].bases.extend([Base(b.type,
#                                                      b.p3 + offset if b.p3 != -1 else -1,
#                                                      b.p5 + offset if b.p5 != -1 else -1)
#                                                 for b in strand.bases])
#             sid += 1
#     print()
#     return top_info
#
#
# def write_top(top_info, path="out_f_merged.top"):
#     with open(path, "w") as file:
#         file.write(f"{top_info.nstrands} {top_info.nbases}\n")
#         for id, strand in enumerate(top_info.strands):
#             for b in strand.bases:
#                 file.write(f"{id + 1} {b.type} {b.p3} {b.p5}\n")
#
#
# def write_oxview(tops, confs, clusters, file_path: Path):
#     assert file_path.parent.exists()
#     oxview_json = {
#         "box": confs[0].box.tolist(),
#         "date": datetime.now().isoformat(),
#         "forces": [],
#         "selections": []
#     }
#
#     system = {
#         "id": 0,
#         "strands": []
#     }
#     sid = 0
#     offset = 0
#     for particle_idx, (top_info, conf_info) in enumerate(zip(tops, confs)):
#         bid = 0
#         for strand in top_info.strands:
#             strand_json = {
#                 "class": "NucleicAcidStrand",
#                 "id": sid,
#                 "end3": float('inf'),  # set later
#                 "end5": 0,  # set later
#                 "monomers": []
#             }
#             for b in strand.bases:
#                 nucleotide = {
#                     "a1": conf_info.a1s[bid].tolist(),
#                     "a3": conf_info.a3s[bid].tolist(),
#                     "class": "DNA",
#                     # "cluster": 0,
#                     "cluster": particle_idx if bid not in clusters[particle_idx] else clusters[particle_idx][bid],
#                     "color": 0,  # NOT AN IMPORTANT BIT
#                     "id": bid + offset,
#                     "n3": b.p3 + offset if b.p3 != -1 else -1,
#                     "n5": b.p5 + offset if b.p5 != -1 else -1,
#                     "p": conf_info.positions[bid].tolist(),
#                     "type": b.type
#                 }
#                 strand_json["end3"] = min(strand_json["end3"], bid + offset)
#                 strand_json["end5"] = max(strand_json["end5"], bid + offset)
#                 strand_json["monomers"].append(nucleotide)
#                 bid += 1
#             system["strands"].append(strand_json)
#             sid += 1
#         offset += top_info.nbases
#     oxview_json["systems"] = [system]
#     with file_path.open("w") as f:
#         json.dump(oxview_json, f)
