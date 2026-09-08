---
name: structure-topology
description: Use when working with pypatchy's Structure/TypedStructure/GenericTypedStructure classes (pypatchy/structure.py) - building a target topology for analysis or design, constructing/interpreting Binding tuples, checking structural homomorphism/isomorphism between assemblies, building a periodic crystal topology from a unit cell (Tiling/splice), or debugging a rotation/direction mismatch in a homomorphism result. This is shared foundation underneath analyze-ensemble's cluster-classification targets, sat-crystal-design's whole topology-matching machinery, and polycubes-rules' PolycubeStructure. Not for the rule-string format itself (polycubes-rules) or PL/oxDNA particle file formats (patchy-model-formats).
---

# `Structure` / topology graphs (`pypatchy/structure.py`)

`Structure` is pypatchy's graph-based representation of "an assembly of
things bound together" — nodes are integer particle/cube-instance IDs
("locations"), edges are directed bindings each carrying a `dirIdx`
(the same [[polycubes-rules]] `RULE_ORDER` face-direction index used
everywhere else in this project). It's the common substrate underneath three
things that otherwise look unrelated: a hand-built comparison target in
[[analyze-ensemble]]'s cluster-classification steps, the entire topology/
target-matching machinery in [[sat-crystal-design]], and — via subclassing —
[[polycubes-rules]]' own `PolycubeStructure` (`polycube_structure.py`'s
`class PolycubeStructure(TypedStructure, Scene)`), so every method
documented here works on an actual assembled polycube, not just an abstract
hand-built target.

## The `Binding` tuple

`Binding = tuple[int, int, int, int]` — `(u, du, v, dv)`: location `u`'s
patch at direction `du` binds location `v`'s patch at direction `dv`. Backed
internally by a matching pair of directed edges (`u→v` with `dirIdx=du`,
`v→u` with `dirIdx=dv`) in an `nx.MultiDiGraph`; `.bindings_list` (a cached
property) collapses that back down to one canonical tuple per binding.
`flip(b)` returns `(v, dv, u, du)` — since edge iteration order isn't
guaranteed stable, code that looks up a specific binding by value generally
needs to check `b in bindings_list or flip(b) in bindings_list`, not just the
former (see `Polysat.matches_problem()` in [[sat-crystal-design]] for exactly
this pattern, done there specifically so a log written by another process/host
still compares equal). A "well-formed" binding (checked by
`is_bindings_euclidian()`, not enforced at construction) has `du`/`dv` as
geometric opposites — `abs(du - dv) == 1` under `RULE_ORDER`'s pairing.

## Core classes

- **`Structure(bindings=[...])`** or **`Structure(graph=some_nx_multidigraph)`** —
  the untyped base. `.vertices()`, `.num_vertices()`/`.num_edges()`/`len()`.
  `.is_connected()` / `.is_multifarious()` (not connected) /
  `.iter_components()` — this last one is literally how a multifarious
  (multi-component) design target gets split into its separate pieces in
  [[sat-crystal-design]] (`Polysat.targets_as_typed`).
  `.substructure(nodes)` / `.substructures()` — extract or enumerate
  connected node subsets, remapping vertex IDs down to a dense `0..n-1` range
  in the result (used e.g. by `local_constraint_subgraphs()` in
  [[sat-crystal-design]] to crop a competing crystal down to a negative-design
  constraint).
  `.rotate(rotation_map)` — apply a direction-index permutation to every
  edge, returning a `StructuralHomomorphism`.
  `.splice(other, splice_points)` — join two structures by cutting a binding
  in each and cross-connecting them instead; the building block `Tiling`
  uses to repeat a unit cell.
  `.slice(axis)` — drop every edge along one axis (its own direction and the
  opposite), leaving the rest — used for per-layer/symmetry analysis.
  `.is_crystal()` — true if there's a same-axis directional cycle along any
  of the 3 axes (i.e. the structure is periodic/infinite along that axis, not
  just finite). `.matrix()` / the internal `._get_cubes_coords()` — assigns
  each vertex an integer grid coordinate via BFS from an arbitrary root
  (requires `is_connected()`). `.draw()` — matplotlib visualization with
  direction labels on each edge. `.export_periodic_topology()` — see "Topology
  JSON format" below.

- **`FiniteLatticeStructure(Structure)`** — adds O(1) position↔index lookup
  (`.cube(idx)`, `.cube_idx(coords)`) on top of `.matrix()`.

- **`TypedStructure` (ABC)` / `GenericTypedStructure`** — adds a particle
  *type* per vertex: `.particle_type(id)`, `.get_particle_types() -> dict`,
  `.num_particle_types()`. **This is the class actually used as a comparison
  target almost everywhere** — untyped `Structure` alone is rarely what you
  want, since real targets need to distinguish particle species (nanoparticle
  positions, multi-species crystals). `GenericTypedStructure(bindings=...,
  types={...}, fill_type=0)` — `fill_type` backfills any vertex missing an
  explicit entry in `types` (e.g. "everything not listed is type 0"). Extra
  methods: `.to_tlm_topology() -> libtlm.TLMTopology` (direct index
  correspondence with `DirIdx`, no translation needed — see [[tlm-simulations]]),
  `.is_symmetric(axis)`.

- **`StructuralMapping` / `StructuralHomomorphism`** — the result of a
  homomorphism/rotation/vertex-remap operation. `.lmap`/`.rlmap` map location
  indices both directions (`.map_location()`/`.rmap_location()`).
  `StructuralHomomorphism` additionally carries a rotation:
  `.map_direction()`/`.rmap_direction()` map *direction* indices through the
  one rotation (of the 24 octahedral symmetries, see below) this particular
  homomorphism used, and `.as_transform() -> (rotation_matrix, translation)`
  recovers an actual geometric transform via SVD superimposition of the two
  structures' `.matrix()` coordinates.

- **`Tiling(source_structure, periodic_bindings, dimensional_tiling_counts)`** —
  builds a periodic crystal by repeating a unit cell `dimensional_tiling_counts[axis]`
  times along each active axis, splicing copies together at the periodic
  bindings. Requires the source to already be `is_bindings_euclidian()` and
  `is_crystal()` — asserted, not silently handled.

## Homomorphism / isomorphism testing — the load-bearing operation

`a.homomorphism(b) -> StructuralHomomorphism | False` and
`a.homomorphisms(b) -> Generator[StructuralHomomorphism]`: does `a`'s
topology embed into `b` as an injective, direction-consistent subgraph, under
*some* rotation? Implemented via igraph's VF2 subgraph-isomorphism
(`to_igraph()` + `get_subisomorphisms_vf2`), tried once per each of the 24
possible octahedral rotations (`util.enumerateRotations()` — the full
rotation group, indices 0-23) since a real match may only appear after
rotating `a`. `TypedStructure` additionally constrains matches to
same-particle-type vertex pairs via a `node_compat_fn`. `to_igraph()` itself
validates as it converts: every vertex must have 1-6 out-edges, and every
edge must have a reciprocal — a malformed `Structure` (e.g. built from a
one-directional binding list by mistake) fails loudly here rather than
producing a silently-wrong homomorphism search.

**The sharp edge, verified against a real bug callout in
`pypatchy/design/sat/solve_steps.py`'s `homomorphism_is_valid()` docstring**:
`Structure.homomorphisms()` compares directions using the **inverse**
rotation mapping internally. So to translate a direction *from* the target
structure passed into `homomorphisms()` *into* the structure it matched
against, you must call `hm.rmap_direction(d)`, **not** `hm.map_direction(d)`
— using the forward map there silently checks the wrong edges. If a
homomorphism check that "should" pass doesn't (or a bogus one passes), check
which of `map_direction`/`rmap_direction` is being used before suspecting the
topology data itself.

## Topology JSON format (`read_topology` / `export_periodic_topology`)

The `~/.pypatchy/input/topologies/*.json` spec files [[sat-crystal-design]]'s
`load_solve_params()` reads are exactly this shape:
```json
{
  "bindings": [[0, 1, 1, 0], ...],
  "extraConnections": [[3, 3, 0, 2], ...],
  "nanoparticles": {"2": 0}
}
```
`bindings` = intra-unit-cell edges; `extraConnections` = periodic/crystal
edges that only exist in the *infinite* tiling, never in a finite assembled
fragment — `read_topology()` merges both into one `GenericTypedStructure` for
the SAT problem itself, but `Polysat` separately keeps a `local_topology`
(bindings only, no `extraConnections`) specifically for homomorphism checks
against finite TLM-assembled polycubes, which physically cannot contain a
periodic bond. `nanoparticles` maps `{"vertex_id": type - 1}` (offset by one
so type `0`/absent can mean "no nanoparticle" — `read_topology` adds 1 back
on load, `GenericTypedStructure.export_periodic_topology()` subtracts 1 and
omits zero entries on save). `Structure.export_periodic_topology()` derives
which bindings are "internal" vs. "extra" by comparing each edge's endpoints
against integer grid coordinates from `._get_cubes_coords()` — a binding
counts as internal iff `coords[v] == coords[u] + RULE_ORDER[du]` exactly.

## Where this shows up elsewhere

- [[analyze-ensemble]]'s `ClassifyClusters`/`ClassifyPolycubeClusters`
  analysis steps take a `Structure`/`TypedStructure` as their comparison
  target — e.g. the "3D cross" example builds one by hand:
  `Structure(bindings=[(0, i, i+1, rdir(i)) for i in range(6)])`.
- [[sat-crystal-design]]'s entire target-matching machinery
  (`SolveParams.topology`, `Polysat.targets_as_typed`, `matches_target`,
  `NegativeDesignStep`'s off-target structures) is `GenericTypedStructure`
  underneath.
- [[polycubes-rules]]' `PolycubeStructure` — the representation of one
  already-assembled polycube — is itself a `TypedStructure` subclass, so
  everything above (homomorphism checks, `.matrix()`, `.is_crystal()`, JSON
  export) applies to a real assembled structure exactly the same way it
  applies to a hand-built abstract target.
